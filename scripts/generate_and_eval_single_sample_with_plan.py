import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import concurrent.futures
import multiprocessing as mp
import logging
import shutil
from tqdm import tqdm

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref, build_compile_cache
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_custom_cuda_from_prompt_template_reflection
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)
inference_server = None
global_config = None

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "local"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]

        # Archon config
        self.archon_config_path = None

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0
        self.max_iteration = 5
        self.device_id = 0
        self.reflection = True
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs/plan")
        self.kernel_eval_build_dir = os.path.join(REPO_TOP_DIR, "cache")
        self.verbose = False
        self.recent_hist_flag = False
        self.best_hist_flag = False
        self.example_flag = False
        self.plan_flag = True
        self.hist_num = 5
        self.sample_num = 10

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

def remove_cache_dir(cache_dir: str, model_name: str, iteration_num, sample_id, config: EvalConfig):
    """
    Remove the cached folder for sample compilation so it can start a clean build next time
    useful for time out, failed build, etc.
    """
    problem_cache_dir = os.path.join(cache_dir, "eval_single_sample", model_name, f"{config.problem_id}", f"{iteration_num}", f"{sample_id}")
    print(f"cache_dir to remove: {problem_cache_dir}")
    if os.path.exists(problem_cache_dir):
        try:
            shutil.rmtree(problem_cache_dir, ignore_errors=True)
            print(f"\n[INFO] Removed cached folder for Iteration {iteration_num}, Sample ID: {sample_id}")
        except Exception as e:
            print(f"\n[WARNING] Failed to remove cache directory {problem_cache_dir}: {str(e)}")

def compile_kernel(prompt, iteration_num, sample_id):
    try:
        build_dir = os.path.join(global_config.kernel_eval_build_dir, "eval_single_sample", global_config.model_name, f"{global_config.problem_id}", f"{iteration_num}", f"{sample_id}")
        response = inference_server(prompt)
        kernel_src = extract_first_code(response, ["python", "cpp"])
        success, stdout, error = build_compile_cache(kernel_src, global_config.verbose, build_dir)
        if success:
            logging.info(f"Compiled kernel for Iteration {iteration_num}, Sample ID: {sample_id}")
            return (iteration_num, sample_id, response, kernel_src)
        else:
            logging.error(f"Compilation FAILED for Iteration {iteration_num}, Sample ID: {sample_id}: {error}")
            remove_cache_dir(global_config.kernel_eval_build_dir, global_config.model_name, iteration_num, sample_id, global_config)
            return None
    except Exception as e:
        logging.error(f"Exception during compilation for Iteration {iteration_num}, Sample ID: {sample_id}: {e}")
        remove_cache_dir(global_config.kernel_eval_build_dir, global_config.model_name, iteration_num, sample_id, global_config)
        return None

def compile_kernel_wrapper(args):
    prompt, iteration_num, sample_id = args
    try:
        return compile_kernel(prompt, iteration_num, sample_id)
    except Exception as e:
        logging.error(f"Error compiling kernel for Iteration {iteration_num}, Sample ID: {sample_id}: {e}")
        return None

def compile_all_kernels(prompt, iteration_num) -> list[tuple[int, int]]:
    """
    Compile all kernels in parallel before evaluation using a single GPU
    Returns a list of successfully compiled kernels
    """
    successful_compilations = []

    with mp.Pool(processes=global_config.sample_num) as pool:
        async_results = [
            pool.apply_async(compile_kernel_wrapper, ((prompt, iteration_num, sample_id),))
            for sample_id in range(global_config.sample_num)
        ]

        for async_result in tqdm(async_results, total=global_config.sample_num, desc="Compiling Kernels"):
            result = async_result.get()
            if result is not None:
                successful_compilations.append(result)

    return successful_compilations

class KernelAgent:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.init_prompt = None
        self.current_prompt = None
        self.ref_arch_src = None
        self.plan = ""

        self.prompts = []
        self.responses = []
        self.results = []
        self.codes = []

        self.best_prompt = None
        self.best_response = None
        self.best_result = None
        self.best_code = None

        self.iteration_num = 0
        self.output_dir = os.path.join(self.config.logdir, self.config.model_name, f"level_{self.config.level}", f"problem_{self.config.problem_id}")
        self.num_inferences = 3  # 新增：每次迭代生成的推理次数

    def initialize_server(self, inference_server):
        self.inference_server = inference_server
    
    def initialize_plan_server(self, plan_server):
        self.plan_server = plan_server

    def initialize_prompt(self, prompt):
        self.init_prompt = prompt
        self.current_prompt = prompt
        self.prompts.append(prompt)

    def refine_prompt(self):
        first_step_flag = True if self.iteration_num == 0 else False
        example_flag = True if self.iteration_num == 0 else self.config.example_flag
        if self.config.plan_flag and not first_step_flag:
            plan_prompt = prompt_generate_custom_cuda_from_prompt_template_reflection(self.ref_arch_src, example_flag=example_flag, plan_flag=self.config.plan_flag, first_step_flag=first_step_flag, generate_plan_flag=True)
            self.plan = self.plan_server(plan_prompt)

        if self.config.reflection:
            if not self.config.best_hist_flag:
                hist_codes = self.codes[-self.config.hist_num:]
                hist_results = self.results[-self.config.hist_num:]
                improvement_prompt = prompt_generate_custom_cuda_from_prompt_template_reflection(self.ref_arch_src, hist_codes, hist_results, self.config.recent_hist_flag, self.config.best_hist_flag, example_flag=example_flag, plan_flag=self.config.plan_flag, first_step_flag=first_step_flag, generate_plan_flag=False, plan=self.plan)
            else:
                sort_idx = sorted(range(len(self.results)), key=lambda x: self.results[x]['speed_up'])
                hist_codes = [self.codes[i] for i in sort_idx][-self.config.hist_num:]
                hist_results = [self.results[i] for i in sort_idx][-self.config.hist_num:]
                improvement_prompt = prompt_generate_custom_cuda_from_prompt_template_reflection(self.ref_arch_src, hist_codes, hist_results, self.config.recent_hist_flag, self.config.best_hist_flag, example_flag=example_flag, plan_flag=self.config.plan_flag, first_step_flag=first_step_flag, generate_plan_flag=False, plan=self.plan)
        else:
            improvement_prompt = prompt_generate_custom_cuda_from_prompt_template(self.ref_arch_src)

        self.current_prompt = improvement_prompt
        self.prompts.append(self.current_prompt)

    def get_code_and_compile(self):
        compile_results = compile_all_kernels(self.current_prompt, self.iteration_num)
        # import pdb; pdb.set_trace()
        results = []
        for problem_id, sample_id, response, code in compile_results:
            result = eval_kernel_against_ref(
                self.ref_arch_src,
                code,
                verbose=self.config.verbose,
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100,
                device = torch.device(f"cuda:{self.config.device_id}"),
                build_dir=os.path.join(self.config.kernel_eval_build_dir, "eval_single_sample", self.config.model_name, f"{self.config.problem_id}", f"{self.iteration_num}", f"{sample_id}")
            )
            result = vars(result)
            result['iteration'] = self.iteration_num
            result['sample_id'] = sample_id
            results.append(result)
            self.save_results_single_sample(code, response, result, sample_id)
        
        result_idxs = range(len(results))
        if len(result_idxs) == 0:
            return {}, "", ""
        min_runtime_idx = min(result_idxs, key=lambda x: results[x]['runtime'] if results[x]['correctness'] else float('inf'))
        import pdb; pdb.set_trace()
        result = results[min_runtime_idx]
        code = compile_results[min_runtime_idx][2]
        response = compile_results[min_runtime_idx][3]
        return result, code, response

    def get_results(self, ref_arch_src, num_correct_trials=5, num_perf_trials=100, verbose=False):
        result, code, response = self.get_code_and_compile()
        self.results.append(result)
        self.responses.append(response)
        self.codes.append(code)
        self.prompts.append(self.current_prompt)

        if self.best_result is None:
            self.best_prompt = self.current_prompt
            self.best_response = response
            self.best_result = result
            self.best_code = code
            print(f"Best result updated: {self.best_result}")
        else:
            if isinstance(result, dict) and result.get("correctness", True):
                if not (isinstance(self.best_result, dict) and self.best_result.get("correctness", False)):
                    self.best_prompt = self.current_prompt
                    self.best_response = response
                    self.best_result = result
                    self.best_code = code
                    print(f"Best result updated: {self.best_result}")
                else:
                    if "speed_up" in result and "speed_up" in self.best_result:
                        if result["runtime"] < self.best_result["runtime"]:
                            self.best_prompt = self.current_prompt
                            self.best_response = response
                            self.best_result = result
                            self.best_code = code
                            print(f"Best result updated: {self.best_result}")
        self.iteration_num += 1

    def update_speedup(self, ):
        baseline_results = [result for result in self.results if result["correctness"]]
        if len(baseline_results) == 0:
            print("No correct results found, cannot update speedup")
            for i, result in enumerate(self.results):
                self.save_results(i)
        else:
            min_baseline_runtime = min([result["baseline_runtime"] for result in baseline_results])
            for i, result in enumerate(self.results):
                self.results[i]["speed_up"] = max(min_baseline_runtime / result["runtime"], 0.0)
                self.save_results(i)

    def save_results_single_sample(self, code, response, result, sample_id=0):
        os.makedirs(self.output_dir, exist_ok=True)
        
        iteration_dir = os.path.join(self.output_dir, f"iteration_{self.iteration_num}") 
        os.makedirs(iteration_dir, exist_ok=True)
        with open(os.path.join(iteration_dir, f"kernel_code_{sample_id}.py"), "w") as f:
            f.write(code)
        with open(os.path.join(iteration_dir, f"response_{sample_id}.text"), "w") as f:
            f.write(response)
        
        if os.path.exists(os.path.join(iteration_dir, f"result_all.json")):
            with open(os.path.join(iteration_dir, f"result_all.json"), "r") as f:
                result_all = json.load(f)
        else:
            result_all = []
        result_all.append(result)
        with open(os.path.join(iteration_dir, f"result_all.json"), "w") as f:
            json.dump(result_all, f, indent=4)

    def save_results(self, iteration_num=0):
        # 保存多个推理结果
        os.makedirs(self.output_dir, exist_ok=True)
        
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration_num}") 
        os.makedirs(iteration_dir, exist_ok=True)

        for idx, (prompt, response, result, code) in enumerate(zip(self.prompts, self.responses, self.results, self.codes)):
            with open(os.path.join(iteration_dir, f"prompt.txt"), "w") as f:
                f.write(prompt)
            with open(os.path.join(iteration_dir, f"response.py"), "w") as f:
                f.write(response)
            with open(os.path.join(iteration_dir, f"kernel_code.py"), "w") as f:
                f.write(code)
            with open(os.path.join(iteration_dir, f"result.json"), "w") as f:
                json.dump(result, f, indent=4)

        # 保存汇总结果
        eval_results_path = os.path.join(self.output_dir, "eval_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
    def save_best_results(self):
        # 保存最佳结果
        if self.best_result:
            best_dir = os.path.join(self.output_dir, "best_results")
            os.makedirs(best_dir, exist_ok=True)
            with open(os.path.join(best_dir, "best_prompt.txt"), "w") as f:
                f.write(self.best_prompt)
            with open(os.path.join(best_dir, "best_response.py"), "w") as f:
                f.write(self.best_response)
            with open(os.path.join(best_dir, "best_result.json"), "w") as f:
                json.dump(self.best_result, f, indent=4)
            with open(os.path.join(best_dir, "best_code.py"), "w") as f:
                f.write(self.best_code)

    def draw_results(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json

        # 提取所有迭代的结果
        all_results = [result['speed_up'] for result in self.results]
        
        # 绘制结果
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(all_results, marker='o', linestyle='-', color='b')
        plt.xlabel('Iteration')
        plt.ylabel('Speed Up')
        plt.title('Speed Up of Each Iteration')
        plt.savefig(os.path.join(self.output_dir, "speed_up_plot.png"))
        plt.close()

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    迭代生成并评估 CUDA 核函数，最终输出最佳结果
    """
    print(f"Starting Eval with config: {config}")
    global inference_server, global_config
    global_config = config

    # 配置数据集
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # 否则为所有架构构建

    os.makedirs(config.logdir, exist_ok=True)
        
    # 问题检查
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. 获取问题和参考代码
    if config.dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1  # 本地数据集 list 为 0-indexed
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # 从文件名中提取问题编号（如 "1_Square_matrix_multiplication_.py" 提取出 "1"）
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, (
        f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    )
    kernel_agent = KernelAgent(config)

    # 2. 初始化推理服务及初始 prompt
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        archon_config_path=config.archon_config_path,
        time_generation=True
    )
    plan_server = create_inference_server_from_presets(
        # server_type="volcengine",
        # model_name="ep-20250214154957-c777d",
        server_type="pandas",
        model_name="gpt-4o-mini",
        temperature=1.0,
        max_tokens=8192,
        verbose=False,
    )
    kernel_agent.initialize_server(inference_server)
    kernel_agent.initialize_plan_server(plan_server)
    kernel_agent.ref_arch_src = ref_arch_src

    kernel_agent.refine_prompt()

    # 3. 迭代生成和评估
    for i in range(config.max_iteration):
        print(f"Iteration {i+1} / {config.max_iteration}")
        kernel_agent.get_results(ref_arch_src, num_correct_trials=5, num_perf_trials=100, verbose=config.verbose)
        kernel_agent.save_results(iteration_num=i)
        # print(f"Iteration {i+1} Response:\n{kernel_agent.responses[-1]}")
        print(f"Iteration {i+1} Evaluation:\n{kernel_agent.results[-1]}")
        # 如果不是最后一轮则更新 prompt
        if i < config.max_iteration - 1:
            kernel_agent.refine_prompt()

    kernel_agent.update_speedup()
    kernel_agent.save_best_results()

    # print(f"Best prompt:\n{kernel_agent.best_prompt}")
    # print(f"Best response:\n{kernel_agent.best_response}")
    kernel_agent.draw_results()
    print(f"Best evaluation result:\n{kernel_agent.best_result}")

    if kernel_agent.results[0]['speed_up'] < 1.0 and kernel_agent.best_result['speed_up'] > 1.0:
        print(f"There's a huge improvement in problem {config.problem_id}!")
    else:
        print(f"No huge improvement~")

    shutil.rmtree(config.kernel_eval_build_dir, ignore_errors=True)

if __name__ == "__main__":
    main()

