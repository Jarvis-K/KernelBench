import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template_s1
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

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
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False
        self.recent_hist_flag = False
        self.best_hist_flag = False
        self.example_flag = False
        self.hist_num = 5
        self.wait_num = 3

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

class KernelAgent:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.init_prompt = None
        self.current_prompt = None
        self.ref_arch_src = None

        self.prompts = []
        self.responses = []
        self.results = []
        self.codes = []

        self.wait_responses = []
        self.wait_results = []

        self.best_prompts = []
        self.best_responses = []
        self.best_results = []
        self.best_codes = []

        self.best_prompt = None
        self.best_response = None
        self.best_result = None
        self.best_code = None
        self.iteration_num = 0
        self.wait_num = 0
        self.output_dir = os.path.join(self.config.logdir, 's1', self.config.model_name, f"level_{self.config.level}", f"problem_{self.config.problem_id}")

    def initialize_server(self, inference_server):
        self.inference_server = inference_server

    def initialize_prompt(self, prompt):
        self.init_prompt = prompt
        self.current_prompt = prompt
        self.prompts.append(prompt)

    def refine_prompt(self):
        if self.iteration_num == 0:
            improvement_prompt = prompt_generate_custom_cuda_from_prompt_template_s1(self.ref_arch_src, self.best_codes, self.best_results, example_flag=True, wait_responses=self.wait_responses, wait_results=self.wait_results, model=self.config.model_name)
        else:
            improvement_prompt = prompt_generate_custom_cuda_from_prompt_template_s1(self.ref_arch_src, self.best_codes, self.best_results, example_flag=self.config.example_flag, wait_responses=self.wait_responses, wait_results=self.wait_results, model=self.config.model_name)

        self.current_prompt = improvement_prompt
        self.prompts.append(self.current_prompt)

    def get_results(self, ref_arch_src, num_correct_trials=5, num_perf_trials=100, verbose=False):
        # 调用推理服务器生成响应
        response = self.inference_server(self.current_prompt)
        self.wait_responses.append(response)
        # 提取首个代码块（指定语言：python 或 cpp）
        code = extract_first_code(response, ["python", "cpp"])
        # 确保生成了有效的代码
        assert response is not None, f"Custom CUDA code generation failed in iteration {self.iteration_num}"
        self.responses.append(response)
        self.codes.append(code)
        # 评估生成的 CUDA 代码
        result = eval_kernel_against_ref(
            ref_arch_src,
            code,
            verbose=verbose,
            measure_performance=True,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            device = torch.device(f"cuda:{self.config.device_id}")
        )
        result.iteration = self.iteration_num
        result = vars(result)
        result["wait_num"] = self.wait_num
        self.results.append(result)
        self.wait_results.append(result)

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
        self.wait_num += 1
        if self.wait_num >= self.config.wait_num:
            self.wait_num = 0
            self.iteration_num += 1
            self.wait_responses = []
            self.wait_results = []
            
            self.best_responses.append(self.best_response)
            self.best_results.append(self.best_result)
            self.best_prompts.append(self.best_prompt)
            self.best_codes.append(self.best_code)
            self.best_response = None
            self.best_result = None
            self.best_prompt = None
            self.best_code = None

    def update_speedup(self, ):
        baseline_results = [result for result in self.results if result["correctness"]]
        if len(baseline_results) == 0:
            print("No correct results found, cannot update speedup")
            for i, result in enumerate(self.results):
                self.save_results(result["iteration"], result["wait_num"])
        else:
            min_baseline_runtime = min([result["baseline_runtime"] for result in baseline_results])
            for i, result in enumerate(self.results):
                self.results[i]["speed_up"] = max(0.0, min_baseline_runtime / result["runtime"])
                self.save_results(result["iteration"], result["wait_num"])

    def save_results(self, iteration_num=0, wait_num=0):
        # 保存当前迭代的 prompt、response 和 result 到指定目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration_num}_wait_{wait_num}")
        os.makedirs(iteration_dir, exist_ok=True)

        with open(os.path.join(iteration_dir, "prompt.txt"), "w") as f:
            f.write(self.prompts[iteration_num*self.config.wait_num+wait_num])
        with open(os.path.join(iteration_dir, "result.json"), "w") as f:
            json.dump(self.results[iteration_num*self.config.wait_num+wait_num], f, indent=4)
        with open(os.path.join(iteration_dir, "response.txt"), "w") as f:
            f.write(self.responses[iteration_num*self.config.wait_num+wait_num])
        if self.codes[iteration_num*self.config.wait_num+wait_num]:
            with open(os.path.join(iteration_dir, "code.py"), "w") as f:
                f.write(self.codes[iteration_num*self.config.wait_num+wait_num])
        
        # 汇总所有迭代的评估结果到一个 JSON 文件中
        eval_results_path = os.path.join(self.output_dir, "eval_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
        # 保存最佳结果到专门的文件夹中
        best_dir = os.path.join(self.output_dir, "best_results")
        os.makedirs(best_dir, exist_ok=True)
        if self.best_results:
            best_results_idx = self.best_results.index(max(self.best_results, key=lambda x: x["speed_up"]))
            with open(os.path.join(best_dir, "best_prompt.txt"), "w") as f:
                f.write(self.best_prompts[best_results_idx])
            with open(os.path.join(best_dir, "best_response.txt"), "w") as f:
                f.write(self.best_responses[best_results_idx])
            with open(os.path.join(best_dir, "best_result.json"), "w") as f:
                json.dump(self.best_results[best_results_idx], f, indent=4)

            if self.best_codes[best_results_idx]:
                with open(os.path.join(best_dir, "best_code.py"), "w") as f:
                    f.write(self.best_codes[best_results_idx])

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

    def get_best_result(self,):
        best_results_idx = self.best_results.index(max(self.best_results, key=lambda x: x["speed_up"]))
        return self.best_results[best_results_idx]


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    迭代生成并评估 CUDA 核函数，最终输出最佳结果
    """
    print(f"Starting Eval with config: {config}")
    

    # 配置数据集
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    # if config.gpu_arch:
    #     set_gpu_arch(config.gpu_arch)  # 否则为所有架构构建

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
    kernel_agent.initialize_server(inference_server)
    kernel_agent.ref_arch_src = ref_arch_src

    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template_s1(ref_arch_src, example_flag=True)
    kernel_agent.initialize_prompt(custom_cuda_prompt)

    # 3. 迭代生成和评估
    for i in range(config.max_iteration):
        print(f"Iteration {i+1} / {config.max_iteration}")
        for j in range(config.wait_num):
            kernel_agent.get_results(ref_arch_src, num_correct_trials=5, num_perf_trials=100, verbose=config.verbose)
            kernel_agent.save_results(iteration_num=i, wait_num=j)
            # print(f"Iteration {i+1} Response:\n{kernel_agent.responses[-1]}")
            print(f"Iteration {i+1} wait {j} Evaluation:\n{kernel_agent.results[-1]}")
            kernel_agent.refine_prompt()

    kernel_agent.update_speedup()

    # print(f"Best prompt:\n{kernel_agent.best_prompt}")
    # print(f"Best response:\n{kernel_agent.best_response}")
    kernel_agent.draw_results()
    
    best_result = kernel_agent.get_best_result()
    print(f"Best evaluation result:\n{best_result}")
    if kernel_agent.results[0]['speed_up'] < 1.0 and best_result['speed_up'] > 1.0:
        print(f"There's a huge improvement in problem {config.problem_id}!")
    else:
        print(f"No huge improvement~")

if __name__ == "__main__":
    main()

