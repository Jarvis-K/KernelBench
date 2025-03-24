import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import concurrent.futures
import multiprocessing as mp
import subprocess
import logging
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import queue
import signal
import re
import random

from datasets import load_dataset
import time

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref, build_compile_cache, run_kernel_with_ncu
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_custom_cuda_from_prompt_template_reflection, prompt_generate_plan_evaluation, parser_plan_evaluation, prompt_generate_custom_cuda_triton, prompt_generate_custom_cuda_persuado
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets
from evaluate_kernel import evaluate_kernel_isolated

# 设置环境变量，限制只使用GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用GPU 1

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
        self.gpu_arch = ['Hopper']

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
        self.run_dir = os.path.join(REPO_TOP_DIR, "runs/")
        self.kernel_eval_build_dir = os.path.join(REPO_TOP_DIR, "cache")
        self.verbose = False
        self.recent_hist_flag = False
        self.best_hist_flag = False
        self.example_flag = False
        self.plan_flag = True
        self.hist_num = 5
        self.sample_num = 10
        self.prompt_file = None
        self.given_example_result = None
        self.given_example_code = None
        self.plan_num = 1
        self.plan_server_type = "pandas"
        self.plan_model_name = "gpt-4o-mini"
        self.use_ncu = False
        self.save_best_code = False
        self.generate_triton_code = False
        self.generate_persuado_code = False
    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

def get_valuable_result_from_error(results):
    valuable_results = []
    regenerate_persuado_code = False
    for result in results:
        if 'metadata' in result and 'correctness_issue' in result['metadata']:
            valuable_results.append(result)
            regenerate_persuado_code = True
    if len(valuable_results) == 0:
        for result in results:
            valid_compile_results = [result for result in results if 'metadata' in result and 'CUDA error' not in str(result['metadata']) and result['metadata'] != 'runtime_error']
            if valid_compile_results:
                result = min(valid_compile_results, key=lambda result: len(str(result['metadata'])))
                regenerate_persuado_code = True
            else:
                result = random.choice(results)
                regenerate_persuado_code = True
    else:
        result = valuable_results[0]
    return result, regenerate_persuado_code

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

# 添加一个可序列化的推理服务器类
class SerializableInferenceServer:
    def __init__(self, server_type, model_name, temperature, max_tokens, verbose=False, 
                 archon_config_path=None, time_generation=False):
        self.server_type = server_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.archon_config_path = archon_config_path
        self.time_generation = time_generation
        # 在初始化时不创建实际的服务器函数
        self._server_fn = None
    
    def __call__(self, prompt):
        # 延迟初始化，在第一次调用时创建服务器函数
        if self._server_fn is None:
            self._server_fn = create_inference_server_from_presets(
                server_type=self.server_type,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
                archon_config_path=self.archon_config_path,
                time_generation=self.time_generation
            )
        # 调用实际的服务器函数
        return self._server_fn(prompt)

# 添加一个可序列化的计划服务器类
class SerializablePlanServer:
    def __init__(self, server_type, model_name, temperature, max_tokens, verbose=False, 
                 archon_config_path=None, time_generation=False):
        self.server_type = server_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.archon_config_path = archon_config_path
        self.time_generation = time_generation
        # 在初始化时不创建实际的服务器函数
        self._server_fn = None
    
    def __call__(self, prompt):
        # 延迟初始化，在第一次调用时创建服务器函数
        if self._server_fn is None:
            self._server_fn = create_inference_server_from_presets(
                server_type=self.server_type,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
                archon_config_path=self.archon_config_path,
                time_generation=self.time_generation
            )
        # 调用实际的服务器函数
        return self._server_fn(prompt)

# 修改compile_kernel_wrapper函数，不再传递inference_server_fn
def compile_kernel_wrapper(args):
    prompt, iteration_num, sample_id, config = args
    kernel_src = ""
    original_code = ""
    try:
        build_dir = os.path.join(config.kernel_eval_build_dir, "eval_single_sample", config.model_name, f"{config.problem_id}", f"{iteration_num}", f"{sample_id}")
        # 在这里创建推理服务器
        inference_server = SerializableInferenceServer(
            server_type=config.server_type,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            verbose=config.verbose,
            archon_config_path=config.archon_config_path,
            time_generation=True
        )
        response, tokens = inference_server(prompt)
        kernel_src, original_code, _ = extract_first_code(response, ["python", "cpp"])
        success, stdout, error = build_compile_cache(kernel_src, config.verbose, build_dir)

        parsered_all_output = list(set(re.findall(r"(error: .*?)\n", stdout, re.DOTALL | re.MULTILINE)))
        warnings = list(set(re.findall(r"(warning: .*?)\n", stdout, re.DOTALL | re.MULTILINE)))
        stdout = '\n'.join(warnings+parsered_all_output) if len(warnings+parsered_all_output) > 0 else stdout
        if success:
            logging.info(f"Compiled kernel for Iteration {iteration_num}, Sample ID: {sample_id}")
            return (iteration_num, sample_id, response, kernel_src, tokens, original_code, True, stdout)
        else:
            logging.error(f"Compilation FAILED for Iteration {iteration_num}, Sample ID: {sample_id}: {error}")
            remove_cache_dir(config.kernel_eval_build_dir, config.model_name, iteration_num, sample_id, config)
            return (iteration_num, sample_id, response, kernel_src, tokens, original_code, False, stdout)
    except Exception as e:
        logging.error(f"Exception during compilation for Iteration {iteration_num}, Sample ID: {sample_id}: {e}")
        remove_cache_dir(config.kernel_eval_build_dir, config.model_name, iteration_num, sample_id, config)
        return (iteration_num, sample_id, response, kernel_src, tokens, original_code, False, str(e))

# 修改compile_all_kernels函数，不再传递inference_server_fn
def compile_all_kernels(prompt, iteration_num, config) -> list[tuple[int, int]]:
    """
    Compile all kernels in parallel before evaluation using a single GPU
    Returns a list of successfully compiled kernels
    """
    successful_compilations = []

    with mp.Pool(processes=config.sample_num) as pool:
        async_results = []
        for sample_id in range(config.sample_num):
            args = (prompt, iteration_num, sample_id, config)  # 不再传递inference_server_fn
            async_result = pool.apply_async(compile_kernel_wrapper, (args,))
            async_results.append((sample_id, async_result))

        for sample_id, async_result in tqdm(async_results, total=config.sample_num, desc="Compiling Kernels"):
            try:
                result = async_result.get(timeout=1200)  # 60秒超时
                if result is not None:
                    successful_compilations.append(result)
            except mp.TimeoutError:
                logging.error(f"Compilation timed out for sample {sample_id}")
                remove_cache_dir(config.kernel_eval_build_dir, config.model_name, iteration_num, sample_id, config)
            except Exception as e:
                logging.error(f"Error compiling kernel for sample {sample_id}: {e}")
                remove_cache_dir(config.kernel_eval_build_dir, config.model_name, iteration_num, sample_id, config)

    return successful_compilations

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("评估超时")

# 将worker函数移到全局作用域
def isolated_worker(func, args, result_queue):
    """在隔离的进程中运行函数并处理异常"""
    try:
        # 设置CUDA相关环境变量
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        
        # 运行目标函数
        result = func(*args)
        
        # 将结果放入队列
        result_queue.put(("success", result))
    except Exception as e:
        # 捕获异常并放入队列
        error_str = str(e)
        result_queue.put(("error", error_str))
        
        # 对CUDA错误进行特殊处理
        if "CUDA" in error_str or "cuda" in error_str or "illegal memory access" in error_str:
            try:
                # 尝试恢复GPU状态
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    device_id = 0  # 假设使用设备0
                    torch.cuda.reset_peak_memory_stats(device_id)
                    torch.cuda.synchronize(device_id)
                time.sleep(5)  # 给GPU一些恢复时间
            except:
                time.sleep(5)
                pass

def run_isolated_process(func, args, timeout=120):
    """
    在完全隔离的进程中运行函数，确保CUDA异常不会影响主进程
    使用进程间通信来传递结果
    """
    # 创建一个队列用于进程间通信
    result_queue = mp.Queue()
    
    # 创建并启动隔离进程，使用全局worker函数
    process = mp.Process(target=isolated_worker, args=(func, args, result_queue))
    process.daemon = True  # 设置为守护进程，确保主进程退出时它也会退出
    process.start()
    
    # 使用超时机制等待结果
    try:
        # 等待结果，设置超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            status, result = result_queue.get()
            # 取消超时信号
            signal.alarm(0)
            
            # 等待进程结束
            process.join(1)
            if process.is_alive():
                # 如果进程仍然存活，强制终止
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
            
            if status == "success":
                return result
            else:
                raise Exception(result)
        except (mp.queues.Empty, queue.Empty):
            # 队列为空，可能是进程崩溃
            raise TimeoutError("进程可能崩溃")
        finally:
            # 确保取消超时信号
            signal.alarm(0)
            
    except TimeoutError:
        # 超时处理
        logging.error("评估超时，正在终止进程...")
        
        # 终止进程
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
        
        # 清理GPU状态
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(0)
            time.sleep(2)  # 给GPU一些恢复时间
        except Exception as e:
            time.sleep(2)
            logging.warning(f"清理GPU状态失败: {str(e)}")
        
        raise TimeoutError("评估超时")

class KernelAgent:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.init_prompt = None
        self.current_prompt = None
        self.ref_arch_src = None
        self.plan = ""
        self.plan_tokens = 0

        self.prompts = []
        self.responses = []
        self.results = []
        self.codes = []
        self.original_codes = []

        self.best_prompt = None
        self.best_response = None
        self.best_result = None
        self.best_code = None
        self.plan_prompt = ""

        self.iteration_num = 0
        self.output_dir = os.path.join(self.config.logdir, self.config.model_name, f"level_{self.config.level}", f"problem_{self.config.problem_id}")
        self.num_inferences = 3  # 新增：每次迭代生成的推理次数
        # 创建进程池，在整个生命周期内重用
        self.process_pool = ProcessPoolExecutor(max_workers=3)  # 限制并发数量
        self.plan_num = 5
        self.ncu_rule_descriptions = None
        self.triton_code = None
        self.persuado_code = None
        self.regenerate_persuado_code = False

    def __del__(self):
        # 确保进程池在对象销毁时关闭
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown()

    def initialize_server(self, inference_server):
        self.inference_server = inference_server
    
    def initialize_plan_server(self, plan_server):
        self.plan_server = plan_server
    
    def initialize_plan_evaluator(self, plan_evaluator):
        self.plan_evaluator = plan_evaluator

    def initialize_prompt(self, prompt):
        self.init_prompt = prompt
        self.current_prompt = prompt
        self.prompts.append(prompt)

    def get_triton_code(self):
        prompt = prompt_generate_custom_cuda_triton(self.ref_arch_src)
        code, tokens = self.plan_server(prompt)
        triton_code_dir = os.path.join(self.config.logdir, self.config.model_name, f"level_{self.config.level}", f"problem_{self.config.problem_id}")
        if not os.path.exists(triton_code_dir):
            os.makedirs(triton_code_dir, exist_ok=True)
        with open(os.path.join(triton_code_dir, "triton_code.py"), "w") as f:
            f.write(code)
        return code
    
    def get_persuado_code(self, first_step_flag=True, prev_cuda_code=None, prev_result=None, prev_persuado_code=None):
        prompt = prompt_generate_custom_cuda_persuado(self.ref_arch_src, first_step_flag=first_step_flag, prev_cuda_code=prev_cuda_code, prev_result=prev_result, prev_persuado_code=prev_persuado_code)
        print(prompt)
        code, tokens = self.plan_server(prompt)
        persuado_code_dir = os.path.join(self.config.logdir, self.config.model_name, f"level_{self.config.level}", f"problem_{self.config.problem_id}")
        if not os.path.exists(persuado_code_dir):
            os.makedirs(persuado_code_dir, exist_ok=True)
        with open(os.path.join(persuado_code_dir, f"persuado_code_{self.iteration_num}.txt"), "w") as f:
            f.write(code)
        return code

    def generate_plans_in_parallel(self, plan_prompt, num_plans=5):
        """
        并行生成多个计划
        
        Args:
            plan_prompt: 用于生成计划的提示
            num_plans: 要生成的计划数量
            
        Returns:
            生成的计划列表和对应的token数量
        """
        plans = []
        tokens_list = []
        
        # 准备工作进程的参数
        args_list = [(
            plan_prompt, 
            self.config.plan_server_type, 
            self.config.plan_model_name, 
            self.config.temperature, 
            self.config.max_tokens, 
            self.config.verbose, 
            self.config.archon_config_path, 
            60
        ) for _ in range(num_plans)]
        
        # 使用进程池并行生成计划
        with ProcessPoolExecutor(max_workers=num_plans) as executor:
            futures = [executor.submit(generate_plan_worker, *args) for args in args_list]
            
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_plans, desc="生成计划"):
                try:
                    plan, tokens = future.result()
                    plans.append(plan)
                    tokens_list.append(tokens)
                except Exception as e:
                    logging.error(f"获取计划结果时出错: {str(e)}")
                    # 如果某个计划生成失败，添加一个空计划
                    plans.append("")
                    tokens_list.append(0)
        
        return plans, tokens_list

    def evaluate_and_integrate_plans(self, arc_src, kernel_src, plans):
        """
        评估多个计划并整合为一个最终计划
        
        Args:
            plans: 要评估的计划列表
            
        Returns:
            整合后的最终计划和使用的token数量
        """
        if not plans or all(not plan for plan in plans):
            logging.warning("没有有效的计划可评估")
            return "", 0
        
        # 构建评估提示
        eval_prompt = prompt_generate_plan_evaluation(arc_src, kernel_src, plans)
        
        # 使用评估服务器评估计划
        final_plan, tokens = self.plan_evaluator(eval_prompt)
        plan_path = os.path.join(self.config.logdir, self.config.model_name, f"level_{self.config.level}", f"problem_{self.config.problem_id}", f"iteration_{self.iteration_num}")
        if not os.path.exists(plan_path):
            os.makedirs(plan_path, exist_ok=True)
        with open(os.path.join(plan_path, f"plan_evaluation.txt"), "w") as f:
            f.write(final_plan)
        final_plan = parser_plan_evaluation(final_plan)

        return final_plan, tokens

    def refine_prompt(self):
        self.triton_code = self.get_triton_code() if self.config.generate_triton_code and self.triton_code is None else self.triton_code
        
        first_step_flag = True if self.iteration_num == 0 else False
        example_flag = True if self.iteration_num == 0 else self.config.example_flag
        if self.config.prompt_file is not None and os.path.exists(self.config.prompt_file) and first_step_flag:
            with open(self.config.prompt_file, "r") as f:
                improvement_prompt = f.read()
            self.init_prompt = improvement_prompt
        else:
            if self.config.given_example_code is not None and os.path.exists(self.config.given_example_code) and first_step_flag:
                with open(self.config.given_example_code, "r") as f:
                    given_example_code = f.read()
                hist_codes = [given_example_code]
            else:
                hist_codes = []
            if self.config.given_example_result is not None and os.path.exists(self.config.given_example_result) and first_step_flag:
                with open(self.config.given_example_result, "r") as f:
                    given_example_result = json.load(f)
                hist_results = [given_example_result]
            else:
                hist_results = []

            if not self.config.best_hist_flag:
                hist_codes = hist_codes + self.original_codes[-self.config.hist_num:]
                hist_results = hist_results + self.results[-self.config.hist_num:]
            else:
                sort_idx = sorted(range(len(self.results)), key=lambda x: self.results[x]['speed_up'] if 'speed_up' in self.results[x] else -100.0)
                hist_codes = hist_codes + [self.original_codes[i] for i in sort_idx][-self.config.hist_num:]
                hist_results = hist_results + [self.results[i] for i in sort_idx][-self.config.hist_num:]

            if first_step_flag:
                self.persuado_code = self.get_persuado_code(first_step_flag) if self.config.generate_persuado_code else self.persuado_code
            else:
                self.persuado_code = self.get_persuado_code(first_step_flag, hist_codes[0], hist_results[0], self.persuado_code) if self.config.generate_persuado_code and self.regenerate_persuado_code else self.persuado_code
            if self.config.plan_flag and not first_step_flag:
                # 生成计划提示
                plan_prompt = prompt_generate_custom_cuda_from_prompt_template_reflection(
                    self.ref_arch_src, hist_codes, hist_results, 
                    self.config.recent_hist_flag, self.config.best_hist_flag, 
                    example_flag=example_flag, plan_flag=self.config.plan_flag, 
                    first_step_flag=first_step_flag, generate_plan_flag=True, plan=self.plan, ncu_rule_descriptions=self.ncu_rule_descriptions, triton_code=self.triton_code, persuado_code=self.persuado_code
                )
                
                # 并行生成多个计划
                plans, plan_tokens_list = self.generate_plans_in_parallel(plan_prompt, num_plans=self.config.plan_num)
                
                # 评估和整合计划
                self.plan, plan_tokens = self.evaluate_and_integrate_plans(self.ref_arch_src, hist_codes[-1], plans) if self.config.plan_num > 1 else (plans[0], plan_tokens_list[0])
                self.plan_tokens = plan_tokens + sum(plan_tokens_list)
                self.plan_prompt = plan_prompt
            
            if self.config.reflection:
                if not self.config.best_hist_flag:
                    improvement_prompt = prompt_generate_custom_cuda_from_prompt_template_reflection(self.ref_arch_src, hist_codes, hist_results, self.config.recent_hist_flag, self.config.best_hist_flag, example_flag=example_flag or self.regenerate_persuado_code, plan_flag=self.config.plan_flag, first_step_flag=first_step_flag, generate_plan_flag=False, plan=self.plan, ncu_rule_descriptions=self.ncu_rule_descriptions, triton_code=self.triton_code, persuado_code=self.persuado_code)
                else:
                    improvement_prompt = prompt_generate_custom_cuda_from_prompt_template_reflection(self.ref_arch_src, hist_codes, hist_results, self.config.recent_hist_flag, self.config.best_hist_flag, example_flag=example_flag or self.regenerate_persuado_code, plan_flag=self.config.plan_flag, first_step_flag=first_step_flag, generate_plan_flag=False, plan=self.plan, ncu_rule_descriptions=self.ncu_rule_descriptions, triton_code=self.triton_code, persuado_code=self.persuado_code)
            else:
                improvement_prompt = prompt_generate_custom_cuda_from_prompt_template(self.ref_arch_src)

        self.current_prompt = improvement_prompt
        self.prompts.append(self.current_prompt)

    def get_code_and_compile(self):
        compile_results = compile_all_kernels(self.current_prompt, self.iteration_num, self.config)
        results = []
        
        # 创建一个字典来存储每个sample_id的评估结果
        sample_results = {}
        
        # 准备评估参数
        eval_args = []
        for cr in compile_results:
            problem_id, sample_id, response, code, tokens, original_code, successful_flag, error = cr
            if successful_flag:
                build_dir = os.path.join(
                    self.config.kernel_eval_build_dir, 
                    "eval_single_sample", 
                    self.config.model_name, 
                    f"{self.config.problem_id}",
                    f"{self.iteration_num}",
                    f"{sample_id}"
                )
                eval_args.append([
                    self.ref_arch_src,
                    code,
                    build_dir,
                    0,  # 使用设备0，因为我们已经设置了CUDA_VISIBLE_DEVICES
                    self.config.verbose,
                    3,
                    30,
                    True if self.config.use_ncu and self.config.sample_num == 1 else False,
                    sample_id
                ])
        
        # 使用进程池进行评估，每个评估任务在独立进程中运行
        futures = []
        
        # 为每个评估任务创建一个新的进程池，确保完全隔离
        for eval_arg in eval_args:
            sample_id = eval_arg[-1]
            eval_arg_without_id = eval_arg[:-1]
            
            try:
                result = run_isolated_process(evaluate_kernel_isolated, eval_arg_without_id, timeout=300)
                result_dict = vars(result)
                result_dict.update({
                    'sample_id': sample_id,
                    'plan_tokens': self.plan_tokens,
                    'inference_tokens': 0
                })
                sample_results[sample_id] = result_dict
            except TimeoutError:
                error_result = {
                    "iteration": self.iteration_num,
                    "compiled": True,
                    "correctness": False,
                    "runtime": 0.0,
                    "baseline_runtime": 0.0,
                    "speed_up": 0.0,
                    "error": "runtime_error",
                    "metadata": "runtime_error",
                    "sample_id": sample_id,
                    "plan_tokens": self.plan_tokens
                }
                sample_results[sample_id] = error_result
            except Exception as e:
                error_str = str(e)
                logging.error(f"评估失败 (sample_id={sample_id}): {error_str}")
                error_result = {
                    "iteration": self.iteration_num,
                    "compiled": True,
                    "correctness": False,
                    "runtime": 0.0,
                    "baseline_runtime": 0.0,
                    "speed_up": 0.0,
                    "error": error_str,
                    "metadata": f"{error_str}",
                    "sample_id": sample_id,
                    "plan_tokens": self.plan_tokens
                }
                sample_results[sample_id] = error_result
            
            # 在每次评估后清理GPU状态
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
            except Exception as e:
                logging.warning(f"清理GPU状态失败: {str(e)}")
        
        # 处理评估结果
        for idx, cr in enumerate(compile_results):
            problem_id, sample_id, response, code, tokens, original_code, successful_flag, error = cr
            
            if not successful_flag:
                # 编译失败的样本
                code = "" if code is None else code
                response = "" if response is None else response
                result = {
                    "iteration": self.iteration_num,
                    "compiled": False,
                    "correctness": False,
                    "sample_id": sample_id,
                    "plan_tokens": self.plan_tokens,
                    "inference_tokens": tokens,
                    "runtime": -1.0,
                    "baseline_runtime": 0.0,
                    "speed_up": 0.0,
                    "metadata": f"{error}"
                }
            else:
                # 编译成功的样本，使用存储的评估结果
                
                if sample_id in sample_results:
                    result = sample_results[sample_id]
                    # 更新token信息
                    result.update({
                        'iteration': self.iteration_num,
                        'sample_id': sample_id,
                        'plan_tokens': self.plan_tokens,
                        'inference_tokens': tokens
                    })
                else:
                    # 这种情况不应该发生，但为了健壮性添加
                    result = {
                        "iteration": self.iteration_num,
                        "compiled": True,
                        "correctness": False,
                        "sample_id": sample_id,
                        "plan_tokens": self.plan_tokens,
                        "inference_tokens": tokens,
                        "runtime": -1.0,
                        "baseline_runtime": 0.0,
                        "speed_up": 0.0,
                        "error": "未找到评估结果",
                        "metadata": f"{error}"
                    }
            
            results.append(result)
            self.save_results_single_sample(code, response, result, sample_id)
        
        # 找出最佳结果
        result_idxs = range(len(results))
        if len(result_idxs) == 0:
            return None, None, None, None
        
        correct_results = [result for result in results if 'correctness' in result and result['correctness']]
        if len(correct_results) == 0:
            # compiled_results = [result for result in results if 'compiled' in result and result['compiled']]
            result, self.regenerate_persuado_code = get_valuable_result_from_error(results)
        else:
            result = min(results, key=lambda result: result['runtime'] if 'correctness' in result and result['correctness'] else float('inf'))
            min_baseline_runtime = min([_result["baseline_runtime"] for _result in correct_results])
            result["speed_up"] = max(min_baseline_runtime / result["runtime"], 0.0)
            self.regenerate_persuado_code = False
        result_id = result['sample_id']
        best_compile_result = [compile_result for compile_result in compile_results if compile_result[1]==result_id][0]
        if 'correctness' in result and result['correctness'] and self.config.use_ncu:
            device = torch.device(f"cuda:{self.config.device_id}")
            build_dir = os.path.join(
                    self.config.kernel_eval_build_dir, 
                    "eval_single_sample", 
                    self.config.model_name, 
                    f"{self.config.problem_id}",
                    f"{self.iteration_num}",
                    f"{result['sample_id']}"
                )
            ncu_log_path = os.path.join(
                self.output_dir,
                f"iteration_{self.iteration_num}",
                f"ncu_logs",
            )
            if not os.path.exists(ncu_log_path):
                os.makedirs(ncu_log_path, exist_ok=True)
            self.ncu_rule_descriptions = run_kernel_with_ncu(self.ref_arch_src, best_compile_result[3], build_dir=build_dir, device=device, ncu_log_path=ncu_log_path)
        else:
            self.ncu_rule_descriptions = ""
        result['ncu_rule_descriptions'] = self.ncu_rule_descriptions
        code = best_compile_result[3]
        response = best_compile_result[2]
        original_code = best_compile_result[5]
        return result, code, response, original_code

    def get_results(self, ref_arch_src, num_correct_trials=5, num_perf_trials=100, verbose=False):
        result, code, response, original_code = self.get_code_and_compile()
        self.results.append(result)
        self.responses.append(response)
        self.codes.append(code)
        self.original_codes.append(original_code)

        if self.best_result is None:
            self.best_prompt = self.current_prompt
            self.best_response = response
            self.best_result = result
            self.best_code = code
            print(f"Best result updated: {self.best_result}")
        else:
            if isinstance(result, dict) and result.get("correctness", False):
                if not (isinstance(self.best_result, dict) or not self.best_result.get("correctness", False)):
                    self.best_prompt = self.current_prompt
                    self.best_response = response
                    self.best_result = result
                    self.best_code = code
                    print(f"Best result updated: {self.best_result}")
                else:
                    if "speed_up" in result and "speed_up" in self.best_result:
                        if result["runtime"] < self.best_result["runtime"] or self.best_result["runtime"] == -1.0:
                            self.best_prompt = self.current_prompt
                            self.best_response = response
                            self.best_result = result
                            self.best_code = code
                            print(f"Best result updated: {self.best_result}")
        self.iteration_num += 1

    def update_speedup(self, ):
        baseline_results = [result for result in self.results if "correctness" in result and result["correctness"]]
        if len(baseline_results) == 0:
            print("No correct results found, cannot update speedup")
            # for i, result in enumerate(self.results):
                # self.save_results(i)
        else:
            min_baseline_runtime = min([result["baseline_runtime"] for result in baseline_results])
            for i, result in enumerate(self.results):
                self.results[i]["speed_up"] = max(min_baseline_runtime / result["runtime"], 0.0)
                # self.save_results(i)

    def save_results_single_sample(self, code, response, result, sample_id=0):
        os.makedirs(self.output_dir, exist_ok=True)
        
        iteration_dir = os.path.join(self.output_dir, f"iteration_{self.iteration_num}") 
        os.makedirs(iteration_dir, exist_ok=True)
        with open(os.path.join(iteration_dir, f"kernel_code_{sample_id}.py"), "w") as f:
            f.write(code)
        with open(os.path.join(iteration_dir, f"response_{sample_id}.txt"), "w") as f:
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

        with open(os.path.join(iteration_dir, "prompt.txt"), "w") as f:
            f.write(self.prompts[iteration_num])
        with open(os.path.join(iteration_dir, "response.txt"), "w") as f:
            f.write(self.responses[iteration_num])
        with open(os.path.join(iteration_dir, "result.json"), "w") as f:
            json.dump(self.results[iteration_num], f, indent=4)
        with open(os.path.join(iteration_dir, "kernel_code.py"), "w") as f:
            f.write(self.codes[iteration_num])
        with open(os.path.join(iteration_dir, "plan_prompt.txt"), "w") as f:
            f.write(self.plan_prompt)

        # 保存汇总结果
        eval_results_path = os.path.join(self.output_dir, "eval_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
    def save_best_results(self):
        # 保存最佳结果
        run_dir = os.path.join(self.config.run_dir, f"test_hf_level_{self.config.level}/{self.config.model_name}/")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
        if self.best_result:
            best_dir = os.path.join(self.output_dir, "best_results")
            os.makedirs(best_dir, exist_ok=True)
            with open(os.path.join(best_dir, "best_prompt.txt"), "w") as f:
                f.write(self.best_prompt)
            with open(os.path.join(best_dir, "best_response.txt"), "w") as f:
                f.write(self.best_response)
            with open(os.path.join(best_dir, "best_result.json"), "w") as f:
                json.dump(self.best_result, f, indent=4)
            with open(os.path.join(best_dir, "best_code.py"), "w") as f:
                f.write(self.best_code)

            if self.config.save_best_code:
                with open(os.path.join(run_dir, f"level_{self.config.level}_problem_{self.config.problem_id}_sample_0_kernel.py"), "w") as f:
                    f.write(self.best_code)

    def draw_results(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json

        # 提取所有迭代的结果，如果speedup不在result中，则为0.0
        all_results = [result['speed_up'] if 'speed_up' in result else 0.0 for result in self.results]
        
        # 绘制结果
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(all_results, marker='o', linestyle='-', color='b')
        plt.xlabel('Iteration')
        plt.ylabel('Speed Up')
        plt.title('Speed Up of Each Iteration')
        
        # 设置横坐标为从0开始的整数
        plt.xticks(range(len(all_results)))
        
        plt.savefig(os.path.join(self.output_dir, "speed_up_plot.png"))
        plt.close()

# 添加工作进程函数，用于生成单个计划
def generate_plan_worker(prompt, server_type, model_name, temperature, max_tokens, verbose=False, archon_config_path=None, time_generation=False):
    """
    工作进程函数，用于生成单个计划
    """
    try:
        # 创建一个新的服务器实例
        server = SerializablePlanServer(
            server_type=server_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            archon_config_path=archon_config_path,
            time_generation=time_generation
        )
        # 生成计划
        return server(prompt)
    except Exception as e:
        logging.error(f"生成计划时出错: {str(e)}")
        return "", 0

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    迭代生成并评估 CUDA 核函数，最终输出最佳结果
    """
    print(f"Starting Eval with config: {config}")
    global global_config
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
    # 如果存在结果文件夹，先删除
    if os.path.exists(kernel_agent.output_dir):
        shutil.rmtree(kernel_agent.output_dir, ignore_errors=True)

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
    
    # 计划生成服务器
    plan_server = create_inference_server_from_presets(
        server_type=config.plan_server_type,
        model_name=config.plan_model_name,
        temperature=1.0 if config.plan_num > 1 else 0.1,
        max_tokens=8192,
        verbose=False,
    )
    
    # 计划评估服务器
    plan_evaluator = create_inference_server_from_presets(
        server_type=config.plan_server_type,
        model_name=config.plan_model_name,
        temperature=config.temperature,  # 评估时使用较低的温度
        max_tokens=8192,
        verbose=False,
    )
    
    kernel_agent.initialize_server(inference_server)
    kernel_agent.initialize_plan_server(plan_server)
    kernel_agent.initialize_plan_evaluator(plan_evaluator)
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

    # kernel_agent.update_speedup()
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

    # 强制使用设备0，因为我们已经设置了CUDA_VISIBLE_DEVICES
    config.device_id = 0

if __name__ == "__main__":
    main()

