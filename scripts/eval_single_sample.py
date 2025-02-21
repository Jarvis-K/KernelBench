import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import re

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_custom_cuda_from_prompt_template_reflection
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        # NOTE: this is the logical index (problem id the problem_name)\
        self.kernel_path = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "local"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ['Hopper']

        # Archon config
        self.archon_config_path = None

        self.device_id = 0
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

class KernelAgent:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.ref_arch_src = None

    def get_results(self, ref_arch_src, num_correct_trials=5, num_perf_trials=100, verbose=False):
        # 调用推理服务器生成响应
        response = read_file(self.config.kernel_path)
        # 提取首个代码块（指定语言：python 或 cpp）
        # response = extract_first_code(response, ["python", "cpp"])
        # 评估生成的 CUDA 代码
        result = eval_kernel_against_ref(
            ref_arch_src,
            response,
            verbose=verbose,
            measure_performance=True,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            device = torch.device(f"cuda:{self.config.device_id}")
        )
        result = vars(result)
        return result


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    迭代生成并评估 CUDA 核函数，最终输出最佳结果
    """
    print(f"Starting Eval with config: {config}")
    
    level = int(re.search(r'level_(\d+)', config.kernel_path).group(1))
    problem_id = int(re.search(r'problem_(\d+)', config.kernel_path).group(1))
    # 配置数据集
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(level)

    # import pdb; pdb.set_trace()
    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # 否则为所有架构构建

    os.makedirs(config.logdir, exist_ok=True)
        
    # 问题检查
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {level} Problem {problem_id}")

    assert problem_id <= num_problems, f"Problem ID {problem_id} out of range for Level {level}"

    # 1. 获取问题和参考代码
    if config.dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif config.dataset_src == "local":
        problem_idx_in_dataset = problem_id - 1  # 本地数据集 list 为 0-indexed
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # 从文件名中提取问题编号（如 "1_Square_matrix_multiplication_.py" 提取出 "1"）
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == problem_id, (
        f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
    )
    kernel_agent = KernelAgent(config)
    kernel_agent.ref_arch_src = ref_arch_src
    result = kernel_agent.get_results(ref_arch_src, num_correct_trials=5, num_perf_trials=100, verbose=config.verbose)
    print(result)

if __name__ == "__main__":
    main()

