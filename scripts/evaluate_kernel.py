import multiprocessing
import concurrent.futures
import torch
import logging
from src.eval import eval_kernel_against_ref
import json
import argparse
import os

# 设置多进程启动方法为 'spawn'
multiprocessing.set_start_method('spawn', force=True)

os.environ["TORCH_CUDA_ARCH_LIST"] = "Hopper"

def evaluate_kernel(ref_arch_src, code, build_dir, device_id, verbose, num_correct_trials, num_perf_trials):
    # 在每个进程中初始化 CUDA 设备
    device = torch.device(f"cuda:{device_id}")
    return eval_kernel_against_ref(
        ref_arch_src,
        code,
        verbose=True,
        measure_performance=True,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        device=device,
        build_dir=build_dir
    )

def evaluate_kernel_wrapper(
    ref_arch_src,       # 参数1
    code,               # 参数2
    build_dir,          # 参数3
    device_id,          # 参数4
    verbose,            # 参数5
    num_correct_trials, # 参数6
    num_perf_trials,    # 参数7
    timeout=60          # 参数8: 超时时间，单位为秒
):

    try:
        # 使用 ProcessPoolExecutor 来处理超时
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(
                evaluate_kernel,
                ref_arch_src,
                code,
                build_dir,
                device_id,
                verbose,
                num_correct_trials,
                num_perf_trials
            )
            # 等待结果，设置超时时间
            result = future.result(timeout=timeout)
        
        # 转换结果并返回
        return vars(result)
    except concurrent.futures.TimeoutError:
        logging.error("Evaluation timed out")
        return {
            "iteration": -1,
            "compiled": True,
            "correctness": False,
            "runtime": float('inf'),
            "baseline_runtime": 0.0,
            "speed_up": 0.0,
            "error": "Evaluation timed out"
        }
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        return {
            "iteration": -1,
            "compiled": True,
            "correctness": False,
            "runtime": float('inf'),
            "baseline_runtime": 0.0,
            "speed_up": 0.0,
            "error": str(e)
        }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate kernel performance.')
    parser.add_argument('--ref_arch_src', type=str, help='Reference architecture source')
    parser.add_argument('--code', type=str, help='Kernel code to evaluate')
    parser.add_argument('--build_dir', type=str, help='Build directory path')
    parser.add_argument('--device_id', type=int, help='CUDA device ID')
    parser.add_argument('--verbose', type=int, help='Verbosity level (0 or 1)')
    parser.add_argument('--num_correct_trials', type=int, help='Number of correctness trials')
    parser.add_argument('--num_perf_trials', type=int, help='Number of performance trials')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout for evaluation in seconds')  # 新增参数
    
    args = parser.parse_args()
    
    # 调用函数并获取结果
    result = evaluate_kernel_wrapper(
        args.ref_arch_src,
        args.code,
        args.build_dir,
        int(args.device_id),
        bool(int(args.verbose)),  # 转换为布尔值
        int(args.num_correct_trials),
        int(args.num_perf_trials),
        int(args.timeout)  # 传递超时时间
    )
    
    # 输出 JSON 格式的结果
    print(json.dumps(result))
