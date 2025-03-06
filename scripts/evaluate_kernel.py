import multiprocessing
import concurrent.futures
import torch
import logging
from src.eval import eval_kernel_against_ref
import json
import argparse
import os
import time
import hashlib

# 设置多进程启动方法为 'spawn'
multiprocessing.set_start_method('spawn', force=True)


def evaluate_kernel(ref_arch_src, code, build_dir, device_id, verbose, num_correct_trials, num_perf_trials):
    # 在每个进程中初始化 CUDA 设备
    # 重置CUDA设备，确保干净的环境
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.synchronize(device_id)
    
    # 设置CUDA错误处理为同步模式，便于立即捕获错误
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    device = torch.device(f"cuda:{device_id}")
    
    result = eval_kernel_against_ref(
        ref_arch_src,
        code,
        verbose=verbose,
        measure_performance=True,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        device=device,
        build_dir=build_dir
    )
    
    # 评估完成后清理CUDA状态
    if 'runtime_error' in result.metadata:
        if "CUDA" in result.metadata['runtime_error'] or "cuda" in result.metadata['runtime_error'] or "illegal memory access" in result.metadata['runtime_error']:
            logging.error("检测到CUDA错误，尝试恢复GPU状态")
            try:
                # 尝试恢复GPU状态
                time.sleep(3)
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device_id)
                    torch.cuda.synchronize(device_id)
                time.sleep(3)
            except:
                time.sleep(3)
                logging.error("GPU状态恢复失败")
            from src.eval import KernelExecResult
            error_result = KernelExecResult(
                iteration=-1,
                compiled=True,
                correctness=False,
                runtime=-1.0,
                baseline_runtime=0.0,
                speed_up=0.0,
                metadata=result.metadata
            )
            return error_result
    
    return result

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
    # 计算代码哈希值作为缓存键
    code_hash = hashlib.md5(code.encode()).hexdigest()
    cache_key = f"{code_hash}_{device_id}_{num_correct_trials}_{num_perf_trials}"
    
    try:
        # 使用 ProcessPoolExecutor 来处理超时
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(
                evaluate_kernel,
                ref_arch_src,
                code,
                build_dir,
                device_id,
                True,
                num_correct_trials,
                num_perf_trials
            )
            # 等待结果，设置超时时间
            result = future.result(timeout=timeout)
        
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
    
# 添加一个完全隔离的评估函数
def evaluate_kernel_isolated(ref_arch_src, code, build_dir, device_id, verbose, num_correct_trials, num_perf_trials, use_ncu):
    """
    在完全隔离的进程中评估内核，确保GPU异常不会影响其他评估任务
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步模式，便于捕获错误
    os.environ["TORCH_USE_CUDA_DSA"] = "1"  # 启用设备端断言
    
    if not os.path.exists(build_dir):
        print("=======================================================================")
        print(f"==============Build directory {build_dir} does not exist==============")
        print("=======================================================================")

    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.synchronize(device_id)
    
    # 设置CUDA错误处理为同步模式，便于立即捕获错误
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    device = torch.device(f"cuda:{device_id}")
    
    result = eval_kernel_against_ref(
        ref_arch_src,
        code,
        verbose=verbose,
        measure_performance=True,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        device=device,
        build_dir=build_dir,
        use_ncu=use_ncu
    )
    
    # 评估完成后清理CUDA状态
    if 'runtime_error' in result.metadata:
        if "CUDA error" in result.metadata['runtime_error'] or "illegal memory access" in result.metadata['runtime_error']:
            logging.error("检测到CUDA错误，尝试恢复GPU状态")
            try:
                # 尝试恢复GPU状态
                time.sleep(3)
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device_id)
                    torch.cuda.synchronize(device_id)
                time.sleep(3)
            except:
                time.sleep(3)
                logging.error("GPU状态恢复失败")
            from src.eval import KernelExecResult
            error_result = KernelExecResult(
                iteration=-1,
                compiled=True,
                correctness=False,
                runtime=-1.0,
                baseline_runtime=0.0,
                speed_up=0.0,
                metadata=result.metadata
            )
            return error_result
    
    return result

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
