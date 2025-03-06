#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析两个kernel代码之间的差异，生成plan，并尝试通过plan改进性能差的代码
"""

import os
import json
import time
import torch
import argparse
import logging
import shutil
from pathlib import Path
import re
import signal
import multiprocessing as mp
import queue
from concurrent.futures import ProcessPoolExecutor

from src.eval import (
    eval_kernel_against_ref,
    build_compile_cache,
)
from evaluate_kernel import evaluate_kernel_isolated
from src.utils import read_file, extract_first_code, create_inference_server_from_presets

os.environ["TORCH_CUDA_ARCH_LIST"] = "Hopper"
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("评估超时")

def isolated_worker(func, args, result_queue, device_id):
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
                    torch.cuda.reset_peak_memory_stats(device_id)
                    torch.cuda.synchronize(device_id)
                time.sleep(5)  # 给GPU一些恢复时间
            except:
                time.sleep(5)
                pass

def run_isolated_process(func, args, timeout=120, device_id=0):
    """
    在完全隔离的进程中运行函数，确保CUDA异常不会影响主进程
    使用进程间通信来传递结果
    """
    # 创建一个队列用于进程间通信
    result_queue = mp.Queue()
    
    # 创建并启动隔离进程
    process = mp.Process(target=isolated_worker, args=(func, args, result_queue, device_id))
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
            torch.cuda.synchronize(device_id)
            time.sleep(2)  # 给GPU一些恢复时间
        except Exception as e:
            time.sleep(2)
            logging.warning(f"清理GPU状态失败: {str(e)}")
        
        raise TimeoutError("评估超时")

class KernelDiffAnalyzer:
    def __init__(self, args):
        self.args = args
        self.output_dir = os.path.join(
            "results", "eval_logs", "ablation", 
            args.model, f"level_{args.level}", f"problem_{args.problem_id}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化推理服务器
        self.inference_server = create_inference_server_from_presets(
            server_type=args.server_type,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            time_generation=True
        )

        self.plan_server = create_inference_server_from_presets(
            server_type="pandas",
            model_name="gpt-4o-mini",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            time_generation=True
        )
        
        # 读取问题和代码
        self.ref_arch_src = read_file(args.problem_file)
        self.good_kernel = read_file(args.good_kernel_file)
        self.bad_kernel = read_file(args.bad_kernel_file)
        
        # 创建临时构建目录
        self.build_dir = os.path.join(
            args.build_dir, 
            "kernel_diff_analysis", 
            args.model, 
            f"level_{args.level}", 
            f"problem_{args.problem_id}"
        )
        os.makedirs(self.build_dir, exist_ok=True)
        
        # 初始化结果
        self.good_result = None
        self.bad_result = None
    
    def evaluate_kernel(self, kernel_code, kernel_type="unknown"):
        """评估kernel代码并返回结果"""
        logging.info(f"开始评估{kernel_type}kernel代码...")
        
        # 为每个kernel创建单独的构建目录
        build_dir = os.path.join(self.build_dir, kernel_type)
        os.makedirs(build_dir, exist_ok=True)
        
        # 编译代码
        success, stdout, error = build_compile_cache(kernel_code, self.args.verbose, build_dir)
        
        if not success:
            logging.error(f"{kernel_type}kernel编译失败: {error}")
            result = {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "baseline_runtime": 0.0,
                "speed_up": 0.0,
                "error": error,
                "metadata": stdout
            }
        else:
            logging.info(f"{kernel_type}kernel编译成功，开始评估...")
            try:
                # 在隔离进程中评估
                eval_args = [
                    self.ref_arch_src,
                    kernel_code,
                    build_dir,
                    0,  # 使用设备0
                    self.args.verbose,
                    3,  # num_correct_trials
                    30  # num_perf_trials
                ]
                
                result_obj = run_isolated_process(evaluate_kernel_isolated, eval_args, timeout=60, device_id=self.args.device)
                result = vars(result_obj)
                
            except TimeoutError:
                logging.error(f"{kernel_type}kernel评估超时")
                result = {
                    "compiled": True,
                    "correctness": False,
                    "runtime": 0.0,
                    "baseline_runtime": 0.0,
                    "speed_up": 0.0,
                    "error": "runtime_error",
                    "metadata": "runtime_error"
                }
            except Exception as e:
                logging.error(f"{kernel_type}kernel评估失败: {str(e)}")
                result = {
                    "compiled": True,
                    "correctness": False,
                    "runtime": 0.0,
                    "baseline_runtime": 0.0,
                    "speed_up": 0.0,
                    "error": str(e),
                    "metadata": str(e)
                }
        
        # 保存结果
        result_file = os.path.join(self.output_dir, f"{kernel_type}_kernel_result.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4)

        with open(os.path.join(self.output_dir, f"{kernel_type}_code.py"), "w") as f:
            f.write(kernel_code)
        
        logging.info(f"{kernel_type}kernel评估完成: 编译={result['compiled']}, 正确性={result.get('correctness', False)}, 加速比={result.get('speed_up', 0.0)}")
        
        return result
    
    def evaluate_kernels(self):
        """评估原始的好/坏kernel代码"""
        # 评估性能差的kernel
        self.bad_result = self.evaluate_kernel(self.bad_kernel, "bad")
        
        # 评估性能好的kernel
        self.good_result = self.evaluate_kernel(self.good_kernel, "good")

    def parser_result(self, result):
        error = str(result.get('metadata', 'N/A'))
        parsered_all_output = list(set(re.findall(r"(error: .*?)\n", error, re.DOTALL | re.MULTILINE)))
        warnings = list(set(re.findall(r"(warning: .*?)\n", error, re.DOTALL | re.MULTILINE)))
        error = '\n'.join(warnings+parsered_all_output) if len(warnings+parsered_all_output) > 0 else error
        if not result.get("compiled", False):
            s = f"Failed to compile: {error}"
        else:
            if not result.get("correctness", False):
                s = f"Code runs incorrectly: {error}"
            else:
                s = f"Runtime: {result.get('runtime', 'N/A')}ms, Speedup: {result.get('speed_up', 'N/A')}"
        return s
    
    def create_diff_prompt(self):
        """创建分析两个kernel差异的prompt"""
        prompt = f"""
You are a CUDA programming and performance optimization expert. I need you to analyze two CUDA kernel implementations of the same problem and create a concise optimization plan.

Problem description:
```python
{self.ref_arch_src}
```

Lower-performance implementation ({self.parser_result(self.bad_result)}):
```python
{self.bad_kernel}
```

Higher-performance implementation ({self.parser_result(self.good_result)}):
```python
{self.good_kernel}
```

Please provide a focused, actionable optimization plan that identifies the MOST CRITICAL differences affecting performance. Your plan should be concise but specific.

For each key optimization (limit to 3-5 most important changes):
1. Identify the specific code section to modify
2. Describe the exact change needed
3. Briefly explain why this change improves performance

Format your response as:
1. One paragraph summary of key performance differences
2. Numbered list of specific optimizations (3-5 items), each with:
   - Target: [Detailed code snippet in the original code]
   - Change: [specific modification]
   - Result: [modified code snippet]
   - Reason: [brief technical explanation]

Focus only on the most impactful changes.
"""
        return prompt
    
    def create_improvement_prompt(self, plan, good_flag=False):
        """创建基于plan改进代码的prompt"""
        prompt = f"""
You are a CUDA programming and performance optimization expert. Implement the following optimization plan to improve a CUDA kernel.

Problem description:
```python
{self.ref_arch_src}
```

Current implementation (Lower performance):
```python
{self.bad_kernel if not good_flag else self.good_kernel}
```

Optimization plan:
{plan}

Implement all the changes from the optimization plan. Your implementation must:
1. Follow each optimization step precisely and strictly
2. Maintain functional correctness
3. Name the optimized architecture ModelNew

Important requirements:
- Your implementation must compile and execute correctly
- Focus on implementing the specific optimizations in the plan strictly
- If you encounter any issues with the plan, implement the changes that make technical sense
- Name your optimized architecture ModelNew

Please provide the complete improved code, including all necessary function and class definitions. Add brief comments before each modified section explaining the optimization applied.
"""
        return prompt
    
    def analyze_diff(self):
        """分析两个kernel的差异并生成plan"""
        logging.info("开始分析kernel差异...")
        
        # 构建prompt
        prompt = self.create_diff_prompt()
        
        # 保存prompt
        with open(os.path.join(self.output_dir, "diff_prompt.txt"), "w") as f:
            f.write(prompt)
        
        # 调用大模型
        response, tokens = self.plan_server(prompt)
        
        # 保存response
        with open(os.path.join(self.output_dir, "diff_response.txt"), "w") as f:
            f.write(response)
        
        logging.info(f"差异分析完成，使用了 {tokens} 个tokens")
        
        # 提取plan (假设plan就是整个response)
        plan = response
        
        return plan
    
    def improve_code(self, plan):
        """基于plan改进代码"""
        logging.info("开始基于plan改进代码...")
        
        # 构建prompt
        prompt = self.create_improvement_prompt(plan)
        
        # 保存prompt
        with open(os.path.join(self.output_dir, "improvement_prompt.txt"), "w") as f:
            f.write(prompt)
        
        # 调用大模型
        response, tokens = self.inference_server(prompt)
        
        # 保存response
        with open(os.path.join(self.output_dir, "improvement_response.txt"), "w") as f:
            f.write(response)
        
        # 提取代码
        new_code, original_code = extract_first_code(response, ["python", "cpp"])
        
        # 保存新代码
        with open(os.path.join(self.output_dir, "new_code.py"), "w") as f:
            f.write(new_code)
        
        logging.info(f"代码改进完成，使用了 {tokens} 个tokens")
        
        return new_code
    
    def compare_results(self, new_result):
        """比较新代码与原始好/坏代码的性能"""
        logging.info("比较性能结果...")
        
        comparison = {
            "bad_kernel": {
                "runtime": self.bad_result.get("runtime", -1),
                "speed_up": self.bad_result.get("speed_up", 0.0)
            },
            "good_kernel": {
                "runtime": self.good_result.get("runtime", -1),
                "speed_up": self.good_result.get("speed_up", 0.0)
            },
            "new_kernel": {
                "runtime": new_result.get("runtime", -1),
                "speed_up": new_result.get("speed_up", 0.0),
                "compiled": new_result.get("compiled", False),
                "correctness": new_result.get("correctness", False)
            }
        }
        
        # 计算与好代码的接近程度
        if new_result.get("correctness", False) and self.good_result.get("speed_up", 0) > 0:
            improvement_ratio = new_result.get("speed_up", 0) / self.good_result.get("speed_up", 1)
            comparison["improvement_ratio"] = improvement_ratio
            comparison["success"] = improvement_ratio >= 0.8  # 如果达到好代码性能的80%以上，视为成功
        else:
            comparison["improvement_ratio"] = 0
            comparison["success"] = False
        
        # 保存比较结果
        with open(os.path.join(self.output_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=4)
        
        return comparison
    
    def run(self):
        """运行完整的分析和改进流程"""
        try:
            # 0. 评估原始kernel代码
            self.evaluate_kernels()
            
            # 1. 分析差异并生成plan
            if self.args.plan_file is None:
                plan = self.analyze_diff()
            else:
                with open(self.args.plan_file, "r") as f:
                    plan = f.read()
            
            # 2. 基于plan改进代码
            new_code = self.improve_code(plan)
            
            # 3. 评估新代码
            new_result = self.evaluate_kernel(new_code, "new")
            
            # 4. 比较结果
            comparison = self.compare_results(new_result)
            
            # 5. 生成总结报告
            self.generate_report(plan, new_code, new_result, comparison)
            
            return comparison
            
        except Exception as e:
            logging.error(f"运行过程中发生错误: {str(e)}")
            raise
        finally:
            # 清理临时目录
            if not self.args.keep_build_dir and os.path.exists(self.build_dir):
                shutil.rmtree(self.build_dir)
    
    def generate_report(self, plan, new_code, new_result, comparison):
        """生成总结报告"""
        report = f"""# Kernel优化分析报告

## Level: {self.args.level}
## 问题ID: {self.args.problem_id}
## 模型: {self.args.model}

## 性能比较

| 实现 | 运行时间 (ms) | 基准时间 (ms) | 加速比 | 编译 | 正确性 |
|------|--------------|--------------|--------|------|--------|
| 性能差的实现 | {self.bad_result.get('runtime', 'N/A')} | {self.bad_result.get('baseline_runtime', 'N/A')} | {self.bad_result.get('speed_up', 'N/A')} | {self.bad_result.get('compiled', True)} | {self.bad_result.get('correctness', False)} |
| 性能好的实现 | {self.good_result.get('runtime', 'N/A')} | {self.good_result.get('baseline_runtime', 'N/A')} | {self.good_result.get('speed_up', 'N/A')} | {self.good_result.get('compiled', True)} | {self.good_result.get('correctness', True)} |
| 改进后的实现 | {new_result.get('runtime', 'N/A')} | {new_result.get('baseline_runtime', 'N/A')} | {new_result.get('speed_up', 'N/A')} | {new_result.get('compiled', False)} | {new_result.get('correctness', False)} |

## 改进比例

改进后的实现达到了性能好的实现的 **{comparison.get('improvement_ratio', 0)*100:.2f}%** 的性能。

## 优化是否成功

**{'是' if comparison.get('success', False) else '否'}**
{self.parser_result(new_result) if not new_result.get('correctness', False) else ''}

## 优化计划摘要

{plan}

## 改进后的代码

```python
{new_code}
```

"""
        
        with open(os.path.join(self.output_dir, "report.md"), "w") as f:
            f.write(report)
        
        logging.info(f"报告已生成到 {os.path.join(self.output_dir, 'report.md')}")

def parse_args():
    parser = argparse.ArgumentParser(description="分析两个kernel代码之间的差异，生成plan，并尝试通过plan改进性能差的代码")
    
    parser.add_argument("--problem-file", type=str, required=True, 
                        help="问题描述文件路径")
    parser.add_argument("--good-kernel-file", type=str, required=True, 
                        help="性能好的kernel代码文件路径")
    parser.add_argument("--bad-kernel-file", type=str, required=True, 
                        help="性能差的kernel代码文件路径")
    parser.add_argument("--plan-file", type=str, default=None,
                        help="优化计划文件路径")
    
    parser.add_argument("--model", type=str, default="claude-3-sonnet-20240229",
                        help="使用的大模型名称")
    parser.add_argument("--server-type", type=str, default="anthropic",
                        help="推理服务器类型")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="最大生成token数")
    parser.add_argument("--build-dir", type=str, default="./cache/build",
                        help="构建目录")
    parser.add_argument("--keep-build-dir", action="store_true",
                        help="保留构建目录")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    parser.add_argument("--device", type=int, default=0,
                        help="使用哪个GPU设备")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    args.problem_id = int(re.search(r'/(\d+).*\.py', args.problem_file).group(1))
    args.level = int(re.search(r'level(\d+)', args.problem_file).group(1))
    
    logging.info(f"开始分析问题 Level {args.level} Problem {args.problem_id} 的kernel差异...")
    
    analyzer = KernelDiffAnalyzer(args)
    comparison = analyzer.run()
    
    if comparison.get("success", False):
        logging.info("优化成功！新代码达到了性能好的代码的性能水平。")
    else:
        logging.info("优化未达到预期效果。请查看报告了解详情。")
    
    logging.info(f"所有结果已保存到 {analyzer.output_dir}")

if __name__ == "__main__":
    main()