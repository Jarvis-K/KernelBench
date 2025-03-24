#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析两个正常工作的kernel代码，使用NCU提取性能特征，生成优化提示
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
import csv
import subprocess
from concurrent.futures import ProcessPoolExecutor

from src.eval import (
    eval_kernel_against_ref,
    build_compile_cache,
    run_kernel_with_ncu,
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

class NCUKernelAnalyzer:
    def __init__(self, args):
        self.args = args
        self.output_dir = os.path.join(
            "results", "ncu_analysis", 
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
        
        # 读取问题和代码
        self.ref_arch_src = read_file(args.problem_file)
        self.good_kernel = read_file(args.good_kernel_file)
        self.bad_kernel = read_file(args.bad_kernel_file)
        
        # 创建临时构建目录
        self.build_dir = os.path.join(
            args.build_dir, 
            "ncu_kernel_analysis", 
            args.model, 
            f"level_{args.level}", 
            f"problem_{args.problem_id}"
        )
        os.makedirs(self.build_dir, exist_ok=True)
        
        # NCU分析结果目录
        self.ncu_dir = os.path.join(self.output_dir, "ncu_results")
        os.makedirs(self.ncu_dir, exist_ok=True)
        
        # 初始化结果
        self.good_result = None
        self.bad_result = None
        self.good_ncu_result = None
        self.bad_ncu_result = None
    
    def evaluate_kernel(self, kernel_code, kernel_type="unknown"):
        """评估kernel代码并返回结果"""
        logging.info(f"开始评估{kernel_type} kernel代码...")
        
        # 为每个kernel创建单独的构建目录
        build_dir = os.path.join(self.build_dir, kernel_type)
        os.makedirs(build_dir, exist_ok=True)
        
        # 编译代码
        success, stdout, error = build_compile_cache(kernel_code, self.args.verbose, build_dir)
        
        if not success:
            logging.error(f"{kernel_type} kernel编译失败: {error}")
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
            logging.info(f"{kernel_type} kernel编译成功，开始评估...")
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
                logging.error(f"{kernel_type} kernel评估超时")
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
                logging.error(f"{kernel_type} kernel评估失败: {str(e)}")
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
        
        logging.info(f"{kernel_type} kernel评估完成: 编译={result['compiled']}, 正确性={result.get('correctness', False)}, 加速比={result.get('speed_up', 0.0)}")
        
        return result
    
    def run_ncu_analysis(self, kernel_code, kernel_type="unknown"):
        """使用NCU分析kernel代码"""
        logging.info(f"开始对{kernel_type} kernel进行NCU性能分析...")
        
        # 为每个kernel创建单独的NCU分析目录
        ncu_output_dir = os.path.join(self.ncu_dir, kernel_type)
        os.makedirs(ncu_output_dir, exist_ok=True)
        
        # 抽取kernel名称
        kernel_name = self.extract_kernel_name(kernel_code) or f"{kernel_type}_kernel"
        kernel_py_path = os.path.join(ncu_output_dir, f"{kernel_name}.py")
        
        # 创建用于NCU分析的Python文件
        with open(kernel_py_path, "w") as f:
            f.write(self.create_ncu_test_code(kernel_code, kernel_name))
        
        # 运行NCU命令进行性能分析
        ncu_output_path = os.path.join(ncu_output_dir, f"{kernel_name}_ncu_output")
        ncu_cmd = f'ncu --set full -o {ncu_output_path} -f --kernel-name-base demangled python {kernel_py_path}'
        
        try:
            subprocess.run(ncu_cmd, shell=True, check=True)
            
            # 将NCU报告转换为CSV以便解析
            csv_output_path = os.path.join(ncu_output_dir, f"{kernel_name}_ncu_output.csv")
            ncu_csv_cmd = f"ncu -i {ncu_output_path}.ncu-rep -f --page details --csv --log-file {csv_output_path}"
            subprocess.run(ncu_csv_cmd, shell=True, check=True)
            
            # 解析NCU结果
            ncu_results = self.parse_ncu_results(csv_output_path)
            
            # 保存NCU分析结果
            result_file = os.path.join(self.output_dir, f"{kernel_type}_ncu_result.json")
            with open(result_file, "w") as f:
                json.dump(ncu_results, f, indent=4)
            
            logging.info(f"{kernel_type} kernel NCU分析完成，找到{len(ncu_results.get('optimization_suggestions', []))}条优化建议")
            
            return ncu_results
            
        except Exception as e:
            logging.error(f"运行NCU分析时出错: {str(e)}")
            error_result = {
                "error": str(e),
                "optimization_suggestions": []
            }
            
            # 保存错误信息
            result_file = os.path.join(self.output_dir, f"{kernel_type}_ncu_result.json")
            with open(result_file, "w") as f:
                json.dump(error_result, f, indent=4)
            
            return error_result
    
    def extract_kernel_name(self, code):
        """从代码中提取CUDA内核名称"""
        # 尝试匹配 __global__ void KERNEL_NAME
        match = re.search(r'__global__\s+void\s+(\w+)', code)
        if match:
            return match.group(1)
        
        # 尝试匹配 class KERNEL_NAME
        match = re.search(r'class\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        
        return None
    
    def create_ncu_test_code(self, kernel_code, kernel_name):
        """创建用于NCU分析的Python测试代码"""
        return f"""
import torch
import os

# 设置CUDA环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TORCH_EXTENSIONS_DIR"] = "{self.build_dir}/{kernel_name}"

# 问题定义代码
{self.ref_arch_src}

# 内核实现代码
{kernel_code}

# 初始化和运行内核
def main():
    # 获取初始化和输入数据
    init_inputs = get_init_inputs()
    init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]
    
    # 初始化模型
    model = ModelNew(*init_inputs)
    
    # 获取输入数据
    inputs = get_inputs()
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
    
    # 运行模型
    torch.cuda.synchronize()
    for _ in range(1):  # 运行多次确保内核被充分调用
        output = model(*inputs)
        torch.cuda.synchronize()
    
    print("已完成内核执行")

if __name__ == "__main__":
    main()
"""
    
    def parse_ncu_results(self, csv_path):
        """解析NCU输出的CSV文件，提取性能分析和优化建议"""
        if not os.path.exists(csv_path):
            return {"error": "NCU CSV文件不存在", "optimization_suggestions": []}
        
        try:
            # 收集所有优化建议和性能数据
            optimization_suggestions = []
            performance_metrics = {}
            
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # 收集优化建议
                    if 'Rule Description' in row and row['Rule Description'] and row['Rule Description'].strip():
                        # 尝试提取影响范围和预期加速效果
                        scope = row.get('Estimated Speedup Type', 'unknown')
                        try:
                            speedup = float(row.get('Estimated Speedup', '0').replace('x', ''))
                        except (ValueError, TypeError):
                            speedup = 0.0
                        
                        suggestion = {
                            'description': row['Rule Description'].strip(),
                            'scope': scope,
                            'estimated_speedup': speedup,
                            'category': row.get('Rule Category', 'unknown')
                        }
                        optimization_suggestions.append(suggestion)
                    
                    # 收集性能指标
                    metric_name = row.get('Metric Name', '')
                    if metric_name and 'Metric Value' in row:
                        try:
                            metric_value = float(row['Metric Value'])
                            performance_metrics[metric_name] = metric_value
                        except (ValueError, TypeError):
                            performance_metrics[metric_name] = row['Metric Value']
            
            # 按照预期加速效果排序建议
            optimization_suggestions.sort(key=lambda x: x['estimated_speedup'], reverse=True)
            
            # 截取最重要的前10条建议
            top_suggestions = optimization_suggestions[:10]
            
            return {
                "optimization_suggestions": top_suggestions,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logging.error(f"解析NCU结果时出错: {str(e)}")
            return {
                "error": str(e),
                "optimization_suggestions": []
            }
    
    def evaluate_kernels(self):
        """评估两个kernel代码并收集性能结果"""
        # 评估性能差的kernel
        self.bad_result = self.evaluate_kernel(self.bad_kernel, "bad")
        
        # 评估性能好的kernel
        self.good_result = self.evaluate_kernel(self.good_kernel, "good")
        
        # 如果代码都编译成功并且结果正确，则进行NCU分析
        if (self.bad_result.get("compiled", False) and self.bad_result.get("correctness", False) and
            self.good_result.get("compiled", False) and self.good_result.get("correctness", False)):
            
            # 对两份代码进行NCU分析
            self.bad_ncu_result = self.run_ncu_analysis(self.bad_kernel, "bad")
            self.good_ncu_result = self.run_ncu_analysis(self.good_kernel, "good")
            
            return True
        else:
            logging.error("有代码编译失败或结果不正确，无法进行NCU分析")
            return False
    
    def parser_performance_result(self, result):
        """解析性能评估结果为易读格式"""
        if not result.get("compiled", False):
            return "编译失败"
        elif not result.get("correctness", False):
            return "结果不正确"
        else:
            return f"运行时间: {result.get('runtime', 'N/A')}ms, 加速比: {result.get('speed_up', 'N/A')}"
    
    def format_ncu_suggestions(self, ncu_result, limit=5):
        """格式化NCU优化建议"""
        suggestions = ncu_result.get("optimization_suggestions", [])
        if not suggestions:
            return "无优化建议"
        
        formatted = []
        for i, sugg in enumerate(suggestions[:limit]):
            formatted.append(f"{i+1}. {sugg['description']}" + 
                           f" (预期加速: {sugg['estimated_speedup']}倍, 范围: {sugg['scope']})")
        
        return "\n".join(formatted)
    
    def format_performance_metrics(self, ncu_result, key_metrics=None):
        """格式化关键性能指标"""
        metrics = ncu_result.get("performance_metrics", {})
        if not metrics:
            return "无性能指标"
        
        # 默认显示一些重要指标
        if key_metrics is None:
            key_metrics = [
                # 占用率相关指标
                "Achieved Occupancy",
                "Achieved Active Warps Per SM",
                "Theoretical Occupancy",
                
                # Warp执行效率
                "Avg. Active Threads Per Warp",
                "Avg. Not Predicated Off Threads Per Warp",
                
                # 内存访问相关指标
                "Memory Throughput",
                "L1/TEX Hit Rate",
                "L2 Hit Rate",
                "DRAM Throughput",
                "L1/TEX Cache Throughput",
                "L2 Cache Throughput",
                
                # 算术运算效率
                "Compute (SM) Throughput",
                "Executed Ipc Active",
                "Executed Ipc Elapsed",
                "Issue Slots Busy",
                
                # 停顿与延迟
                "No Eligible",
                "Warp Cycles Per Executed Instruction",
                "Warp Cycles Per Issued Instruction",
                "SM Busy"
            ]
        
        formatted = []
        for metric in key_metrics:
            if metric in metrics:
                # 为某些指标添加百分号以便更清晰地显示
                if "%" in str(metrics[metric]) or metric.endswith(("Occupancy", "Throughput", "Rate", "Busy", "Eligible")):
                    formatted.append(f"{metric}: {metrics[metric]}%")
                else:
                    formatted.append(f"{metric}: {metrics[metric]}")
        
        return "\n".join(formatted)
    
    def create_optimization_prompt(self):
        """创建优化提示prompt"""
        prompt = f"""# CUDA内核优化分析

## 问题描述
```python
{self.ref_arch_src}
```

## 性能较差的实现（{self.parser_performance_result(self.bad_result)}）
```python
{self.bad_kernel}
```
### 关键性能指标
{self.format_performance_metrics(self.bad_ncu_result)}

## 性能较好的实现（{self.parser_performance_result(self.good_result)}）
```python
{self.good_kernel}
```

### 关键性能指标
{self.format_performance_metrics(self.good_ncu_result)}

## 任务

你是一位CUDA性能优化专家。请基于上述两个实现的代码和性能分析，提供以下内容：

1. 总结性能差异：概述两个实现在性能上的主要差异，重点指出哪些关键因素导致了性能差距。

2. 总结：请根据性能差异，用一段文本描述代码优化器在后续遇到相似问题和优化建议时的代码改进方向，请不要分点，同时要求总结具体、细节、可执行且对新手友好，请与代码紧密相关，不要输出改进的影响。将总结放置在<Summary></Summary>之间。

请确保你的分析清晰、具体、技术上准确，并能帮助理解CUDA优化的核心原则。
"""
        return prompt
    
    def create_match_verification_prompt(self, summary, bad_ncu_suggestions):
        """创建验证总结是否匹配NCU建议的prompt"""
        prompt = f"""# 性能优化总结与NCU建议匹配验证

## 性能较差的代码
```python
{self.bad_kernel}
```

## 性能较差代码的NCU建议（前5条）
{self.format_ncu_suggestions(self.bad_ncu_result, limit=5)}

## 性能较好的代码
```python
{self.good_kernel}
```

## 性能较好的代码的NCU建议（前5条）
{self.format_ncu_suggestions(self.good_ncu_result, limit=5)}

## 性能对比
| 实现 | 运行时间 (ms) | 基准时间 (ms) | 加速比 |
|------|--------------|--------------|--------|
| 性能差的实现 | {self.bad_result.get('runtime', 'N/A')} | {self.bad_result.get('baseline_runtime', 'N/A')} | {self.bad_result.get('speed_up', 'N/A')} |
| 性能好的实现 | {self.good_result.get('runtime', 'N/A')} | {self.good_result.get('baseline_runtime', 'N/A')} | {self.good_result.get('speed_up', 'N/A')} |

## 性能提升总结
{summary}

## 任务
1. 判断上述性能提升总结是否与性能较差代码的NCU建议相对应？
2. 如果对应，请指出总结最接近NCU建议中的哪一条？请返回NCU建议的序号和完整描述。

将是否对应回答放置在<Match>Yes/No</Match>，将对应NCU建议放置在<Suggestion></Suggestion>之间。
"""
        return prompt

    def extract_summary(self, response):
        """从响应中提取性能总结"""
        summary_match = re.search(r'<Summary>(.*?)</Summary>', response, re.DOTALL)
        if summary_match:
            return summary_match.group(1).strip()
        else:
            logging.warning("在响应中未找到<Summary>标签，将使用整个响应作为总结")
            return response
            
    def run(self):
        """运行完整的分析流程"""
        try:
            # 1. 评估kernel代码并收集性能结果
            if not self.evaluate_kernels():
                logging.error("评估失败，无法继续分析")
                return
            
            # 2. 创建优化提示prompt
            prompt = self.create_optimization_prompt()
            
            # 3. 保存prompt
            with open(os.path.join(self.output_dir, "optimization_prompt.txt"), "w") as f:
                f.write(prompt)
            
            # 4. 调用大模型生成优化建议
            logging.info("开始生成优化建议...")
            response, tokens = self.inference_server(prompt)
            
            # 5. 保存大模型响应
            with open(os.path.join(self.output_dir, "optimization_response.txt"), "w") as f:
                f.write(response)
            
            logging.info(f"优化建议生成完成，使用了 {tokens} 个tokens")
            
            # 6. 从响应中提取性能总结
            summary = self.extract_summary(response)
            with open(os.path.join(self.output_dir, "performance_summary.txt"), "w") as f:
                f.write(summary)
                
            # 7. 创建验证总结与NCU建议匹配的prompt
            match_prompt = self.create_match_verification_prompt(summary, self.bad_ncu_result)
            with open(os.path.join(self.output_dir, "match_verification_prompt.txt"), "w") as f:
                f.write(match_prompt)
                
            # 8. 调用大模型验证总结是否匹配NCU建议
            logging.info("开始验证性能总结与NCU建议的匹配度...")
            match_response, match_tokens = self.inference_server(match_prompt)
            
            # 9. 保存验证响应
            with open(os.path.join(self.output_dir, "match_verification_response.txt"), "w") as f:
                f.write(match_response)
                
            logging.info(f"匹配度验证完成，使用了 {match_tokens} 个tokens")
            
            # 10. 生成总结报告
            self.generate_report(prompt, response, summary, match_response)
            
            return response
            
        except Exception as e:
            logging.error(f"运行过程中发生错误: {str(e)}")
            raise
        finally:
            # 清理临时目录
            if not self.args.keep_build_dir and os.path.exists(self.build_dir):
                shutil.rmtree(self.build_dir)
    
    def generate_report(self, prompt, response, summary=None, match_response=None):
        """生成总结报告"""
        # 从match_response中提取匹配结果
        match_result = "未验证"
        matching_suggestion = "无"
        
        if match_response:
            match_match = re.search(r'<Match>(Yes|No)</Match>', match_response)
            if match_match:
                match_result = match_match.group(1)
                
            suggestion_match = re.search(r'<Suggestion>(.*?)</Suggestion>', match_response, re.DOTALL)
            if suggestion_match:
                matching_suggestion = suggestion_match.group(1).strip()
        
        report = f"""# NCU内核优化分析报告

## Level: {self.args.level}
## 问题ID: {self.args.problem_id}
## 模型: {self.args.model}

## 性能比较

| 实现 | 运行时间 (ms) | 基准时间 (ms) | 加速比 | 编译 | 正确性 |
|------|--------------|--------------|--------|------|--------|
| 性能差的实现 | {self.bad_result.get('runtime', 'N/A')} | {self.bad_result.get('baseline_runtime', 'N/A')} | {self.bad_result.get('speed_up', 'N/A')} | {self.bad_result.get('compiled', True)} | {self.bad_result.get('correctness', False)} |
| 性能好的实现 | {self.good_result.get('runtime', 'N/A')} | {self.good_result.get('baseline_runtime', 'N/A')} | {self.good_result.get('speed_up', 'N/A')} | {self.good_result.get('compiled', True)} | {self.good_result.get('correctness', True)} |

## NCU分析概要

### 关键性能指标
{self.format_performance_metrics(self.bad_ncu_result)}

### 性能差的实现优化建议
{self.format_ncu_suggestions(self.bad_ncu_result)}

### 关键性能指标
{self.format_performance_metrics(self.good_ncu_result)}

### 性能好的实现优化建议
{self.format_ncu_suggestions(self.good_ncu_result)}

## 性能提升总结

{summary if summary else "无总结"}

## 总结与NCU建议匹配度

匹配结果: {match_result}

对应建议: {matching_suggestion}

## 优化建议

{response}

"""
        
        with open(os.path.join(self.output_dir, "report.md"), "w") as f:
            f.write(report)
        
        logging.info(f"报告已生成到 {os.path.join(self.output_dir, 'report.md')}")

def parse_args():
    parser = argparse.ArgumentParser(description="分析两个正常工作的kernel代码，使用NCU提取性能特征，生成优化提示")
    
    parser.add_argument("--problem-file", type=str, required=True, 
                        help="问题描述文件路径")
    parser.add_argument("--good-kernel-file", type=str, required=True, 
                        help="性能好的kernel代码文件路径")
    parser.add_argument("--bad-kernel-file", type=str, required=True, 
                        help="性能差的kernel代码文件路径")
    
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
    
    # 提取问题ID和级别
    parser.add_argument("--problem-id", type=int, default=None,
                        help="问题ID")
    parser.add_argument("--level", type=int, default=None,
                        help="问题级别")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 如果未指定问题ID和级别，尝试从文件路径中提取
    if args.problem_id is None:
        match = re.search(r'/(\d+).*\.py', args.problem_file)
        if match:
            args.problem_id = int(match.group(1))
        else:
            logging.error("无法从问题文件路径提取问题ID，请使用--problem-id参数指定")
            return
    
    if args.level is None:
        match = re.search(r'level(\d+)', args.problem_file)
        if match:
            args.level = int(match.group(1))
        else:
            logging.error("无法从问题文件路径提取问题级别，请使用--level参数指定")
            return
    
    logging.info(f"开始分析问题 Level {args.level} Problem {args.problem_id} 的kernel性能...")
    
    analyzer = NCUKernelAnalyzer(args)
    analyzer.run()
    
    logging.info(f"所有结果已保存到 {analyzer.output_dir}")

if __name__ == "__main__":
    main() 