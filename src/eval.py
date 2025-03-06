"""
Helpers for Evaluations
"""

import requests
import torch
import torch.nn as nn
import os, subprocess
from pydantic import BaseModel
import numpy as np
import random
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import sys
import threading
from . import utils
import re
import time
import csv

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def fetch_kernel_from_database(
    run_name: str, problem_id: int, sample_id: int, server_url: str
):
    """
    Intenral to us with our django database
    Return a dict with kernel hash, kernel code, problem_id
    """
    response = requests.get(
        f"{server_url}/get_kernel_by_run_problem_sample/{run_name}/{problem_id}/{sample_id}",
        json={"run_name": run_name, "problem_id": problem_id, "sample_id": sample_id},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert str(response_json["problem_id"]) == str(problem_id)
    return response_json


def fetch_ref_arch_from_problem_id(problem_id, problems, with_name=False) -> str:
    """
    Fetches the reference architecture in string for a given problem_id
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem_path = problems[problem_id]

    # problem_path = os.path.join(REPO_ROOT_PATH, problem)
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file at {problem_path} does not exist.")

    ref_arch = utils.read_file(problem_path)
    if not with_name:
        return ref_arch
    else:
        return (problem_path, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
    dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    return fetch_ref_arch_from_problem_id(problem_id, dataset, with_name)


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """
    iteration: int = -1
    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance
    baseline_runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    baseline_runtime_stats: dict = {}  # only recorded if we decide to measure performance
    speed_up: float = 0.0  # only recorded if we decide to measure performance
    tokens: int = 0
    ncu_rule_descriptions: str = ""  # 新增字段，存储NCU规则描述


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(curr_context: dict, device: torch.device):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?

def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str, str]: whether compilation is successful, stdout content as string, error message
    """
    # 这里我们必须使用子进程方法来捕获所有编译输出
    # 因为底层编译工具的输出无法通过Python的标准输出/错误重定向捕获
    
    # 确保设置TORCH_USE_CUDA_DSA环境变量
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    if build_dir:
        # 添加必要的环境变量设置到源代码
        custom_model_src = (
            "import os\n" 
            f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
            "os.environ['TORCH_USE_CUDA_DSA'] = '1'\n"
        ) + custom_model_src

    # 创建临时Python文件
    kernel_hash = hash(custom_model_src)
    tmp_file = os.path.join(build_dir if build_dir else "/tmp", f"tmp_compile_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
    
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    # 使用子进程执行编译，这样可以捕获所有输出
    process = subprocess.Popen(
        ['python', tmp_file], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ, TORCH_USE_CUDA_DSA="1")  # 确保子进程也设置了环境变量
    )
    stdout, stderr = process.communicate()
    
    # 清理临时文件
    try:
        os.remove(tmp_file)
    except:
        pass
    
    # 合并输出
    all_output = stdout + "\n" + stderr
    
    # 检查编译是否成功
    if process.returncode != 0:
        if verbose:
            print(f"[Compilation] Failed to compile custom CUDA kernel. Return code: {process.returncode}")
            print("========================================================")
            print(all_output)
            print("========================================================")
        return False, all_output, f"Compilation failed with return code {process.returncode}"
    
    if verbose:
        print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
        print("========================================================")
        print(all_output)
        print("========================================================")
    
    # 尝试加载模型以验证编译是否真正成功

    try:
        context = {}
        from io import StringIO
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            load_custom_model(custom_model_src, context, build_dir)
        return True, all_output, None
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"[Compilation] Compiled but failed to load model: {error_msg}")
        return False, all_output, error_msg


def build_compile_cache_with_capturing(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(['python', tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)


    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode('utf-8'))
        print("[CPU Precompile] stderr: \n", stderr.decode('utf-8')) 

    return returncode, stdout.decode('utf-8'), stderr.decode('utf-8')

def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
    use_ncu: bool = False,
    ncu_log_path: os.PathLike = None,
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # 用于调试
    
    # 尝试提取CUDA kernel名称
    kernel_name = utils.extract_kernel_name(custom_model_src)
    
    try:
        set_seed(seed_num)  # set seed for reproducible input
        init_inputs = get_init_inputs()
        init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            original_model = Model(*init_inputs)
            assert hasattr(original_model, "forward")
            if verbose:
                print("[Eval] Original Model Loaded")
        if verbose:
            print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        # torch.cuda.init()
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            try:
                graceful_eval_cleanup(context, device)
            except:
                pass
            return KernelExecResult(
                compiled=False, metadata=metadata
            )
        else:
            metadata["compilation_error"] = str(e)
            try:
                graceful_eval_cleanup(context, device)
            except:
                pass
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # 此时我们已通过编译
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device)
        metadata["runtime_error"] = str(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps
    except Exception as e:
        print(f"[Eval] Error in loading custom CUDA kernel: {e}")
        graceful_eval_cleanup(context, device)
        metadata["runtime_error"] = str(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = str(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                # # 如果有kernel_name，创建一个可运行的Python文件用于ncu分析
                # if use_ncu:
                #     ncu_rule_descriptions = run_kernel_with_ncu(original_model_src, custom_model_src, seed_num, verbose, build_dir, device, ncu_log_path)
                #     if ncu_rule_descriptions:
                #         metadata["ncu_rule_descriptions"] = ncu_rule_descriptions
                #     kernel_exec_result.ncu_rule_descriptions = ncu_rule_descriptions

                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["median"]
                kernel_exec_result.runtime_stats = runtime_stats

                model = original_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                baseline_runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Baseline Performance Stats: {baseline_runtime_stats}")
                kernel_exec_result.baseline_runtime = baseline_runtime_stats["median"]
                kernel_exec_result.baseline_runtime_stats = baseline_runtime_stats

                kernel_exec_result.speed_up = max(0.0, kernel_exec_result.baseline_runtime / kernel_exec_result.runtime)
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = str(e)

    try:
        graceful_eval_cleanup(context, device)
    except:
        pass 
        
    return kernel_exec_result

def run_kernel_with_ncu(original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
    ncu_log_path: os.PathLike = None,
):
    kernel_name = utils.extract_kernel_name(custom_model_src)
    kernel_py_path = os.path.join(ncu_log_path, f"{kernel_name}.py")
    if not build_dir:
        build_dir = None
    else:
        build_dir = f"\"{build_dir}\""
    with open(kernel_py_path, "w") as f:
        f.write(f"""
import torch
import sys
sys.path.append("{os.path.dirname(ncu_log_path)}")
sys.path.append("{REPO_TOP_PATH}")
from src.eval import load_original_model_and_inputs, load_custom_model, set_seed

# 加载原始模型和自定义模型
original_model_src = '''{original_model_src}'''
custom_model_src = '''{custom_model_src}'''

context = {{}}
Model, get_init_inputs, get_inputs = load_original_model_and_inputs(original_model_src, context)
ModelNew = load_custom_model(custom_model_src, context, {build_dir})

# 设置种子并获取输入
set_seed({seed_num})
init_inputs = get_init_inputs()
init_inputs = [x.cuda(device="{device}") if isinstance(x, torch.Tensor) else x for x in init_inputs]

# 创建模型实例
custom_model = ModelNew(*init_inputs).to("{device}")

# 获取输入数据
set_seed({seed_num})
inputs = get_inputs()
inputs = [x.cuda(device="{device}") if isinstance(x, torch.Tensor) else x for x in inputs]

# 运行模型
with torch.no_grad():
    for _ in range(2):  # 运行多次以确保内核被调用
        output = custom_model(*inputs)
        torch.cuda.synchronize(device="{device}")
print("Completed running")
""")
    
    # 运行ncu命令进行性能分析
    ncu_output_path = os.path.join(ncu_log_path, "output")
    ncu_cmd = f'ncu --set full -o {ncu_output_path} -f --kernel-name "{kernel_name}" python {kernel_py_path}'
    
    try:
        subprocess.run(ncu_cmd, shell=True, check=True)
        
        # 将ncu报告转换为CSV
        csv_output_path = os.path.join(ncu_log_path, "output.csv")
        ncu_csv_cmd = f"ncu -i {ncu_output_path}.ncu-rep -f --page details --csv --log-file {csv_output_path}"
        subprocess.run(ncu_csv_cmd, shell=True, check=True)
        
        # 读取CSV文件中的Rule Description
        rule_descriptions = []
        speedup_data = []
        
        # 收集所有global类型的优化建议
        with open(csv_output_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if 'Rule Description' in row and row['Rule Description'] and row['Rule Description'].strip():
                    if 'Estimated Speedup Type' in row and row['Estimated Speedup Type'] == 'global':
                        try:
                            # 尝试提取Estimated Speedup值并转换为浮点数
                            speedup = float(row.get('Estimated Speedup', '0').replace('x', ''))
                            speedup_data.append({
                                'description': row['Rule Description'].strip(),
                                'speedup': speedup
                            })
                        except (ValueError, TypeError):
                            # 如果无法转换为浮点数，则使用0作为默认值
                            speedup_data.append({
                                'description': row['Rule Description'].strip(),
                                'speedup': 0.0
                            })
        # 根据speedup值排序（从高到低）
        speedup_data.sort(key=lambda x: x['speedup'], reverse=True)
        
        # 提取排序后的描述
        rule_descriptions = [f"{item['description']}" for item in speedup_data][:1]
        
        # 将规则描述添加到元数据
        return '\n'.join(rule_descriptions)
    except Exception as e:
        if verbose:
            print(f"[评估] 运行NCU命令时出错: {e}")
        return None

def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception(
                    "runtime_error", str(e), metadata, truncate=True
                )
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata

def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: list[str], baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
        "median": float(f"{np.median(elapsed_times):.3g}"),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


# if __name__ == "__main__":
# fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# fetch_baseline_time("level1", 0, ["1_Square_matrix_multiplication_.py"], "tests/baseline_time_matx3.json")
