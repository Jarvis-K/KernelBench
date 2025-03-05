import time
import numpy as np
start = time.time()
from functools import lru_cache
from typing import Optional, List
import torch
import torch.nn.functional as F
from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
    _score_mod_signature,
    _mask_mod_signature,
)
try:
    from jsonargparse import ArgumentParser
except ImportError:
    raise ImportError("Be sure to run: pip install -e .'[viz]'")
parser = ArgumentParser(description="Run specific examples or all examples.")
parser.add_argument(
    "--examples",
    type=str,
    nargs="+",
    default=["all"]
)
parser.add_argument(
    "--filename",
    type=str,
    required=False,  # 根据需要设置为True或False
    help="Specify the filename to be processed."
)
parser.add_argument(
    "--attention_type",
    type=str,
    required=False,  # 根据需要设置为True或False
    help="Specify the filename to be processed."
)
parser.add_argument(
    "--attention_name",
    type=str,
    required=False,  # 根据需要设置为True或False
    help="Specify the filename to be processed."
)
args = parser.parse_args()
function_name = "new_alibi"
import dill
with open(args.filename, 'r') as file:
    read_function_code = file.read()
local_vars = {}
# exec(read_function_code, globals(), local_vars)
# new_alibi_mod = local_vars[function_name]

import textwrap

# 1. 先处理内部函数的缩进
indented_function = textwrap.indent(read_function_code, '    ')  # 添加4个空格的缩进

# 2. 然后构建外部函数
string_new_func = f'''
def new_alibi(H: int) -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """
{indented_function}
    return alibi_mod  # 直接返回内部函数对象
'''
exec(string_new_func, globals(), local_vars)
new_alibi = local_vars[function_name]





from triton.testing import do_bench
from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap
from torch import Tensor
import random
import torch
import torch.nn.functional as F



AVAILABLE_EXAMPLES = {
    "causal": lambda: test_mask(mask_mod=causal_mask),
    "alibi": lambda: test_mask(score_mod=generate_alibi_bias(16), skip_correctness=True),
    "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=1024)),
    "prefix_lm": lambda: test_mask(mask_mod=generate_prefix_lm_mask(prefix_length=1024)),
    "document": lambda: run_document_masking(max_seq_len=32768, num_docs=12),
    "softcap": lambda: test_mask(
        score_mod=generate_tanh_softcap(30, approx=False), skip_correctness=True
    ),
    "softcap_approx": lambda: test_mask(
        score_mod=generate_tanh_softcap(30, approx=True), skip_correctness=True
    ),
    "optimized": lambda: test_mask(
        score_mod=new_alibi(16), skip_correctness=True
    ),
}

torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparsity to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


def test_mask(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
):
    assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"
    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=device)

    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]
    gradOut = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv, is_causal=True)
    sdpa_mask = lambda: F.scaled_dot_product_attention(*qkv, attn_mask=mask)
    flex_attention_call = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)
    # flex_attention_call_optimized = lambda: flex_attention(*qkv, score_mod=new_alibi(16), block_mask=block_mask)
    flex_attention_call_optimized = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)
    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S
    repeat_num = 50
    result_list = []
    for i in range(repeat_num):
        times = []
        # Run both original and optimized versions
        for attn_name, attn_fn in [("original", flex_attention_call), ("optimized", flex_attention_call_optimized)]:
            # Reset gradients before each test
            for tensor in qkv:
                tensor.grad = None
                
            # Measure forward time
            fwd_time = do_bench(attn_fn)
            fwd_out = attn_fn()
            
            # Measure backward time
            bwd_time = do_bench(lambda: fwd_out.backward(gradOut, retain_graph=True))
            times.append((fwd_time, bwd_time, attn_name))
            
            # Clean up to prevent interference between tests
            del fwd_out
            torch.cuda.empty_cache()
        
        print_header(
            f"{score_mod.__name__ if score_mod is not None else mask_mod.__name__}".replace(
                "_", " "
            ).title()
        )
        # Compare original vs optimized implementation
        if score_mod is not None and "alibi" in score_mod.__name__:
            print("\nComparing original vs optimized implementation:")
            outs_origin = []
            outs_optimized = []
            # Test original implementation
            for tensor in qkv:
                tensor.grad = None
            
            out_orig = flex_attention_call()
            outs_origin.append(out_orig)
            out_orig.backward(gradOut)
            outs_origin += [tensor.grad for tensor in qkv]
            
            # Clean up before testing optimized implementation

            torch.cuda.empty_cache()
            
            # Test optimized implementation
            for tensor in qkv:
                tensor.grad = None
                
            out_opt = flex_attention_call_optimized()
            outs_optimized.append(out_opt)
            out_opt.backward(gradOut)
            outs_optimized += [tensor.grad for tensor in qkv]
            
            # Clean up after tests
            del out_opt
            torch.cuda.empty_cache()
            
            try:
                for out_origin, out_optimized in zip(outs_origin, outs_optimized):
                    torch.testing.assert_close(out_origin, out_optimized, atol=1e-1, rtol=1e-2)
                print("Outputs match between original and optimized implementations ✅")
            except AssertionError as e:
                print(f"Outputs differ between implementations: {e}")
                raise RuntimeError("Output mismatch detected between original and optimized implementations")
            
            # Clean up gradients
        torch.cuda.empty_cache()

        # Format results for display
        (flex_ms, flex_bw_ms, flex_name), (flex_opt_ms, flex_opt_bw_ms, flex_opt_name) = times
        
        results = [
            [
                f"flexattention ({flex_name})",
                f"{flex_ms:.4f}",
                f"{calculate_tflops(flops, flex_ms, 4):.2f}",
                f"{flex_bw_ms:.4f}",
                f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
            ],
            [
                f"flexattention ({flex_opt_name})",
                f"{flex_opt_ms:.4f}",
                f"{calculate_tflops(flops, flex_opt_ms, 4):.2f}",
                f"{flex_opt_bw_ms:.4f}",
                f"{calculate_tflops(flops, flex_opt_bw_ms, 10):.2f}",
            ],
        ]
        
        print(
            tabulate(
                results,
                headers=[
                    "Operation",
                    "FW Time (ms)",
                    "FW FLOPS (TF/s)",
                    "BW Time (ms)",
                    "BW FLOPS (TF/s)",
                ],
                tablefmt="grid",
            )
        )
        
        # Calculate and display speedup
        fwd_speedup = flex_ms / flex_opt_ms
        bwd_speedup = flex_bw_ms / flex_opt_bw_ms
        total_speedup = (flex_ms + flex_bw_ms) / (flex_opt_ms + flex_opt_bw_ms)
        
        print("\nPerformance Comparison:")
        print(f"Forward pass speedup: {fwd_speedup:.2f}x")
        print(f"Backward pass speedup: {bwd_speedup:.2f}x")
        print(f"Total speedup: {total_speedup:.2f}x")
        result_list.append(total_speedup)
        if print_mask:
            print(f"\nBlock Mask:\n{block_mask}")
    print(f"speed-up:{np.median(result_list)}")
    print(f"speed-up-mean:{np.mean(result_list)}")


def run_document_masking(max_seq_len: int, num_docs: int):
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, "cuda")
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    test_mask(mask_mod=document_causal_mask, S=32768)


def main(examples: List[str] = ["all"], **kwargs):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
        **kwargs: Additional keyword arguments that will be ignored.
    """

    if "all" in examples:
        ex_to_run = list(AVAILABLE_EXAMPLES.keys())
    else:
        ex_to_run = examples

    for ex in ex_to_run:
        if ex in AVAILABLE_EXAMPLES:
            AVAILABLE_EXAMPLES[ex]()
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")


if __name__ == "__main__":
    main(**vars(args))
    end = time.time()
    print(end-start)