import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------
# CUDA 内核实现（支持 half）
# ----------------------------
alibi_bias_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <type_traits>

// 通过编译时宏传入 head 数量，即 generate_alibi_bias 中的 H
#ifndef HEAD_COUNT
#define HEAD_COUNT 1
#endif

template <typename scalar_t>
__global__ void alibi_bias_kernel(const scalar_t* score,
                                  scalar_t* out,
                                  const float* q_idx,
                                  const float* kv_idx,
                                  const float* head,
                                  int B, int H, int Q, int K) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * K;
    if (index < total) {
        // 根据 score 的索引计算四个维度的下标
        int k = index % K;
        int q = (index / K) % Q;
        int head_idx = (index / (K * Q)) % H;
        // b 维度虽然计算出来了，但在本核函数中并不使用
        // int b = index / (K * Q * H);

        // 这里从传入的 h 张量中取出当前 head 的值，
        // 并利用固化的 HEAD_COUNT 计算 scale： scale = exp2(-((h + 1)*8.0/HEAD_COUNT))
        float h_val = head[head_idx];
        float scale = exp2f(-((h_val + 1) * 8.0f / float(HEAD_COUNT)));
        // q_idx 与 kv_idx 均为 1D 浮点张量，分别表示 query 和 key 的位置索引
        float bias = (kv_idx[k] - q_idx[q]) * scale;
        
        // 针对 half 类型需要做特殊转换
#if __CUDA_ARCH__ >= 530
        if (std::is_same<scalar_t, at::Half>::value) {
            // __half2float 与 __float2half 用于 half 与 float 之间的转换
            out[index] = __float2half(__half2float(score[index]) + bias);
        } else {
            out[index] = score[index] + bias;
        }
#else
        out[index] = score[index] + bias;
#endif
    }
}

// C++ 封装函数，对外暴露给 Python 的接口
torch::Tensor alibi_bias_cuda(torch::Tensor score,
                              torch::Tensor b,
                              torch::Tensor h,
                              torch::Tensor q_idx,
                              torch::Tensor kv_idx) {
    // 假设 score 为 4D 张量，形状为 [B, HEAD_COUNT, Q, K]
    const auto B = score.size(0);
    const auto H_dim = score.size(1);  // 应与 HEAD_COUNT 一致
    const auto Q = score.size(2);
    const auto K = score.size(3);
    
    // 创建与 score 同形状和类型的输出张量
    auto out = torch::empty_like(score);
    
    int total = score.numel();
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(score.scalar_type(), "alibi_bias_cuda", ([&] {
        alibi_bias_kernel<scalar_t><<<blocks, threads>>>(
            score.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            q_idx.data_ptr<float>(),
            kv_idx.data_ptr<float>(),
            h.data_ptr<float>(),
            B, H_dim, Q, K
        );
    }));
    
    return out;
}
"""

def generate_alibi_bias(H: int):
    """
    返回一个 alibi bias score_mod 函数，其逻辑与原始 Python 实现一致：
      scale = exp2(-((h + 1)*8.0/H))
      bias = (kv_idx - q_idx) * scale
      返回 score + bias

    参数:
        H: head 数量，会在 CUDA 代码中固化

    返回:
        alibi_bias: 一个函数，输入 (score, b, h, q_idx, kv_idx)，输出经过 alibi bias 修正后的 score 张量
    """
    # 将 H 固化到 CUDA 代码中（通过宏定义 HEAD_COUNT）
    cuda_source_with_H = alibi_bias_cuda_source.format(H=H)
    
    # 通过 load_inline 编译并加载 CUDA 模块
    alibi_bias_module = load_inline(
        name='alibi_bias_module',
        cpp_sources=[],  # 此处只需 CUDA 实现，无需额外 C++ 声明
        cuda_sources=[cuda_source_with_H],
        functions=['alibi_bias_cuda'],
        verbose=True
    )
    
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return alibi_bias_module.alibi_bias_cuda(score, b, h, q_idx, kv_idx)
    
    return alibi_mod
cuda_source_with_H = alibi_bias_cuda_source.format(16)
alibi_bias_module = load_inline(
    name='alibi_bias_module',
    cpp_sources=[],  # 此处只需 CUDA 实现，无需额外 C++ 声明
    cuda_sources=[cuda_source_with_H],
    functions=['alibi_bias_cuda'],
    verbose=True
)
# ----------------------------
# 测试一下
# ----------------------------
if __name__ == "__main__":
    # model = NoopModel().cuda()

    # 准备一些测试张量
    score = torch.randn(5, 3, device='cuda')
    b = torch.randn(1, device='cuda')
    h = torch.randn(1, device='cuda')

    out = alibi_bias_module.alibi_bias_cuda(score, b, h, 0, 0)

    print("输入 score =", score)
    print("输出 out   =", out)
    print("是否相同  =", torch.allclose(score, out))
