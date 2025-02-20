
extern "C" __global__ void diag_matmul_kernel(const float *A, const float *B, float *C, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int col = 0; col < M; col++) {
            C[row * M + col] = A[row] * B[row * M + col];
        }
    }
}
