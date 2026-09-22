/*
Part 2: naive attention baseline.

Three separate kernels, each writing its result to HBM:
  1. scores_kernel   S = Q @ K^T * scale        (B,H,N,N) in HBM
  2. softmax_kernel  P = softmax(S, dim=-1)      row-wise, stable (PROVIDED)
  3. output_kernel   O = P @ V                   (B,H,N,d) in HBM

This is the memory-bound baseline the fused kernel is compared against; it is
deliberately simple (one thread per output element, no cuBLAS, no tiling).

You implement two blocks: ASSIGN1_2_1 (scores) and ASSIGN1_2_2 (output).

extern "C" launcher matches the ctypes grader (kernel_tests/grade_naive.py).
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// ---- Kernel 1: S = Q @ K^T * scale, optional causal mask ------------------
// Grid is flattened over (B*H, N, N): one thread per score entry S[b,h,q,k].
//// B is the batch size
// H is the number of attention heads
// N is sequence length
// Every thread computes one element S[bh, q, k]

__global__ void scores_kernel(const float *Q, const float *K, float *S,
                              int BH, int N, int d, float scale, int causal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)BH * N * N;
  if (idx >= total) return;

  int k = idx % N;
  int q = (idx / N) % N;
  int bh = idx / (N * N);

  const float *q_row = Q + ((long)bh * N + q) * d;
  const float *k_row = K + ((long)bh * N + k) * d;

  // BEGIN ASSIGN1_2_1
  // TODO: compute dot = sum_x q_row[x] * k_row[x], scale it, apply the causal
  // mask (S = -inf when k > q), and write S[idx].
  // END ASSIGN1_2_1
}

// ---- Kernel 2: row-wise numerically stable softmax (PROVIDED) -------------
// One thread per row (b,h,q); reads/writes the N entries of that row in S.
__global__ void softmax_kernel(float *S, int BH, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= (long)BH * N) return;
  float *s = S + (long)row * N;

  float m = -INFINITY;
  for (int k = 0; k < N; k++) m = fmaxf(m, s[k]);
  float l = 0.0f;
  for (int k = 0; k < N; k++) {
    float e = __expf(s[k] - m);
    s[k] = e;
    l += e;
  }
  float inv = 1.0f / l;
  for (int k = 0; k < N; k++) s[k] *= inv;
}

// ---- Kernel 3: O = P @ V --------------------------------------------------
// Grid flattened over (B*H, N, d): one thread per output entry O[b,h,q,x].
__global__ void output_kernel(const float *P, const float *V, float *O,
                              int BH, int N, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)BH * N * d;
  if (idx >= total) return;

  int x = idx % d;
  int q = (idx / d) % N;
  int bh = idx / (N * d);

  const float *p_row = P + ((long)bh * N + q) * N;

  // BEGIN ASSIGN1_2_2

  // END ASSIGN1_2_2
}

extern "C" {
void launch_naive_attn_fw(const float *Q, const float *K, const float *V,
                          float *O, int batch_size, int nhead, int seq_len,
                          int head_dim, int causal, cudaStream_t stream) {
  const int BH = batch_size * nhead;
  const int N = seq_len;
  const int d = head_dim;
  const float scale = 1.0f / sqrtf((float)d);

  size_t qkv_size = (size_t)BH * N * d * sizeof(float);
  size_t s_size = (size_t)BH * N * N * sizeof(float);  // the O(N^2) blowup

  float *d_Q, *d_K, *d_V, *d_O, *d_S;
  cudaMalloc(&d_Q, qkv_size);
  cudaMalloc(&d_K, qkv_size);
  cudaMalloc(&d_V, qkv_size);
  cudaMalloc(&d_O, qkv_size);
  cudaMalloc(&d_S, s_size);  // S and P live here, in HBM -- the whole point

  cudaMemcpyAsync(d_Q, Q, qkv_size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_K, K, qkv_size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_V, V, qkv_size, cudaMemcpyHostToDevice, stream);

  int T = 256;
  long n_scores = (long)BH * N * N;
  long n_out = (long)BH * N * d;
  long n_rows = (long)BH * N;

  scores_kernel<<<(n_scores + T - 1) / T, T, 0, stream>>>(d_Q, d_K, d_S, BH, N,
                                                          d, scale, causal);
  softmax_kernel<<<(n_rows + T - 1) / T, T, 0, stream>>>(d_S, BH, N);
  output_kernel<<<(n_out + T - 1) / T, T, 0, stream>>>(d_S, d_V, d_O, BH, N, d);

  cudaMemcpyAsync(O, d_O, qkv_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("launch_naive_attn_fw CUDA error: %s\n", cudaGetErrorString(err));

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);
  cudaFree(d_S);
}
}
