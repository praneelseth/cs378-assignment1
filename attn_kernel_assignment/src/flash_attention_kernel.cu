/*
Part 3: fused Flash Attention forward kernel.

Follows the FlashAttention paper's notation (Br/Bc block sizes, l = running
softmax denominator, m = running row max) and the one-warp-per-(batch,head)
structure of flash-attention-minimal. S and P are NEVER written to HBM: each
tile of scores exists only in shared memory for the duration of its merge.

You implement three blocks, in order (each is tested cumulatively):
  ASSIGN1_3_1  tile scores + causal mask + tile row max
  ASSIGN1_3_2  tile softmax numerator exp(S - row_m) + tile row sum
  ASSIGN1_3_3  the online-softmax merge into (O, l, m)

Constraints (fine for teaching): Br == Bc == 32 fixed; seq_len a multiple of
32; shared memory = (3*Bc*d + Bc*Br) floats, so head_dim <= 64 stays under the
48 KB default per-block limit on a T4.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void fill_kernel(float *data, float value, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = value;
}

__global__ void flash_attn_fw_kernel(const float *Q, const float *K,
                                     const float *V, const int N, const int d,
                                     const int Tc, const int Tr, const int Bc,
                                     const int Br, const float softmax_scale,
                                     const int causal, float *l, float *m,
                                     float *O) {
  int tx = threadIdx.x; // thread tx handles global query row 
  int bx = blockIdx.x;
  int by = blockIdx.y;  // batch and head index

  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N);

  extern __shared__ float sram[];
  int tile_size = Bc * d;  // size of Qi, Kj, Vj
  float *Qi = sram;
  float *Kj = &sram[tile_size];
  float *Vj = &sram[tile_size * 2];
  float *S = &sram[tile_size * 3];

  for (int j = 0; j < Tc; j++) {
    // Load Kj, Vj to SRAM
    for (int x = 0; x < d; x++) {
      Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
      Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
    }
    __syncthreads();

    for (int i = 0; i < Tr; i++) {
      // Causal tile-skip: thread tx handles global query row (i*Br + tx). If
      // every column of K/V tile j is in that row's future, skip the tile.
      if (causal && (j * Bc > i * Br + tx)) continue;

      for (int x = 0; x < d; x++) {
        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
      }
      float row_m_prev = m[lm_offset + (Br * i) + tx];
      float row_l_prev = l[lm_offset + (Br * i) + tx];

      // --- Step 1: scores for this tile -------------------------------
      // For each of the Bc columns y: S[tx*Bc + y] = softmax_scale *
      // dot(Qi[tx,:], Kj[y,:]); track row_m = max_y S. Causal: mask entry to
      // -INFINITY when global column (j*Bc + y) > global row (i*Br + tx). The
      // diagonal is kept, so row_m stays finite.
      float row_m = -INFINITY;
      // BEGIN ASSIGN1_3_1
      // TODO: your implementation of ASSIGN1_3_1 here
      // END ASSIGN1_3_1

      // --- Step 2: unnormalized softmax of this tile ------------------
      // Overwrite S[tx*Bc + y] with exp(S - row_m) (use __expf) and
      // accumulate row_l = sum_y S[tx*Bc + y].
      float row_l = 0.0f;
      // BEGIN ASSIGN1_3_2
      // TODO: your implementation of ASSIGN1_3_2 here
      // END ASSIGN1_3_2

      // --- Step 3: online softmax merge -------------------------------
      //   m_new = max(row_m_prev, row_m)
      //   l_new = e^{row_m_prev - m_new} * row_l_prev
      //           + e^{row_m - m_new} * row_l
      //   O[row, x] = ( row_l_prev * e^{row_m_prev - m_new} * O[row, x]
      //                 + e^{row_m - m_new} * sum_y S[tx,y] * Vj[y,x] ) / l_new
      // Then write m_new -> m[...], l_new -> l[...].
      // BEGIN ASSIGN1_3_3
      // TODO: your implementation of ASSIGN1_3_3 here
      // END ASSIGN1_3_3
    }
    __syncthreads();
  }
}

extern "C" {
void launch_flash_attn_fw(const float *Q, const float *K, const float *V,
                          float *O, int batch_size, int nhead, int seq_len,
                          int head_dim, int causal, cudaStream_t stream) {
#ifndef BLOCK
#define BLOCK 32
#endif
  const int Bc = BLOCK;
  const int Br = BLOCK;
  const int N = seq_len;
  const int d = head_dim;

  if (N % Bc != 0) {
    printf("launch_flash_attn_fw: seq_len (%d) must be a multiple of %d\n", N,
           Bc);
    return;
  }

  const int Tc = (N + Bc - 1) / Bc;
  const int Tr = (N + Br - 1) / Br;
  const float softmax_scale = 1.0f / sqrtf((float)d);

  size_t qkv_size = (size_t)batch_size * nhead * N * d * sizeof(float);
  size_t lm_size = (size_t)batch_size * nhead * N * sizeof(float);

  float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
  cudaMalloc(&d_Q, qkv_size);
  cudaMalloc(&d_K, qkv_size);
  cudaMalloc(&d_V, qkv_size);
  cudaMalloc(&d_O, qkv_size);
  cudaMalloc(&d_l, lm_size);
  cudaMalloc(&d_m, lm_size);

  cudaMemcpyAsync(d_Q, Q, qkv_size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_K, K, qkv_size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_V, V, qkv_size, cudaMemcpyHostToDevice, stream);

  // O accumulates from 0; l (denominator) from 0; m (running max) from -inf.
  cudaMemsetAsync(d_O, 0, qkv_size, stream);
  cudaMemsetAsync(d_l, 0, lm_size, stream);
  int lm_count = batch_size * nhead * N;
  fill_kernel<<<(lm_count + 255) / 256, 256, 0, stream>>>(d_m, -INFINITY,
                                                          lm_count);

  const int sram_size = (3 * Bc * d + Bc * Br) * sizeof(float);
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  if (sram_size > max_sram_size) {
    printf("launch_flash_attn_fw: requested shared memory %d exceeds device "
           "limit %d (reduce head_dim)\n",
           sram_size, max_sram_size);
    return;
  }

  dim3 grid_dim(batch_size, nhead);
  dim3 block_dim(Bc);

  flash_attn_fw_kernel<<<grid_dim, block_dim, sram_size, stream>>>(
      d_Q, d_K, d_V, N, d, Tc, Tr, Bc, Br, softmax_scale, causal, d_l, d_m,
      d_O);

  cudaMemcpyAsync(O, d_O, qkv_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("launch_flash_attn_fw CUDA error: %s\n", cudaGetErrorString(err));

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);
  cudaFree(d_l);
  cudaFree(d_m);
}
}
