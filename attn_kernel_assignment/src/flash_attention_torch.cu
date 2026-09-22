/*
PyTorch extension wrapper for the fused Flash Attention kernel.

#includes the student's kernel file and launches the __global__ kernel
directly on GPU-resident torch tensors (no host copies). Students do NOT
modify this file; the graded work is in flash_attention_kernel.cu.
*/
#include <torch/extension.h>

#include <cmath>

// Pull in the student's kernel + the fill_kernel helper.
#include "flash_attention_kernel.cu"

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      bool causal) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(Q.dtype() == torch::kFloat32, "this kernel is float32");
  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();

  const int B = Q.size(0), H = Q.size(1), N = Q.size(2), d = Q.size(3);
  const int Bc = 32, Br = 32;
  TORCH_CHECK(N % Bc == 0, "seq_len must be a multiple of 32");

  const int Tc = (N + Bc - 1) / Bc;
  const int Tr = (N + Br - 1) / Br;
  const float scale = 1.0f / std::sqrt((float)d);

  auto O = torch::zeros_like(Q);
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto l = torch::zeros({B, H, N}, opts);
  auto m = torch::full({B, H, N}, -INFINITY, opts);

  const int sram = (3 * Bc * d + Bc * Br) * sizeof(float);
  dim3 grid(B, H), block(Bc);
  flash_attn_fw_kernel<<<grid, block, sram>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
      Tr, Bc, Br, scale, (int)causal, l.data_ptr<float>(),
      m.data_ptr<float>(), O.data_ptr<float>());
  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("forward", &forward, "Flash attention forward (CUDA)");
}
