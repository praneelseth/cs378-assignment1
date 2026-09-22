"""Part 4c: sweep the fused kernel over block size (Br=Bc) and sequence
length, reporting achieved TFLOPs/s. For each block size we recompile the
kernel with -DBLOCK=<n> into its own .so, then time it with CUDA events.

Usage:  python3 bench/sweep.py
Requires a CUDA toolchain (nvcc) and a GPU. Produces the grid the report asks
for; correctness is graded separately by grade_flash.py.
"""
import ctypes
import os
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = [8, 16, 32]
SEQLENS = [64, 128, 512, 1024, 4096]
B, H, d = 1, 16, 64


def attn_flops(B, H, N, d):
    # 2 matmuls (QK^T and PV), each 2*N*N*d FLOPs, per (batch, head).
    return 2 * (2.0 * N * N * d) * B * H


def build(block):
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    so = os.path.join(ROOT, "build", f"flash_block{block}.so")
    src = os.path.join(ROOT, "src", "flash_attention_kernel.cu")
    subprocess.run(
        ["nvcc", "-O2", f"-DBLOCK={block}", "-o", so, "--shared", src,
         "-Xcompiler", "-fPIC"],
        check=True, cwd=ROOT)
    lib = ctypes.CDLL(so)
    lib.launch_flash_attn_fw.argtypes = [
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p]
    lib.launch_flash_attn_fw.restype = None
    return lib


def time_ms(lib, Q, K, V, iters=10):
    O = np.zeros_like(Q)
    args = (Q.ravel(), K.ravel(), V.ravel(), O.reshape(-1),
            B, H, Q.shape[2], d, 0, None)
    lib.launch_flash_attn_fw(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        lib.launch_flash_attn_fw(*args)
    return (time.perf_counter() - t0) * 1e3 / iters


def main():
    rng = np.random.default_rng(0)
    print(f"{'N':>7} | " + " | ".join(f"Br=Bc={b:<4}" for b in BLOCKS)
          + "   (TFLOPs/s)")
    print("-" * (10 + 12 * len(BLOCKS)))
    libs = {b: build(b) for b in BLOCKS}
    for N in SEQLENS:
        Q = rng.standard_normal((B, H, N, d), dtype=np.float32)
        K = rng.standard_normal((B, H, N, d), dtype=np.float32)
        V = rng.standard_normal((B, H, N, d), dtype=np.float32)
        cells = []
        for b in BLOCKS:
            ms = time_ms(libs[b], Q, K, V)
            tflops = attn_flops(B, H, N, d) / (ms * 1e-3) / 1e12
            cells.append(f"{tflops:>8.3f}")
        print(f"{N:>7} | " + " | ".join(cells))
    print("\nNote: this teaching kernel uses scalar inner products on CUDA "
          "cores, so absolute TFLOPs/s is low; the report asks about the "
          "*shape* of the grid across (N, block), not peak throughput.")


if __name__ == "__main__":
    main()
