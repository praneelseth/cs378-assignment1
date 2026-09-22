"""Graded correctness test for the Part 3 fused Flash Attention kernel
(ctypes), plus the HBM-traffic-vs-baseline data for the Part 3 plot.

Usage:
  python3 kernel_tests/grade_flash.py --causal {0,1}
  python3 kernel_tests/grade_flash.py --hbm        # traffic vs naive baseline
"""
import argparse
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
from attention_ref import check_against_reference, make_inputs  # noqa: E402

lib = ctypes.CDLL("build/flash_kernel.so")
lib.launch_flash_attn_fw.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p,
]
lib.launch_flash_attn_fw.restype = None


def run(Q, K, V, causal):
    B, H, N, d = Q.shape
    O = np.zeros_like(Q)
    lib.launch_flash_attn_fw(Q.ravel(), K.ravel(), V.ravel(), O.reshape(-1),
                             B, H, N, d, causal, None)
    return O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--hbm", action="store_true")
    args = ap.parse_args()

    if args.hbm:
        # Fused kernel touches only Q,K,V,O,l,m in HBM -- no (N,N) matrix.
        print(f"{'N':>6} {'naive_MB':>10} {'flash_MB':>10} {'ratio':>8}")
        for N in (512, 1024, 2048, 4096):
            B, H, d = 1, 8, 64
            naive = (3 * B * H * N * N * 4 + 3 * B * H * N * d * 4) / 1e6
            flash = (4 * B * H * N * d * 4 + 2 * B * H * N * 4) / 1e6
            print(f"{N:>6} {naive:>10.1f} {flash:>10.1f} {naive/flash:>8.1f}x")
        return

    fails = 0
    for (B, H, N, d) in [(1, 1, 32, 16), (2, 4, 128, 32), (4, 8, 256, 64),
                         (1, 8, 512, 64)]:
        Q, K, V = make_inputs(B, H, N, d)
        O = run(Q, K, V, args.causal)
        if not check_against_reference(O, Q, K, V, causal=bool(args.causal),
                                       label=f"B={B} H={H} N={N} d={d}"):
            fails += 1
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
