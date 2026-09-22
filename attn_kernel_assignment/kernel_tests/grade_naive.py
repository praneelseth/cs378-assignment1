"""Graded correctness test for the Part 2 naive baseline (ctypes), plus the
HBM-traffic / runtime table the report asks for.

Usage:
  python3 kernel_tests/grade_naive.py --causal {0,1}
  python3 kernel_tests/grade_naive.py --bench      # timing + HBM-bytes table
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
from attention_ref import check_against_reference, make_inputs  # noqa: E402

lib = ctypes.CDLL("build/naive_kernel.so")
lib.launch_naive_attn_fw.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p,
]
lib.launch_naive_attn_fw.restype = None


def run(Q, K, V, causal):
    B, H, N, d = Q.shape
    O = np.zeros_like(Q)
    lib.launch_naive_attn_fw(Q.ravel(), K.ravel(), V.ravel(), O.reshape(-1),
                             B, H, N, d, causal, None)
    return O


def hbm_bytes(B, H, N, d):
    """Dominant HBM traffic: S written once + P (=S) read once + reread for PV.
    S/P is (B,H,N,N) floats; that O(N^2) term is the point of the table.
    """
    s = B * H * N * N * 4
    qkv = 3 * B * H * N * d * 4
    return 2 * s + s + qkv  # write S, read S in softmax(+write), read P in PV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    if args.bench:
        print(f"{'N':>6} {'runtime_ms':>12} {'HBM_MB':>10}")
        for N in (512, 1024, 2048, 4096):
            Q, K, V = make_inputs(1, 8, N, 64)
            run(Q, K, V, 0)  # warmup
            t0 = time.perf_counter()
            run(Q, K, V, 0)
            dt = (time.perf_counter() - t0) * 1e3
            mb = hbm_bytes(1, 8, N, 64) / 1e6
            print(f"{N:>6} {dt:>12.2f} {mb:>10.1f}")
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
