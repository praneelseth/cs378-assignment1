#!/usr/bin/env bash
# Compile the kernels for the standalone (ctypes) graded tests.
# The PyTorch extension (flash_attention.py) compiles separately and
# automatically on first use.
set -e
mkdir -p build
nvcc -O2 -o build/naive_kernel.so --shared src/naive_attention_kernel.cu -Xcompiler -fPIC
nvcc -O2 -o build/flash_kernel.so --shared src/flash_attention_kernel.cu -Xcompiler -fPIC
echo "KERNELS COMPILED"
