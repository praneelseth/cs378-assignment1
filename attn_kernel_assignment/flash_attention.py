"""PyTorch integration for the fused Flash Attention kernel (provided).

flash_attention(q, k, v, causal=False)  -- autograd-capable attention backed
                                           by the student's CUDA forward kernel
                                           and a provided recompute backward.
naive_attention(q, k, v, causal=False)  -- materialized baseline for timing.

Students do NOT modify this file.
"""
import math
import os

import torch

_ext = None


def load_extension(verbose=False):
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        _ext = load(name="flash_attn_ext",
                    sources=[os.path.join(src, "flash_attention_torch.cu")],
                    extra_cuda_cflags=["-O2"], verbose=verbose)
    return _ext


def _reference_probs(q, k, causal):
    scale = 1.0 / math.sqrt(q.size(-1))
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        n = q.size(-2)
        s = s + torch.triu(torch.full((n, n), float("-inf"), device=q.device),
                           diagonal=1)
    return torch.softmax(s, dim=-1)


class FlashAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        out = load_extension().forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v = ctx.saved_tensors
        scale = 1.0 / math.sqrt(q.size(-1))
        p = _reference_probs(q, k, ctx.causal)
        d_v = torch.matmul(p.transpose(-1, -2), d_out)
        d_p = torch.matmul(d_out, v.transpose(-1, -2))
        d_s = p * (d_p - (d_p * p).sum(dim=-1, keepdim=True))
        d_q = torch.matmul(d_s, k) * scale
        d_k = torch.matmul(d_s.transpose(-1, -2), q) * scale
        return d_q, d_k, d_v, None


def flash_attention(q, k, v, causal=False):
    return FlashAttnFunction.apply(q.contiguous(), k.contiguous(),
                                   v.contiguous(), causal)


def naive_attention(q, k, v, causal=False):
    return torch.matmul(_reference_probs(q, k, causal).to(v.dtype), v)
