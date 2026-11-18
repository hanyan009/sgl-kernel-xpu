"""MiniMax-H3 QK RMSNorm and partial RoPE on XPU."""

import torch


def can_use_h3_qk_norm_rope(q, k, q_weight, k_weight, cache, positions) -> bool:
    return (
        q.device.type == "xpu"
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[0] > 0
        and q.shape[1] > 0
        and q.shape[2] == 128
        and q.stride(2) == 1
        and q.stride(1) == 128
        and k.device == q.device
        and k.dtype == q.dtype
        and k.shape == q.shape
        and k.stride(2) == 1
        and k.stride(1) == 128
        and q_weight.device == k_weight.device == q.device
        and q_weight.dtype == k_weight.dtype == q.dtype
        and q_weight.shape == k_weight.shape == (128,)
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and cache.device == q.device
        and cache.dtype == q.dtype
        and cache.ndim == 2
        and cache.shape[1] == 96
        and cache.is_contiguous()
        and positions.device == q.device
        and positions.dtype in (torch.int32, torch.int64)
        and positions.shape == (q.shape[0],)
        and positions.is_contiguous()
    )


def h3_qk_norm_rope_(q, k, q_weight, k_weight, cache, positions, eps):
    if not can_use_h3_qk_norm_rope(q, k, q_weight, k_weight, cache, positions):
        raise ValueError("unsupported H3 Q/K norm + RoPE shape, dtype or layout")
    torch.ops.sgl_kernel.h3_qk_norm_rope(
        q, k, q_weight, k_weight, cache, positions, eps
    )
