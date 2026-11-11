"""MiniMax-H3 native XPU fusions (SYCL device kernels)."""

import torch


def can_use_h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices) -> bool:
    return (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[0] > 0
        and 0 < x.shape[1] <= 8192
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[1],)
        and weight.is_contiguous()
        and scale.device == shift.device == x.device
        and scale.dtype == shift.dtype == x.dtype
        and scale.ndim == shift.ndim == 2
        and scale.shape == shift.shape
        and scale.shape[1] == x.shape[1]
        and scale.stride(1) == shift.stride(1) == 1
        and indices.device == x.device
        and indices.ndim == 1
        and indices.numel() == x.shape[0]
        and indices.dtype in (torch.int32, torch.int64)
    )


def h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, eps):
    if not can_use_h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices):
        raise ValueError("unsupported H3 RMSNorm/AdaLN shape, dtype, layout or device")
    return torch.ops.sgl_kernel.h3_indexed_rms_adaln(
        x, weight, shift, scale, indices, eps
    )


def can_use_h3_indexed_gate(x, gate, other, indices) -> bool:
    return (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[0] > 0
        and 0 < x.shape[1] <= 8192
        and other.device == x.device
        and other.dtype == x.dtype
        and other.shape == x.shape
        and other.is_contiguous()
        and gate.device == x.device
        and gate.dtype == x.dtype
        and gate.ndim == 2
        and gate.shape[1] == x.shape[1]
        and gate.stride(1) == 1
        and indices.device == x.device
        and indices.ndim == 1
        and indices.numel() == x.shape[0]
        and indices.dtype in (torch.int32, torch.int64)
    )


def h3_indexed_gate(x, gate, other, indices, *, inplace: bool = False):
    if not can_use_h3_indexed_gate(x, gate, other, indices):
        raise ValueError("unsupported H3 indexed gate shape, dtype, layout or device")
    out = x if inplace else torch.empty_like(x)
    torch.ops.sgl_kernel.h3_indexed_gate(out, x, gate, other, indices)
    return out


def can_use_h3_silu_mul(x) -> bool:
    return (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] % 2 == 0
        and 0 < x.shape[1] // 2 <= 16384
    )


def h3_silu_mul(x):
    if not can_use_h3_silu_mul(x):
        raise ValueError("unsupported H3 SiLU x up shape, dtype, layout or device")
    return torch.ops.sgl_kernel.h3_silu_mul(x)
