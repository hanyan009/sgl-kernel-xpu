"""Correctness and contract checks for H3 QK Norm/RoPE."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel.h3_fused import can_use_h3_qk_norm_rope, h3_qk_norm_rope_

pytestmark = pytest.mark.arch("xe20")


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
@pytest.mark.parametrize("rows", [128, 16464])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_h3_qk_norm_rope_packed_qkv_views(rows, position_dtype):
    torch.manual_seed(1110 + rows)
    heads, dim, rotary = 7, 128, 96
    qkv = torch.randn(rows, 3 * heads * dim, device="xpu", dtype=torch.bfloat16)
    q, k, v = (part.view(rows, heads, dim) for part in qkv.split(heads * dim, dim=-1))
    v_before = v.clone()
    q_weight = torch.randn(dim, device="xpu", dtype=torch.bfloat16)
    k_weight = torch.randn(dim, device="xpu", dtype=torch.bfloat16)
    angles = torch.randn(rows, rotary // 2, device="xpu")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    positions = torch.randperm(rows, device="xpu").to(position_dtype)
    selected_cache = cache.index_select(0, positions.long())

    def reference(tensor, weight):
        normalized = F.rms_norm(tensor, (dim,), weight, 1e-5)
        active = normalized[..., :rotary]
        left, right = active.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        cosine = selected_cache[:, :48].repeat(1, 2).unsqueeze(1)
        sine = selected_cache[:, 48:].repeat(1, 2).unsqueeze(1)
        return torch.cat(
            (active * cosine + rotated * sine, normalized[..., rotary:]), dim=-1
        )

    expected_q = reference(q, q_weight)
    expected_k = reference(k, k_weight)
    h3_qk_norm_rope_(q, k, q_weight, k_weight, cache, positions, 1e-5)
    torch.testing.assert_close(q.float(), expected_q.float(), atol=0.03, rtol=0.01)
    torch.testing.assert_close(k.float(), expected_k.float(), atol=0.03, rtol=0.01)
    assert torch.equal(v, v_before)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_h3_qk_norm_rope_rejects_invalid_inputs():
    qkv = torch.randn(7, 3 * 7 * 128, device="xpu", dtype=torch.bfloat16)
    q, k, _ = (part.view(7, 7, 128) for part in qkv.split(7 * 128, dim=-1))
    weight = torch.ones(128, device="xpu", dtype=torch.bfloat16)
    cache = torch.ones(7, 96, device="xpu", dtype=torch.bfloat16)
    positions = torch.arange(7, device="xpu", dtype=torch.int64)
    assert not can_use_h3_qk_norm_rope(q, k, weight, weight, cache[:, :64], positions)
    with pytest.raises(ValueError, match="unsupported H3 Q/K norm"):
        h3_qk_norm_rope_(q, k, weight, weight, cache[:, :64], positions, 1e-5)
    with pytest.raises(RuntimeError, match="eps must be positive"):
        h3_qk_norm_rope_(q, k, weight, weight, cache, positions, 0.0)
