"""Accuracy checks for MiniMax-H3 XPU fusion kernels."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel.h3_fused import h3_rmsnorm_indexed_adaln


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_h3_rmsnorm_indexed_adaln_real_table_stride():
    torch.manual_seed(1110)
    rows, hidden = 128, 5376
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="xpu", dtype=torch.bfloat16)
    packed = torch.randn(3, 6 * hidden, device="xpu", dtype=torch.bfloat16)
    shift, scale = packed.chunk(6, dim=-1)[:2]
    indices = torch.randint(0, 3, (rows,), dtype=torch.int64).to("xpu")
    expected = F.rms_norm(x, (hidden,), weight, 1e-5)
    expected = expected * (1 + scale.index_select(0, indices))
    expected = expected + shift.index_select(0, indices)

    actual = h3_rmsnorm_indexed_adaln(x, weight, shift, scale, indices, 1e-5)
    # A few BF16 rounding differences remain in the fused reduction.
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.125, rtol=0.01)
