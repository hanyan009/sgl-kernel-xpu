"""One native attention schema dispatches both dtypes on the current queue."""

import pytest
import torch
from sgl_kernel import diffusion

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_sdp_schema_and_stream(dtype):
    assert diffusion.is_available()
    names = [
        schema.name
        for schema in torch._C._jit_get_all_schemas()
        if schema.name.startswith("sgl_kernel::diffusion_sdp")
    ]
    assert names == ["sgl_kernel::diffusion_sdp"]

    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        q = torch.ones((1, 33, 3, 128), device="xpu", dtype=dtype)
        v = q * 3
        output = diffusion.sdp(q, q, v)
        delta = (output.float() - v.float()).abs().max()
    stream.synchronize()
    assert output.dtype == dtype and output.shape == q.shape
    assert delta.item() < 0.01


def test_native_sdp_rejects_unsupported_dtype():
    assert diffusion.is_available()
    q = torch.ones((1, 17, 1, 128), device="xpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="only fp16/bf16"):
        diffusion.sdp(q, q, q)
