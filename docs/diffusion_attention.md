# Diffusion attention

This feature ports the single-dispatch dense attention implementation from
[`feature/v0.4.0-cute-single-dispatch`](https://github.com/intel-sandbox/ecosystem.framework.sgl-kernel-xpu-diffusion/tree/feature/v0.4.0-cute-single-dispatch)
onto `prep/h3-xpu-fusions`. It is independent of the H3 RMSNorm, gated
residual, SiLU and Q/K norm plus RoPE fusion operators already on that branch.

`sgl_kernel.diffusion.sdp(q, k, v)` accepts BMG FP16/BF16 dense non-causal
self-attention with matching devices, dtypes and shapes `[1, L, H, 128]` and
scale `1/sqrt(128)`. Strided inputs are materialized by the native implementation.
Masks, causal attention, GQA, cross-attention and batch sizes greater than one
belong to the framework's other attention routes; invalid native calls fail.

## Build and dispatch

A BMG repository build installs one attention library,
`sgl_kernel/_xe20/libsgl-ops-sycl-DiffusionAttentionXe20.so`, and the Python
wrapper. The only
public native schema is
`sgl_kernel::diffusion_sdp(Tensor q, Tensor k, Tensor v) -> Tensor`.
Native dispatch selects private FP16/BF16 implementations in separate
translation units sharing `DiffusionAttentionXe20Impl.inc`:

| Translation unit | Compilation |
| --- | --- |
| `DiffusionAttentionFp16Xe20.cpp` | O3/NDEBUG, static QK tile bound and deferred row-sum reduction |
| `DiffusionAttentionBf16Xe20.cpp` | O3/NDEBUG, original BF16 reduction schedule |

Both objects are linked into the same library. The FP16 two-edit schedule uses
the default contraction setting: paired investigation found that adding
`-ffp-contract=off` to the two-edit version made all eight measured FP16 shapes
slower by 2.03%–5.50%. The BF16 schedule is retained because the same two
edits made the measured Z-Image BF16 shape 13.06% slower.
Separate kernel tags prevent device-image cache collisions. Attention's
target settings are restored before building other libraries. No manual
dtype-specific build or wheel assembly is needed.

Attention reuses the repository's shared BMG SYCL-TLA pin
`7b56b936b87f799f80dde60c2fd8d11c1128cebb`. The checked build-local overlay
preserves the original Omni dense mainloop/softmax, adapts the newer API,
derives tail-mask key coordinates from the MMA partition, and uses ceil-divided
KV tiles. No additional old-pin dependency or Omni runtime package is needed.

The policy is Q256/KV32/V32/MMA-K32, 16 subgroups, two stages and 256 GRFs.
BF16 `L=4128,H=30` retains its validated MMA-K16 specialization. Kernels
submit work to PyTorch's current XPU queue on the input device.

After preparing the project's oneAPI and Torch build environment, use the
normal `pip install -v .` or wheel build, selecting `DPCPP_SYCL_TARGET=bmg`.
For a focused build without GPU access:

```bash
cmake -S . -B build-attention -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDPCPP_SYCL_TARGET=bmg \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DUSE_MOE=OFF -DUSE_FMHA=OFF -DUSE_GDN=OFF -DUSE_MLA=OFF
IGC_allowDecompose2DBlockFuncs=0 \
  cmake --build build-attention --target sgl-ops-sycl-DiffusionAttentionXe20 -j2
```

After installation, GPU tests are `tests/test_diffusion_attention.py` and
`tests/test_diffusion_sdp_dispatch.py`. They cover both dtypes, tails, packed
views, nondefault streams, the sole schema and invalid contracts.

## Validation provenance

The attention implementation and compile configuration were validated in
commit `d84483f4bcfc6371b88a3560fdfbad26c6edba8d` on the earlier integrated
baseline: two independent full wheel builds, 54 FP32-reference cases per
wheel, existing diffusion regressions, and paired BF16 Z-Image/Qwen-Image
CLI runs in Ubuntu 26.04 with Torch 2.13 XPU, oneAPI 2026.1 and Arc Pro B65.
The paired runs preserved performance and produced pixel-identical images;
no strict speedup or FP16 full-pipeline validation is claimed. Those results
establish the implementation's validation history, not a new full-pipeline
acceptance run of this independent v0.4.0 feature branch. The FP16 two-edit
decision follows the 2026-10-28 SYCL-TLA/CUTE regression report: ten paired
shapes were within -1.40% to +0.95% of the old tree and passed numerical checks.
The original `v0.4.0` attention target built on oneAPI 2026.1 and
23 operator/dispatch tests passed on XPU for both dtypes, including nondefault
streams and non-divisible sequence lengths. After the port to this H3 PR
branch, the focused CUTE target was rebuilt with PyTorch 2.14 XPU on BMG;
the numerical and end-to-end H3 measurements remain separate from the
original Z-Image/Qwen-Image validation history. Full wheel packaging and
latent-held-fixed H3 image quality still need validation before deployment.

The wrapper derives from `omni_xpu_kernel/cute/cute_fmha_torch.cpp` at Omni
base `b9f43d0`, current source revision
`a69d2b903b9cdff9d285bc13e53d41338bd61da8` (Apache-2.0). The generated
SYCL-TLA overlay retains Intel/Codeplay BSD-3-Clause notices.
