"""Compare H3 QK Norm/RoPE with RMSNorm plus eager rotary-96 RoPE.

Example: python benchmark/bench_h3_qknorm_rope.py --case 16464,7
"""

import argparse
import importlib.metadata
import json
import statistics

import torch
import torch.nn.functional as F
from sgl_kernel import fused_inplace_qknorm_rope, rmsnorm
from sgl_kernel.h3_fused import h3_qk_norm_rope_

HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-5


def parse_case(text):
    try:
        rows, heads = map(int, text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be rows,heads") from exc
    if rows <= 0 or heads <= 0:
        raise argparse.ArgumentTypeError("rows and heads must be positive")
    return rows, heads


def elapsed_ms(fn):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def paired_medians(candidate, baseline, reset_candidate, reset_baseline, samples):
    for _ in range(3):
        reset_candidate()
        candidate()
        reset_baseline()
        baseline()
    torch.xpu.synchronize()
    candidate_ms, baseline_ms = [], []
    for i in range(samples):
        order = (
            (
                (baseline, reset_baseline, baseline_ms),
                (candidate, reset_candidate, candidate_ms),
            )
            if i % 2
            else (
                (candidate, reset_candidate, candidate_ms),
                (baseline, reset_baseline, baseline_ms),
            )
        )
        for fn, reset, times in order:
            reset()
            times.append(elapsed_ms(fn))
    return statistics.median(candidate_ms), statistics.median(baseline_ms)


def rotate_half(norm, cache):
    active = norm[..., :ROTARY_DIM]
    left, right = active.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    cos = cache[:, :48].repeat(1, 2).unsqueeze(1)
    sin = cache[:, 48:].repeat(1, 2).unsqueeze(1)
    return torch.cat((active * cos + rotated * sin, norm[..., ROTARY_DIM:]), dim=-1)


def run_case(rows, heads, samples):
    qkv = torch.randn(rows, 3 * heads * HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    candidate_qkv, baseline_qkv = qkv.clone(), qkv.clone()

    def views(buffer):
        return tuple(
            part.view(rows, heads, HEAD_DIM)
            for part in buffer.split(heads * HEAD_DIM, dim=-1)
        )

    candidate_q, candidate_k, _ = views(candidate_qkv)
    baseline_q, baseline_k, _ = views(baseline_qkv)
    original_q, original_k, _ = views(qkv)
    q_weight = torch.randn(HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    angles = torch.randn(rows, ROTARY_DIM // 2, device="xpu")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    positions = torch.arange(rows, device="xpu", dtype=torch.int64)

    def candidate():
        h3_qk_norm_rope_(
            candidate_q, candidate_k, q_weight, k_weight, cache, positions, EPS
        )

    def baseline():
        q_norm = rmsnorm(baseline_q, q_weight, EPS)
        k_norm = rmsnorm(baseline_k, k_weight, EPS)
        baseline_q.copy_(rotate_half(q_norm, cache))
        baseline_k.copy_(rotate_half(k_norm, cache))

    try:
        fused_inplace_qknorm_rope(
            baseline_q,
            baseline_k,
            q_weight,
            k_weight,
            cache.float(),
            positions,
            True,
            EPS,
            HEAD_DIM,
            ROTARY_DIM,
        )
        wheel_fused = "accepted"
    except RuntimeError as exc:
        wheel_fused = str(exc).splitlines()[0]
    candidate()
    q_ref = rotate_half(F.rms_norm(original_q, (HEAD_DIM,), q_weight, EPS), cache)
    k_ref = rotate_half(F.rms_norm(original_k, (HEAD_DIM,), k_weight, EPS), cache)
    torch.testing.assert_close(candidate_q.float(), q_ref.float(), atol=0.03, rtol=0.01)
    torch.testing.assert_close(candidate_k.float(), k_ref.float(), atol=0.03, rtol=0.01)
    q_max_abs = (candidate_q.float() - q_ref.float()).abs().max().item()
    k_max_abs = (candidate_k.float() - k_ref.float()).abs().max().item()
    candidate_ms, baseline_ms = paired_medians(
        candidate,
        baseline,
        lambda: candidate_qkv.copy_(qkv),
        lambda: baseline_qkv.copy_(qkv),
        samples,
    )
    return {
        "rows": rows,
        "heads": heads,
        "head_dim": HEAD_DIM,
        "rotary_dim": ROTARY_DIM,
        "dtype": "bf16",
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "sglang_kernel_xpu": importlib.metadata.version("sglang-kernel-xpu"),
        "wheel_fused": wheel_fused,
        "samples": samples,
        "q_max_abs_vs_eager": q_max_abs,
        "k_max_abs_vs_eager": k_max_abs,
        "candidate_ms": round(candidate_ms, 5),
        "baseline_ms": round(baseline_ms, 5),
        "speedup": round(baseline_ms / candidate_ms, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1117)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if not torch.xpu.is_available():
        parser.error("XPU is required")
    torch.manual_seed(args.seed)
    for rows, heads in args.case or [(128, 7), (16464, 7)]:
        print(json.dumps(run_case(rows, heads, args.samples)), flush=True)


if __name__ == "__main__":
    main()
