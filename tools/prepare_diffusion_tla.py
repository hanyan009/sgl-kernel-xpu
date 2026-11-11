"""Prepare the dense noncausal FP16/BF16 attention overlay."""

import argparse
import shutil
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"attention overlay pattern changed: {old[:80]}")
    return text.replace(old, new)


def prepare(source, destination):
    node = Path(__file__).resolve().parent
    source = Path(source)
    destination = Path(destination)
    relative = Path("flash_attention_v2/collective")
    root = source / "applications" / relative
    text = (root / "xe_fmha_fwd_mainloop.hpp").read_text()
    partial = "  using FragSPartialRow = decltype(reduce<1, ReduceMode::Vertical>(FragS{}, sycl::plus<void>{}));"
    if text.count(partial) != 1:
        raise RuntimeError("current pin partial-sum declaration changed")
    text = text.replace(
        partial,
        """#if !SGLD_ATTENTION_TWO_EDIT
  using FragSPartialRow = FragSRow;
#else
  using FragSPartialRow = decltype(reduce<1, ReduceMode::Vertical>(FragS{}, sycl::plus<void>{}));
#endif""",
    )
    begin = text.index(
        "    using namespace sycl::ext::oneapi::this_work_item;",
        text.index("  operator()"),
    )
    end = text.index("\n  // Single step of blocked softmax.", begin)
    body = (node / "diffusion_dense_body.inc").read_text().rstrip()
    start_mask = body.index("          FragSRow k_rem_mask;")
    end_mask = body.index("\n        }", start_mask)
    body = (
        body[:start_mask]
        + """          // Use CuTe's MMA coordinates for every key in a KV32 tail.
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
                                 make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            if (get<1>(cS_thread(i)) >= seq_len) tSrS(i) = ElementS(-INFINITY);
          }"""
        + body[end_mask:]
    )
    body = (
        """    static_assert(!CausalMask && !CachedKV && !PagedKV && !BlockScale &&
                  !F8kvF16mma && !PerTensorScale && !FP4Input,
                  "Only dense noncausal unscaled FP16/BF16 is supported");
"""
        + body
    )
    deferred_body = replace_once(
        body,
        "for (int D = 0; D < size<4>(tKgK); D++) {",
        "for (int D = 0; D < DTiles; D++) {",
    )
    deferred_body = replace_once(
        deferred_body,
        "auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);",
        """auto [rescale, tS_partial_sum] = softmax(tSrS, tA_max, tA_sum);
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      constexpr int kSumSize = decltype(tA_sum.size())::value;
      constexpr bool kSumDivVT = (kSumSize % VTiles == 0);
      constexpr int kSumPerVT = kSumDivVT ? (kSumSize / VTiles) : 0;""",
    )
    deferred_body = replace_once(
        deferred_body,
        "        cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));",
        """        if constexpr (kSumDivVT) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kSumPerVT; j++) {
            int const i = VV * kSumPerVT + j;
            tA_sum(i) = tA_sum(i) * group_broadcast(sg, rescale(0), i) + tS_partial_sum(i);
          }
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));""",
    )
    soft_start = end
    soft_end = text.index("\n};", soft_start)
    text = (
        text[:begin]
        + "#if !SGLD_ATTENTION_TWO_EDIT\n"
        + body
        + "\n  }\n\n"
        + (node / "diffusion_softmax.inc").read_text()
        + "#else\n"
        + deferred_body
        + "\n  }\n\n"
        + (node / "diffusion_deferred_softmax.inc").read_text()
        + "#endif\n"
        + text[soft_end:]
    )
    output = destination / relative
    output.mkdir(parents=True, exist_ok=True)
    (output / "xe_fmha_fwd_mainloop.hpp").write_text(text)
    shutil.copyfile(root / "fmha_fusion.hpp", output / "fmha_fusion.hpp")
    kernel = (
        source / "applications/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
    ).read_text()
    floor = "k_blocks = static_cast<int>(static_cast<unsigned>(seq_len) / static_cast<unsigned>(get<1>(TileShapeQK{})));"
    if kernel.count(floor) != 1:
        raise RuntimeError("current pin dense block count changed")
    kernel = kernel.replace(
        floor, "k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));"
    )
    epilogue = "      epilogue(O(_,_,head_q,l_coord),"
    if kernel.count(epilogue) != 1:
        raise RuntimeError("current pin dense epilogue call changed")
    kernel = kernel.replace(
        epilogue,
        """#if !SGLD_ATTENTION_TWO_EDIT
      epilogue.template operator()<true>(O(_,_,head_q,l_coord),
#else
      epilogue(O(_,_,head_q,l_coord),
#endif""",
    )
    kernel_output = destination / "flash_attention_v2/kernel"
    kernel_output.mkdir(parents=True, exist_ok=True)
    (kernel_output / "xe_fmha_fwd_kernel.hpp").write_text(kernel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    prepare(args.source, args.destination)
