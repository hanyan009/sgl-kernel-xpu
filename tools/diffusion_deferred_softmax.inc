  // Keep lane-local partial sums until the output epilogue.
  CUTLASS_DEVICE
  auto
  softmax(FragS &tS, FragSRow &tS_max, FragSPartialRow &tS_sum) {
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});
    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    auto tS_partial_sum = reduce<1, ReduceMode::Vertical>(tS, sycl::plus<void>{});
    if constexpr (decltype(tS_sum.size())::value % VTiles != 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++)
        tS_sum(i) = tS_sum(i) * group_broadcast(sg, rescale(0), i) + tS_partial_sum(i);
    }
    return cute::make_tuple(rescale, tS_partial_sum);
  }
