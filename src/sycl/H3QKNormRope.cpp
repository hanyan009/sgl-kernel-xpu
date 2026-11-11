#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int kHeadDim = 128;
constexpr int kRotaryDim = 96;
template <typename Index>
class H3QKNormRopeKernel;

template <typename Index>
void launch_h3_qk_norm_rope(
    sycl::queue& queue,
    bf16* q,
    bf16* k,
    const bf16* qw,
    const bf16* kw,
    const bf16* cache,
    const Index* positions,
    int64_t rows,
    int64_t heads,
    int64_t q_row_stride,
    int64_t k_row_stride,
    float eps) {
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> qnorm(sycl::range<1>(kHeadDim), cgh);
    sycl::local_accessor<float, 1> knorm(sycl::range<1>(kHeadDim), cgh);
    cgh.parallel_for<H3QKNormRopeKernel<Index>>(
        sycl::nd_range<1>(sycl::range<1>(rows * heads * kHeadDim), sycl::range<1>(kHeadDim)),
        [=](sycl::nd_item<1> item) {
          const int64_t group = item.get_group(0);
          const int64_t row = group / heads;
          const int64_t head = group % heads;
          const int64_t col = item.get_local_id(0);
          const int64_t qo = row * q_row_stride + head * kHeadDim + col;
          const int64_t ko = row * k_row_stride + head * kHeadDim + col;
          const float qv = static_cast<float>(q[qo]);
          const float kv = static_cast<float>(k[ko]);
          const float qs = sycl::reduce_over_group(item.get_group(), qv * qv, sycl::plus<float>());
          const float ks = sycl::reduce_over_group(item.get_group(), kv * kv, sycl::plus<float>());
          qnorm[col] = static_cast<float>(
              static_cast<bf16>(qv * sycl::rsqrt(qs / kHeadDim + eps) * static_cast<float>(qw[col])));
          knorm[col] = static_cast<float>(
              static_cast<bf16>(kv * sycl::rsqrt(ks / kHeadDim + eps) * static_cast<float>(kw[col])));
          sycl::group_barrier(item.get_group());
          if (col < kRotaryDim) {
            const int64_t partner = col < kRotaryDim / 2 ? col + kRotaryDim / 2 : col - kRotaryDim / 2;
            const float sign = col < kRotaryDim / 2 ? -1.0f : 1.0f;
            const int64_t pos = static_cast<int64_t>(positions[row]);
            const float cosine = static_cast<float>(cache[pos * kRotaryDim + col % (kRotaryDim / 2)]);
            const float sine = static_cast<float>(cache[pos * kRotaryDim + kRotaryDim / 2 + col % (kRotaryDim / 2)]);
            const bf16 qleft = static_cast<bf16>(qnorm[col] * cosine);
            const bf16 qright = static_cast<bf16>(sign * qnorm[partner] * sine);
            const bf16 kleft = static_cast<bf16>(knorm[col] * cosine);
            const bf16 kright = static_cast<bf16>(sign * knorm[partner] * sine);
            q[qo] = static_cast<bf16>(static_cast<float>(qleft) + static_cast<float>(qright));
            k[ko] = static_cast<bf16>(static_cast<float>(kleft) + static_cast<float>(kright));
          } else {
            q[qo] = static_cast<bf16>(qnorm[col]);
            k[ko] = static_cast<bf16>(knorm[col]);
          }
        });
  });
}
}  // namespace

SGL_KERNEL_EXPORT void h3_qk_norm_rope(
    at::Tensor& q,
    at::Tensor& k,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& cache,
    const at::Tensor& positions,
    double eps) {
  TORCH_CHECK(
      q.is_xpu() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(0) > 0 && q.size(1) > 0 &&
          q.size(2) == kHeadDim && q.stride(2) == 1 && q.stride(1) == kHeadDim,
      "H3 Q/K norm + RoPE requires BF16 XPU [tokens, heads, 128] Q");
  TORCH_CHECK(
      k.device() == q.device() && k.scalar_type() == q.scalar_type() && k.sizes() == q.sizes() && k.stride(2) == 1 &&
          k.stride(1) == kHeadDim,
      "H3 Q/K norm + RoPE K mismatch");
  TORCH_CHECK(
      q_weight.device() == q.device() && k_weight.device() == q.device() && q_weight.scalar_type() == q.scalar_type() &&
          k_weight.scalar_type() == q.scalar_type() && q_weight.sizes() == at::IntArrayRef{kHeadDim} &&
          k_weight.sizes() == at::IntArrayRef{kHeadDim} && q_weight.is_contiguous() && k_weight.is_contiguous(),
      "H3 Q/K norm + RoPE weight mismatch");
  TORCH_CHECK(
      cache.device() == q.device() && cache.scalar_type() == q.scalar_type() && cache.dim() == 2 &&
          cache.size(1) == kRotaryDim && cache.is_contiguous(),
      "H3 Q/K norm + RoPE cache mismatch");
  TORCH_CHECK(
      positions.device() == q.device() && positions.dim() == 1 && positions.numel() == q.size(0) &&
          positions.is_contiguous() && (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong),
      "H3 Q/K norm + RoPE positions mismatch");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "H3 Q/K norm + RoPE eps must be positive and finite");
  c10::DeviceGuard guard(q.device());
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  auto* qp = reinterpret_cast<bf16*>(q.data_ptr<at::BFloat16>());
  auto* kp = reinterpret_cast<bf16*>(k.data_ptr<at::BFloat16>());
  const auto* qwp = reinterpret_cast<const bf16*>(q_weight.data_ptr<at::BFloat16>());
  const auto* kwp = reinterpret_cast<const bf16*>(k_weight.data_ptr<at::BFloat16>());
  const auto* cp = reinterpret_cast<const bf16*>(cache.data_ptr<at::BFloat16>());
  if (positions.scalar_type() == at::kLong) {
    launch_h3_qk_norm_rope(
        queue,
        qp,
        kp,
        qwp,
        kwp,
        cp,
        positions.data_ptr<int64_t>(),
        q.size(0),
        q.size(1),
        q.stride(0),
        k.stride(0),
        static_cast<float>(eps));
  } else {
    launch_h3_qk_norm_rope(
        queue,
        qp,
        kp,
        qwp,
        kwp,
        cp,
        positions.data_ptr<int32_t>(),
        q.size(0),
        q.size(1),
        q.stride(0),
        k.stride(0),
        static_cast<float>(eps));
  }
}
}  // namespace at::native::xpu
