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
constexpr int kThreads = 256;
template <typename Index>
class H3IndexedRmsAdaLNKernel;

template <typename Index>
void launch_h3_indexed_rms_adaln(
    sycl::queue& queue,
    const bf16* x,
    const bf16* weight,
    const bf16* shift,
    const bf16* scale,
    const Index* indices,
    bf16* output,
    int64_t rows,
    int64_t hidden,
    int64_t shift_stride,
    int64_t scale_stride,
    int64_t index_stride,
    float eps) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<H3IndexedRmsAdaLNKernel<Index>>(
        sycl::nd_range<1>(sycl::range<1>(rows * kThreads), sycl::range<1>(kThreads)), [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0);
          const int64_t lane = item.get_local_id(0);
          const bf16* x_row = x + row * hidden;
          float squares = 0.0f;
          for (int64_t col = lane; col < hidden; col += kThreads) {
            const float value = static_cast<float>(x_row[col]);
            squares += value * value;
          }
          const float sum = sycl::reduce_over_group(item.get_group(), squares, sycl::plus<float>());
          const float inv_rms = sycl::rsqrt(sum / static_cast<float>(hidden) + eps);
          const int64_t table_row = static_cast<int64_t>(indices[row * index_stride]);
          const bf16* shift_row = shift + table_row * shift_stride;
          const bf16* scale_row = scale + table_row * scale_stride;
          bf16* out_row = output + row * hidden;
          for (int64_t col = lane; col < hidden; col += kThreads) {
            const bf16 normalized =
                static_cast<bf16>(static_cast<float>(x_row[col]) * inv_rms * static_cast<float>(weight[col]));
            const bf16 factor = static_cast<bf16>(1.0f + static_cast<float>(scale_row[col]));
            const bf16 scaled = static_cast<bf16>(static_cast<float>(normalized) * static_cast<float>(factor));
            out_row[col] = static_cast<bf16>(static_cast<float>(scaled) + static_cast<float>(shift_row[col]));
          }
        });
  });
}
}  // namespace

SGL_KERNEL_EXPORT at::Tensor h3_indexed_rms_adaln(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& shift,
    const at::Tensor& scale,
    const at::Tensor& indices,
    double eps) {
  TORCH_CHECK(
      x.is_xpu() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
      "H3 RMS AdaLN requires contiguous BF16 XPU rows");
  const auto rows = x.size(0);
  const auto hidden = x.size(1);
  TORCH_CHECK(rows > 0 && hidden > 0 && hidden <= 8192, "unsupported H3 RMS AdaLN shape");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "H3 RMS AdaLN eps must be positive and finite");
  TORCH_CHECK(
      weight.device() == x.device() && weight.scalar_type() == x.scalar_type() && weight.dim() == 1 &&
          weight.numel() == hidden && weight.is_contiguous(),
      "H3 RMS AdaLN weight mismatch");
  TORCH_CHECK(
      shift.device() == x.device() && scale.device() == x.device() && shift.scalar_type() == x.scalar_type() &&
          scale.scalar_type() == x.scalar_type() && shift.dim() == 2 && scale.dim() == 2 &&
          shift.sizes() == scale.sizes() && shift.size(1) == hidden && shift.stride(1) == 1 && scale.stride(1) == 1,
      "H3 RMS AdaLN modulation table mismatch");
  TORCH_CHECK(
      indices.device() == x.device() && indices.dim() == 1 && indices.numel() == rows &&
          (indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt),
      "H3 RMS AdaLN indices mismatch");
  c10::DeviceGuard guard(x.device());
  auto output = at::empty_like(x);
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  auto* x_ptr = reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>());
  auto* w_ptr = reinterpret_cast<const bf16*>(weight.data_ptr<at::BFloat16>());
  auto* shift_ptr = reinterpret_cast<const bf16*>(shift.data_ptr<at::BFloat16>());
  auto* scale_ptr = reinterpret_cast<const bf16*>(scale.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<bf16*>(output.data_ptr<at::BFloat16>());
  if (indices.scalar_type() == at::kLong) {
    launch_h3_indexed_rms_adaln(
        queue,
        x_ptr,
        w_ptr,
        shift_ptr,
        scale_ptr,
        indices.data_ptr<int64_t>(),
        out_ptr,
        rows,
        hidden,
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        static_cast<float>(eps));
  } else {
    launch_h3_indexed_rms_adaln(
        queue,
        x_ptr,
        w_ptr,
        shift_ptr,
        scale_ptr,
        indices.data_ptr<int32_t>(),
        out_ptr,
        rows,
        hidden,
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        static_cast<float>(eps));
  }
  return output;
}
}  // namespace at::native::xpu
