#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int kThreads = 256;
class H3SiluMulKernel;
}  // namespace

SGL_KERNEL_EXPORT at::Tensor h3_silu_mul(const at::Tensor& input) {
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.is_contiguous() &&
          input.size(0) > 0 && input.size(1) > 0 && input.size(1) % 2 == 0 && input.size(1) <= 32768,
      "H3 SiLU x up requires contiguous packed BF16 XPU rows");
  const int64_t rows = input.size(0);
  const int64_t hidden = input.size(1) / 2;
  c10::DeviceGuard guard(input.device());
  auto output = at::empty({rows, hidden}, input.options());
  const auto* x = reinterpret_cast<const bf16*>(input.data_ptr<at::BFloat16>());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr<at::BFloat16>());
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<H3SiluMulKernel>(
        sycl::nd_range<1>(sycl::range<1>(rows * kThreads), sycl::range<1>(kThreads)), [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0);
          for (int64_t col = item.get_local_id(0); col < hidden; col += kThreads) {
            const int64_t offset = row * 2 * hidden + col;
            const float gate = static_cast<float>(x[offset]);
            const float up = static_cast<float>(x[offset + hidden]);
            const bf16 activated = static_cast<bf16>(gate / (1.0f + sycl::exp(-gate)));
            out[row * hidden + col] = static_cast<bf16>(static_cast<float>(activated) * up);
          }
        });
  });
  return output;
}
}  // namespace at::native::xpu
