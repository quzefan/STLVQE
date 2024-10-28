#include <torch/extension.h>
at::Tensor modulated_deform_conv2d_forward_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias,
    at::Tensor offset,
    const int kernel_h,const int kernel_w,const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,const int dilation_w,
    const int group, const int deformable_group,const int in_step,
    const bool with_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("modulated_deform_conv2d", &modulated_deform_conv2d_forward_cuda, "modulated_deform_conv2d_forward_cuda");
  // m.def("backward", &quadrilinear4d_backward_cuda, "Quadrilinear backward");
}