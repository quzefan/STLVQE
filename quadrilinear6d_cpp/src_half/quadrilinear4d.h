#ifndef TRILINEAR_H
#define TRILINEAR_H

#include<torch/extension.h>
#include <cuda_fp16.h>
int quadrilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, half binsize, int width, int height, int batch);

//int quadrilinear_backward(torch::Tensor image, torch::Tensor image_grad,torch::Tensor lut, torch::Tensor lut_grad,
//                       int lut_dim, int shift, float binsize, int width, int height, int batch);

#endif
