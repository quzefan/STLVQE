#include <math.h>
#include <float.h>
// #include "quadrilinear4d_kernel.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
#include <assert.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


__global__ void QuadriLinearForward(const int nthreads, const half* luts, const half* image1, const half* image2, const half* image3, const half* image4, half* output, const int dim, const int shift, const half binsize, const int width, const int height, const int batch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int index_batch = floor(index / (width * height));
        int index_height = floor((index - (index_batch * width * height)) / width);
        int index_width = index - index_batch * width * height - index_height * width;
        // index = index_batch * width * height + index_height * width + index_width
        for (int index_channel = 0; index_channel < 3; index_channel += 1)
        {
            half a = image1[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            half b = image2[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            half c = image3[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            half d = image4[index_batch*3*height*width + index_channel*height*width + index_height*width + index_width];
            
            int a_id = floor(a / binsize);
            int b_id = floor(b / binsize);
            int c_id = floor(c / binsize);
            int d_id = floor(d / binsize);

            half a_d = fmod(a,binsize) / binsize;
            half b_d = fmod(b,binsize) / binsize;
            half c_d = fmod(c,binsize) / binsize;
            half d_d = fmod(d,binsize) / binsize;
            

            int id0000 = (a_id * dim * dim * dim + b_id * dim * dim + c_id * dim + d_id)*shift*shift;
            int id0100 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + d_id)*shift*shift;
            int id0010 = (a_id * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id0001 = (a_id * dim * dim * dim + b_id * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id0110 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id0011 = (a_id * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;
            int id0101 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id0111 = (a_id * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;

            int id1000 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + c_id * dim + d_id)*shift*shift;
            int id1100 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + d_id)*shift*shift;
            int id1010 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id1001 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id1110 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + d_id)*shift*shift;
            int id1011 = ((a_id + 1) * dim * dim * dim + b_id * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;
            int id1101 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + c_id * dim + (d_id + 1))*shift*shift;
            int id1111 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;

            half w0000 = (1-a_d)*(1-b_d)*(1-c_d)*(1-d_d);
            half w0100 = (1-a_d)*b_d*(1-c_d)*(1-d_d);
            half w0010 = (1-a_d)*(1-b_d)*c_d*(1-d_d);
            half w0001 = (1-a_d)*(1-b_d)*(1-c_d)*d_d;
            half w0110 = (1-a_d)*b_d*c_d*(1-d_d);
            half w0011 = (1-a_d)*(1-b_d)*c_d*d_d;
            half w0101 = (1-a_d)*b_d*(1-c_d)*d_d;
            half w0111 = (1-a_d)*b_d*c_d*d_d;

            half w1000 = a_d*(1-b_d)*(1-c_d)*(1-d_d);
            half w1100 = a_d*b_d*(1-c_d)*(1-d_d);
            half w1010 = a_d*(1-b_d)*c_d*(1-d_d);
            half w1001 = a_d*(1-b_d)*(1-c_d)*d_d;
            half w1110 = a_d*b_d*c_d*(1-d_d);
            half w1011 = a_d*(1-b_d)*c_d*d_d;
            half w1101 = a_d*b_d*(1-c_d)*d_d;
            half w1111 = a_d*b_d*c_d*d_d;

            // 4x4 output pixel
            int output_index = (index_batch*3*height*width + index_channel*height*width + index_height*width + index_width)*shift*shift;
            for (int i = 0; i < shift*shift; i += 1)
            {   
                output[output_index+i] = w0000 * luts[id0000+i] + w0100 * luts[id0100+i] + w0010 * luts[id0010+i] + 
                                w0001 * luts[id0001+i] + w0110 * luts[id0110+i] + w0011 * luts[id0011+i] + 
                                w0101 * luts[id0101+i] + w0111 * luts[id0111+i] +
                                w1000 * luts[id1000+i] + w1100 * luts[id1100+i] + w1010 * luts[id1010+i] + 
                                w1001 * luts[id1001+i] + w1110 * luts[id1110+i] + w1011 * luts[id1011+i] + 
                                w1101 * luts[id1101+i] + w1111 * luts[id1111+i];
            }
        }
    }
}


int QuadriLinearForwardLaucher(const half* luts, const half* image1, const half* image2, const half* image3, const half* image4, half* output, const int luts_dim, const int shift, const float binsize, const int width, const int height, const int batch) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;


    QuadriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0>>>(output_size, luts, image1, image2, image3, image4, output, luts_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

int quadrilinear4d_forward_cuda(torch::Tensor luts, torch::Tensor image1, torch::Tensor image2, torch::Tensor image3, torch::Tensor image4, torch::Tensor output,
                           int luts_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    half * luts_flat = luts.data<half>();
    half * image_flat1 = image1.data<half>();
    half * image_flat2 = image2.data<half>();
    half * image_flat3 = image3.data<half>();
    half * image_flat4 = image4.data<half>();
    half * output_flat = output.data<half>();

    QuadriLinearForwardLaucher(luts_flat, image_flat1, image_flat2, image_flat3, image_flat4, output_flat, luts_dim, shift, binsize, width, height, batch);

    return 1;
}

