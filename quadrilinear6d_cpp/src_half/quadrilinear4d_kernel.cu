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
#include <cuda_fp16.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)

__device__ half fmod(half a, half b)
{
    half t = __habs(__hdiv(a,b));
    t = __hsub(t, hfloor(t));
    half c = __hmul(t, __habs(b));
    return (__hlt(a, __int2half_rd(0))) ? __hsub(__int2half_rd(0), c) : c;   /* if ( a < 0 ) c = 0-c */
}

__global__ void QuadriLinearForward(const int nthreads, const half* luts, const half* tri_index, const half* weight, const half* image1, const half* image2, const half* image3, const half* image4, half* output, const int luts_num, const int dim, const int shift, const half binsize, const int width, const int height, const int batch) {
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
            
            int a_id = __half2int_rd(hfloor(__hdiv(a, binsize)));
            int b_id = __half2int_rd(hfloor(__hdiv(b, binsize)));
            int c_id = __half2int_rd(hfloor(__hdiv(c, binsize)));
            int d_id = __half2int_rd(hfloor(__hdiv(d, binsize)));

            int a_d = __half2int_rd(fmod(a,binsize));
            int b_d = __half2int_rd(fmod(b,binsize));
            int c_d = __half2int_rd(fmod(c,binsize));
            int d_d = __half2int_rd(fmod(d,binsize));

            //TODO
	    int t = __half2int_rd(binsize);
            int tri_index_id = (a_d*t*t*t + b_d*t*t + c_d*t + d_d)*7;
            half sorted_a_d = __hdiv(tri_index[tri_index_id], binsize);
            half sorted_b_d = __hdiv(tri_index[tri_index_id+1], binsize);
            half sorted_c_d = __hdiv(tri_index[tri_index_id+2], binsize);
            half sorted_d_d = __hdiv(tri_index[tri_index_id+3], binsize);
            int index_o1 = __half2int_rd(tri_index[tri_index_id+4]);
            int index_o2 = __half2int_rd(tri_index[tri_index_id+5]);
            int index_o3 = __half2int_rd(tri_index[tri_index_id+6]);

            int index_o1_4 = index_o1 & 1;
            int index_o1_3 = (index_o1 >> 1) & 1;
            int index_o1_2 = (index_o1 >> 2) & 1;
            int index_o1_1 = (index_o1 >> 3) & 1;
            int index_o2_4 = index_o2 & 1;
            int index_o2_3 = (index_o2 >> 1) & 1;
            int index_o2_2 = (index_o2 >> 2) & 1;
            int index_o2_1 = (index_o2 >> 3) & 1;
            int index_o3_4 = index_o3 & 1;
            int index_o3_3 = (index_o3 >> 1) & 1;
            int index_o3_2 = (index_o3 >> 2) & 1;
            int index_o3_1 = (index_o3 >> 3) & 1;

            int id0 = (a_id * dim * dim * dim + b_id * dim * dim + c_id * dim + d_id)*shift*shift;
            int id1 = ((a_id + index_o1_1) * dim * dim * dim + (b_id + index_o1_2) * dim * dim + (c_id + index_o1_3) * dim + (d_id + index_o1_4))*shift*shift;
            int id2 = ((a_id + index_o2_1) * dim * dim * dim + (b_id + index_o2_2) * dim * dim + (c_id + index_o2_3) * dim + (d_id + index_o2_4))*shift*shift;
            int id3 = ((a_id + index_o3_1) * dim * dim * dim + (b_id + index_o3_2) * dim * dim + (c_id + index_o3_3) * dim + (d_id + index_o3_4))*shift*shift;
            int id4 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;

            half w0 = __hsub(__int2half_rd(1), sorted_a_d);
            half w1 = __hsub(sorted_a_d, sorted_b_d);
            half w2 = __hsub(sorted_b_d, sorted_c_d);
            half w3 = __hsub(sorted_c_d, sorted_d_d);
            half w4 = sorted_d_d;

            // 4x4 output pixel
            int lut_step = dim * dim * dim * dim * shift * shift;
            int output_index = (index_batch*3*height*width + index_channel*height*width + index_height*width + index_width)*shift*shift;
            // for each LUT base 
            for (int j = 0; j < luts_num; j += 1)
                // for 4x4 pixel
                for (int i = 0; i < shift*shift; i += 1)
                {   
                    half w = weight[index_batch*luts_num*height*width + j*height*width + index_height*width + index_width];
                    output[output_index+i] = __hadd(output[output_index+i],  __hmul(w, (__hadd(__hadd(__hadd(__hadd(__hmul(w0, luts[j*lut_step+id0+i]), __hmul(w1, luts[j*lut_step+id1+i])),  __hmul(w2, luts[j*lut_step+id2+i])), __hmul(w3, luts[j*lut_step+id3+i])), __hmul(w4, luts[j*lut_step+id4+i])))));
                }
        }
    }
}


int QuadriLinearForwardLaucher(const half* luts, const half* tri_index, const half* weight, const half* image1, const half* image2, const half* image3, const half* image4, half* output, const int luts_num, const int luts_dim, const int shift, const half binsize, const int width, const int height, const int batch) {
    const int kThreadsPerBlock = 512;
    const int output_size = height * width * batch;
    cudaError_t err;


    QuadriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0>>>(output_size, luts, tri_index, weight, image1, image2, image3, image4, output, luts_num,  luts_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

int quadrilinear4d_forward_cuda(torch::Tensor luts, torch::Tensor tri_index, torch::Tensor weight, torch::Tensor image1, torch::Tensor image2, torch::Tensor image3, torch::Tensor image4, torch::Tensor output,
                           int luts_num, int luts_dim, int shift, half binsize, int width, int height, int batch)
{
    // Grab the input tensor
    half * luts_flat = (half*)luts.data<float>();
    half * tri_index_flat = (half*)tri_index.data<float>();
    half * weight_flat = (half*)weight.data<float>();
    half * image_flat1 = (half*)image1.data<float>();
    half * image_flat2 = (half*)image2.data<float>();
    half * image_flat3 = (half*)image3.data<float>();
    half * image_flat4 = (half*)image4.data<float>();
    half * output_flat = (half*)output.data<float>();

    QuadriLinearForwardLaucher(luts_flat, tri_index_flat, weight_flat, image_flat1, image_flat2, image_flat3, image_flat4, output_flat, luts_num, luts_dim, shift, binsize, width, height, batch);

    return 1;
}


/*
__global__ void QuadriLinearBackward(const int nthreads, const half* luts, const half* tri_index, const half* weight, half* weight_grad, const half* image1, const half* image2, const half* image3, const half* image4, const half* output_grad, const int luts_num, const int dim, const int shift, const half binsize, const int width, const int height, const int batch) {
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

            int a_d = fmod(a,binsize);
            int b_d = fmod(b,binsize);
            int c_d = fmod(c,binsize);
            int d_d = fmod(d,binsize);

            //TODO
            int tri_index_id = (a_d*binsize*binsize*binsize + b_d*binsize*binsize + c_d*binsize + d_d)*7;
            half sorted_a_d = tri_index[tri_index_id] / binsize;
            half sorted_b_d = tri_index[tri_index_id+1] / binsize;
            half sorted_c_d = tri_index[tri_index_id+2] / binsize;
            half sorted_d_d = tri_index[tri_index_id+3] / binsize;
            int index_o1 = tri_index[tri_index_id+4];
            int index_o2 = tri_index[tri_index_id+5];
            int index_o3 = tri_index[tri_index_id+6];

            int index_o1_4 = index_o1 & 1;
            int index_o1_3 = (index_o1 >> 1) & 1;
            int index_o1_2 = (index_o1 >> 2) & 1;
            int index_o1_1 = (index_o1 >> 3) & 1;
            int index_o2_4 = index_o2 & 1;
            int index_o2_3 = (index_o2 >> 1) & 1;
            int index_o2_2 = (index_o2 >> 2) & 1;
            int index_o2_1 = (index_o2 >> 3) & 1;
            int index_o3_4 = index_o3 & 1;
            int index_o3_3 = (index_o3 >> 1) & 1;
            int index_o3_2 = (index_o3 >> 2) & 1;
            int index_o3_1 = (index_o3 >> 3) & 1;

            int id0 = (a_id * dim * dim * dim + b_id * dim * dim + c_id * dim + d_id)*shift*shift;
            int id1 = ((a_id + index_o1_1) * dim * dim * dim + (b_id + index_o1_2) * dim * dim + (c_id + index_o1_3) * dim + (d_id + index_o1_4))*shift*shift;
            int id2 = ((a_id + index_o2_1) * dim * dim * dim + (b_id + index_o2_2) * dim * dim + (c_id + index_o2_3) * dim + (d_id + index_o2_4))*shift*shift;
            int id3 = ((a_id + index_o3_1) * dim * dim * dim + (b_id + index_o3_2) * dim * dim + (c_id + index_o3_3) * dim + (d_id + index_o3_4))*shift*shift;
            int id4 = ((a_id + 1) * dim * dim * dim + (b_id + 1) * dim * dim + (c_id + 1) * dim + (d_id + 1))*shift*shift;

            half w0 = (1 - sorted_a_d);
            half w1 = (sorted_a_d - sorted_b_d);
            half w2 = (sorted_b_d - sorted_c_d);
            half w3 = (sorted_c_d - sorted_d_d);
            half w4 = sorted_d_d;

            // 4x4 output pixel
            int lut_step = dim * dim * dim * dim * shift * shift;
            int output_index = (index_batch*3*height*width + index_channel*height*width + index_height*width + index_width)*shift*shift;
            // for each LUT base 
            for (int j = 0; j < luts_num; j += 1)
                // for 4x4 pixel
                for (int i = 0; i < shift*shift; i += 1)
                {   
                    int w_index = index_batch*luts_num*height*width + j*height*width + index_height*width + index_width;
                    // atomicAdd(weight_grad + w_index, output_grad[output_index+i]*(w0 * luts[j*lut_step+id0+i] + w1 * luts[j*lut_step+id1+i] + w2 * luts[j*lut_step+id2+i] + w3 * luts[j*lut_step+id3+i] + w4 * luts[j*lut_step+id4+i]));
                    weight_grad[w_index] += output_grad[output_index+i]*(w0 * luts[j*lut_step+id0+i] + w1 * luts[j*lut_step+id1+i] + w2 * luts[j*lut_step+id2+i] + w3 * luts[j*lut_step+id3+i] + w4 * luts[j*lut_step+id4+i]);
                    // output[output_index+i] += w*(w0 * luts[j*lut_step+id0+i] + w1 * luts[j*lut_step+id1+i] + w2 * luts[j*lut_step+id2+i] + w3 * luts[j*lut_step+id3+i] + w4 * luts[j*lut_step+id4+i]);
                }
        }
    }
}
*/
int QuadriLinearBackwardLaucher(const half* luts, const half* tri_index, const half* weight, half* weight_grad, const half* image1, const half* image2, const half* image3, const half* image4, const half* output_grad, const int luts_num, const int luts_dim, const int shift, const half binsize, const int width, const int height, const int batch) {
/*    const int kThreadsPerBlock = 512;
    const int output_size = height * width * batch;
    cudaError_t err;


    QuadriLinearBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0>>>(output_size, luts, tri_index, weight, weight_grad, image1, image2, image3, image4, output_grad, luts_num, luts_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }*/

    return 1;
}

int quadrilinear4d_backward_cuda(torch::Tensor luts, torch::Tensor tri_index, torch::Tensor weight, torch::Tensor weight_grad, torch::Tensor image1, torch::Tensor image2, torch::Tensor image3, torch::Tensor image4, torch::Tensor output_grad,
                           int lut_num, int lut_dim, int shift, half binsize, int width, int height, int batch)
{
    // Grab the input tensor
    /*
    half * luts_flat = luts.data<half>();
    half * tri_index_flat = tri_index.data<half>();
    half * weight_flat = weight.data<half>();
    half * weight_grad_flat = weight_grad.data<half>();
    half * image_flat1 = image1.data<half>();
    half * image_flat2 = image2.data<half>();
    half * image_flat3 = image3.data<half>();
    half * image_flat4 = image4.data<half>();
    half * output_grad_flat = output_grad.data<half>();

    QuadriLinearBackwardLaucher(luts_flat, tri_index_flat, weight_flat, weight_grad_flat, image_flat1, image_flat2, image_flat3, image_flat4, output_grad_flat, lut_num, lut_dim, shift, binsize, width, height, batch);
*/
    return 1;
}
