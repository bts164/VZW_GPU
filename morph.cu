#include <stdio.h>
#include <iostream>
#include <limits>

#include <math_constants.h>
#include <nppdefs.h>

#define MAX_ELEMENT_SIZE 9
// store the element in constant memory for fast broadcast to all threads
__constant__ unsigned char element_ptr[MAX_ELEMENT_SIZE*MAX_ELEMENT_SIZE];

template<typename srcT, typename dstT,
         int BLOCK_WIDTH, int BLOCK_HEIGHT,
         int ELEM_SIZE>
__global__
void laplace_morph_kernel(srcT * __restrict__ src,
                          dstT * __restrict__ dst,
                          unsigned char * __restrict__ elem,
                          int width, int height)
{
    // number of halo pixels around this block
    const int NHALO = ELEM_SIZE / 2;

    // round size up to the nearest multiple of 16 to keep
    // memory aligned and avoid bank conflicts
    const int SHMEM_SIZE = 16*((BLOCK_WIDTH+2*NHALO+15)/16);

    // do calculations in floating point to ensure 4-byte memory alignment
    // in shared memory
    __shared__ float sh_src[BLOCK_HEIGHT+2*NHALO][SHMEM_SIZE];

    // global index of the upper left pixel in the cuda block
    int block_i0 = blockIdx.x * blockDim.x;
    int block_j0 = blockIdx.y * blockDim.y;

    // load image and halo pixels into shared_memory
    for (int j = threadIdx.y; j < (BLOCK_HEIGHT + 2*NHALO); j += blockDim.y) {
        int idx_j = block_j0 - NHALO + j;
#pragma unroll
        for (int i = threadIdx.x; i < (BLOCK_WIDTH + 2*NHALO); i += blockDim.x) {
            int idx_i = block_i0 - NHALO + i;
            if (0 <= idx_i && idx_i < width && 0 <= idx_j && idx_j < height) {
                sh_src[j][i] = src[idx_j*width+idx_i];
            } else {
                sh_src[j][i] = CUDART_NAN_F;
            }
        }
    }
    __syncthreads();

    int global_i = block_i0 + threadIdx.x;
    int global_j = block_j0 + threadIdx.y;
    if (global_i >= width || global_j >= height)
        return;

    // calculate the min/max over the element
    float min = (NPP_MAXABS_32F/10),
        max = -(NPP_MAXABS_32F/10);
    for (int j = 0; j < ELEM_SIZE; ++j) {
        for (int i = 0; i < ELEM_SIZE; ++i) {
            if (!element_ptr[j*ELEM_SIZE+i])
                continue;
            float val = sh_src[threadIdx.y+j][threadIdx.x+i];
            if (isnan(val)) continue;
            if (val > max) max = val;
            if (val < min) min = val;
        }
    }

    // write the result back to global memory
    float result = 0.5 * (max + min - 2 * sh_src[NHALO+threadIdx.y][NHALO+threadIdx.x]) + 0.5;
    if (result < 0.0f) result = 0.0f;
    if (result > 255.0f) result = 255.0f;
    dst[global_j*width+global_i] = result;

    return;
}

template<int ELEM_SIZE, typename srcT, typename dstT>
inline void call_laplace_morph_kernel(
    srcT *src, dstT *dst,
    unsigned char *elem,
    int width, int height)
{
    cudaMemcpyToSymbol(element_ptr, elem, ELEM_SIZE);

    // size calculated using CUDA occupancy calculator
    const int BLOCK_WIDTH = 32;
    const int BLOCK_HEIGHT = 12;
    dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid_dim((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
                  (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    // process Red component
    laplace_morph_kernel<srcT, dstT, BLOCK_WIDTH, BLOCK_HEIGHT, ELEM_SIZE>
        <<<grid_dim, block_dim>>>(src+0*width*height,
                                  dst+0*width*height,
                                  elem, width, height);
    // process Blue component
    laplace_morph_kernel<srcT, dstT, BLOCK_WIDTH, BLOCK_HEIGHT, ELEM_SIZE>
        <<<grid_dim, block_dim>>>(src+1*width*height,
                                  dst+1*width*height,
                                  elem, width, height);
    // process Green component
    laplace_morph_kernel<srcT, dstT, BLOCK_WIDTH, BLOCK_HEIGHT, ELEM_SIZE>
        <<<grid_dim, block_dim>>>(src+2*width*height,
                                  dst+2*width*height,
                                  elem, width, height);
    cudaDeviceSynchronize();
    return;
}

template<typename srcT, typename dstT>
void laplace_morph(srcT *src, dstT *dst,
                   unsigned char *elem, int elem_size,
                   int width, int height)
{
    // Hard code calls to template functions with different element sizes.
    // This allows multiple template kernels to be compiled with optimized size
    // of shared memory required for the element size being used hard coded.

    // The down side obviously is that only a small set of sizes is supported

    // The amount of shared memory in this particular kernel is actually so small
    // that this probably does't matter that much in this case, but this is a common
    // optimization technique used
    switch (elem_size)
    {
    case 1:
        call_laplace_morph_kernel<1>(src, dst, elem, width, height);
        break;
    case 3:
        call_laplace_morph_kernel<3>(src, dst, elem, width, height);
        break;
    case 5:
        call_laplace_morph_kernel<5>(src, dst, elem, width, height);
        break;
    case 7:
        call_laplace_morph_kernel<7>(src, dst, elem, width, height);
        break;
    case 9:
        call_laplace_morph_kernel<9>(src, dst, elem, width, height);
        break;
    default:
        std::cerr << "Error: unsupported element size being used for Morphological Laplacian\n";
        break;
    }
}

// template instantiation
template
void laplace_morph(float *src, unsigned char *dst, unsigned char *elem, int elem_size,
                   int width, int height);
