#include <stdio.h>

template<typename srcT, typename dstT>
__global__
void yuv2rgb_kernel(srcT *src, dstT *dst, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;

    int yIdx = j * width + i;
    int uvIdx = (j/2) * (width/2) + i/2;
    srcT *Y = src;
    srcT *U = Y + width * height;
    srcT *V = U + width * height / 4;
    dstT *R = dst;
    dstT *G = R + width * height;
    dstT *B = G + width * height;
    R[yIdx] = Y[yIdx] + 1.370705 * (V[uvIdx] - 128.0);
    G[yIdx] = Y[yIdx] - 0.698001 * (V[uvIdx] - 128.0) - 0.337633 * (U[uvIdx] - 128.0);
    B[yIdx] = Y[yIdx] + 1.732446 * (U[uvIdx] - 128.0);
}

template<typename srcT, typename dstT>
int yuv2rgb(srcT *src, dstT *dst, int width, int height)
{
    dim3 blockSize(32, 12);
    dim3 nBlocks((width+blockSize.x-1)/blockSize.x,
                 (height+blockSize.y-1)/blockSize.y);
    yuv2rgb_kernel<<<nBlocks, blockSize>>>(src, dst, width, height);
    cudaDeviceSynchronize();
    return 0;
}

template
int yuv2rgb(unsigned char *src, float *dst, int width, int height);
