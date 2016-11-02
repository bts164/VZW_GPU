#include <stdio.h>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

template<typename srcT, typename dstT>
int yuv2rgb(srcT *src, dstT *dst, int width, int height);

template<typename srcT, typename dstT>
void laplace_morph(srcT *src, dstT *dst, unsigned char *elem, int elem_size,
                   int width, int height);

int main(int argc, char *argv[])
{
    if (2 > argc) {
        std::cout << "Please specify the input image as the first argument\n";
        return 1;
    }

    if (3 > argc) {
        std::cout << "Please specify the input image width as the second argument\n";
        return 1;
    }

    if (4 > argc) {
        std::cout << "Please specify the input image height as the third argument\n";
        return 1;
    }

    // some constants
    const int width = atoi(argv[2]);     // image width
    const int height = atoi(argv[3]);    // image height
    const int N = 5;                     // mask size
    const int R = (N + 1) / 2;           // mask radius (rounded up)

    // sizes of the binary data
    size_t yuv_size = 3 * width * height / 2;
    size_t rgb_size = 3 * width * height;

    // try to open the input file
    FILE *fin = fopen(argv[1], "rb");
    if (NULL == fin) {
        std::cout << "Error: Could not open file '" << argv[1] << "'\n";
        return 1;
    }

    // initialize the NxN disk element mask in host memory
    std::vector<unsigned char> mask(N*N, 0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mask[i*N+j] = (i - R + 1) * (i - R + 1) +
                (j - R + 1) * (j - R + 1) <=
                (R - 1) * (R - 1) ? 1 : 0;
        }
    }

    // read the yuv data into a host array
    std::vector<unsigned char> h_yuv(yuv_size);
    fread(h_yuv.data(), 1, yuv_size, fin);
    fclose(fin);

    // copy the yuv data to device
    unsigned char *d_yuv = NULL;
    cudaMalloc((void**)&d_yuv, yuv_size);
    cudaMemcpy(d_yuv, h_yuv.data(), yuv_size, cudaMemcpyHostToDevice);

    // allocate device rgb buffer. intermediate calculations are done
    // in floating point to preserve accuracy
    float *d_rgb = NULL;
    cudaMalloc((void**)&d_rgb, rgb_size * sizeof(float));
    cudaMemset(d_rgb, 0, rgb_size * sizeof(float));

    // allocate the output rgb buffer
    unsigned char *d_out = NULL;
    cudaMalloc((void**)&d_out, rgb_size * sizeof(float));
    cudaMemset(d_out, 0, rgb_size * sizeof(float));

    // initialize the timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // process the image
    cudaEventRecord(start);
    yuv2rgb(d_yuv, d_rgb, width, height);
    laplace_morph(d_rgb, d_out, mask.data(), 5, width, height);
    cudaEventRecord(stop);

    // update the timers
    cudaEventSynchronize(stop);
    float mSecs = 0.0;
    cudaEventElapsedTime(&mSecs, start, stop);

    // print a few performance metrics
    printf("Finished!!!\n");
    printf("Elapsed running time was %.2f ms\n", mSecs);
    printf("Throughput was %.2f MPixels/s or %.2f frames/s\n",
           (width * height) / mSecs, 1e3/mSecs);

    // copy planar rgb data back to host
    std::vector<unsigned char> h_rgb(rgb_size);
    cudaMemcpy(h_rgb.data(), d_out, rgb_size, cudaMemcpyDeviceToHost);

#ifdef OUTPUT_PACKED_RGB
    // This is useful if you want to run the output image through ffmpeg to
    // generate a jpg image. The version of ffmpeg I was using only supports
    // packed rgb, not planar. This is a little slow, so just disable it normally

    // convert planar index to packed index
    auto planar2packed = [width, height](int idx) -> int {
        int comp = idx / (width * height); idx -= comp * width * height;
        int row = idx / width; idx -= row * width;
        int col = idx;
        return 3 * (row * width + col) + comp;
    };

    // gather planar rgb data into packed rgb vector
    std::vector<int> permutation(3 * width * height);
    thrust::transform(thrust::host, thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(3*width*height),
                      permutation.begin(), planar2packed);
    std::vector<unsigned char> packed_rgb(3*width*height);
    thrust::gather(thrust::host, permutation.begin(), permutation.end(),
                   h_rgb.data(), packed_rgb.begin());

    // write out packed rgb data
    FILE *fout = fopen("rgbdata.dat", "wb");
    fwrite(packed_rgb.data(), 1, rgb_size, fout);
    fclose(fout);
#else
    // write out planar rgb data
    FILE *fout = fopen("rgbdata.dat", "wb");
    fwrite(h_rgb.data(), 1, rgb_size, fout);
    fclose(fout);
#endif

    // free the allocated buffers
    cudaFree(d_yuv);
    cudaFree(d_rgb);
    cudaFree(d_out);
    
    return 0;
}
