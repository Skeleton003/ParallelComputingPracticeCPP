#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "ticktock.h"

template <class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    out[y * nx + x] = in[x * ny + y];
}

int main() {
    int nx = 1<<12, ny = 1<<12;
    // std::vector<int, CudaAllocator<int>> in(nx * ny);
    // std::vector<int, CudaAllocator<int>> out(nx * ny);

    int *in;
    checkCudaErrors(cudaMallocManaged(&in, nx * ny * sizeof(int)));
    int *out;
    checkCudaErrors(cudaMallocManaged(&out, nx * ny * sizeof(int)));

    for (int i = 0; i < nx * ny; i++) {
        in[i] = i;
    }

    TICK(parallel_transpose);
    parallel_transpose<<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>
        (out, in, nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_transpose);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            if (out[y * nx + x] != in[x * ny + y]) {
                printf("Wrong At x=%d,y=%d: %d != %d\n", x, y,
                       out[y * nx + x], in[x * ny + y]);
                return -1;
            }
        }
    }

    checkCudaErrors(cudaFree(in));

    checkCudaErrors(cudaFree(out));

    printf("All Correct!\n");
    return 0;
}
