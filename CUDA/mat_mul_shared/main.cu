
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "ticktock.h"



template <int Block_Size,class T>
__global__ void parallel_mul(T * c, T const * a , T const * b, int M, int N, int K) {
    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;

    if(m >= M || n >= N) return;

    int tmp = 0;
    for(int t = 0 ; t < K; t++){
        tmp += a[m * K + t] * b[t * N + n];
    } 

    c[m * N + n] = tmp;
}


template <int Block_Size, class T>
__global__ void parallel_mul_shared(T * c, T const * a , T const * b, int M, int N, int K){
    
    int nRow = threadIdx.y + blockIdx.y * blockDim.y;
    int nCol = threadIdx.x + blockIdx.x * blockDim.x;

    /* 
    shared_mem 用来存储a, b的局部内容
    Asub = A , blockRow , i      
    Bsub = B , i , blockCol

    A row col
    subA = A.stride * BLOCK_SIZE * row  + BLOCK_SIZE * col;
     */
    __shared__ T shTileA[Block_Size][Block_Size];
    __shared__ T shTileB[Block_Size][Block_Size];

    T c_val = 0;

    int nIte = ( K + Block_Size - 1) / Block_Size;
    for(int sub = 0 ; sub < nIte ; sub++){

        // 把数据从global memory读取到shared memory
        shTileA[threadIdx.y][threadIdx.x] = a[nRow * K + sub * Block_Size + threadIdx.x];
        shTileB[threadIdx.y][threadIdx.x] = b[(sub * Block_Size + threadIdx.y) * N + nCol];


        // 第一次同步，等待读取完成
        __syncthreads();

        // 计算子矩阵相乘
        for(int l = 0 ; l < Block_Size; l++){
            c_val += shTileA[threadIdx.y][l] * shTileB[l][threadIdx.x];
        }

        // 第二次同步，等待block中的所有线程完成计算
        __syncthreads();
    }

    c[nRow * N + nCol] = c_val;
}

int main() {
    int m = 1 << 11;
    int k = 1 << 11;
    int n = 1 << 11;
    /* 
    CPU 初始化
    GPU 初始化
    */

    std::vector<int> c_a( m * k );
    std::vector<int> c_b( k * n );
    std::vector<int> c_res(m * n , 0);


    // std::vector<int, CudaAllocator<int>> g_a(m * k);
    // std::vector<int, CudaAllocator<int>> g_b(k * n);
    // std::vector<int, CudaAllocator<int>> g_res(m * n , 0);

    int *g_a, *g_b, *g_res;
    checkCudaErrors(cudaMallocManaged(&g_a, m * k * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&g_b, k * n * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&g_res, m * n * sizeof(int)));

    for (int i = 0; i < m * k; i++) {
        int cur = rand() % 100;
        c_a[i] = cur;
        g_a[i] = cur;
    }

    for (int i = 0; i < k * n; i++) {
        int cur = rand() % 100;
        c_b[i] = cur;
        g_b[i] = cur;
    }

    /* 
    CPU 计算
     */
    
    TICK(cpu_time)
    for(int i = 0 ; i < m; i++){
        for(int j = 0; j < n; j++){
            int c = 0;
            for(int t = 0; t < k; t++){
                c += c_a[i * k + t] * c_b[t * n + j];
            }
            c_res [i * n + j]= c;
        }   
    }
    TOCK(cpu_time)

    

    TICK(gpu_mul);
    parallel_mul_shared<32><<<dim3((m + 32 -1 ) / 32, (n + 32 - 1) / 32, 1), dim3(32, 32, 1)>>>
        (g_res, g_a, g_b,m, n, k);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(gpu_mul);

    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            if (c_res[x * n + y] != g_res[x * n + y]) {
                printf("Wrong At x=%d,y=%d: %d != %d\n", x, y,
                      c_res[x * n + y],  g_res[x * n + y]);
                return -1;
            }
        }
    }

    printf("All Correct!\n");

    checkCudaErrors(cudaFree(g_a));
    checkCudaErrors(cudaFree(g_b));
    checkCudaErrors(cudaFree(g_res));

    return 0;
}
