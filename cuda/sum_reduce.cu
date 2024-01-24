#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

const int N = 1<<20;  //1<<14
const int BLOCK_SIZE = 1024;

__global__ void sumReduce(int *a, int *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sum[BLOCK_SIZE];
    sum[threadIdx.x]=a[tid];
    __syncthreads();
    for(int s=1;s<blockDim.x;s*=2)
    {
        int index=2*s*threadIdx.x;
        if(index<blockDim.x)
        {
            sum[index]+=sum[index+s];
        }
        __syncthreads();
    }
    if(threadIdx.x==0)
    {
        b[blockIdx.x]=sum[0];
    }

    // Uncomment for single kernel call

    // __syncthreads();
    // if(blockIdx.x==0)
    // {
    //     sum[threadIdx.x]=b[tid];
    //     __syncthreads();
    //     for(int s=1;s<blockDim.x;s*=2)
    //     {
    //         int index=2*s*threadIdx.x;
    //         if(index<blockDim.x)
    //         {
    //             sum[index]+=sum[index+s];
    //         }
    //         __syncthreads();
    //     }
    //     if(threadIdx.x==0)
    //     {
    //         b[blockIdx.x]=sum[0];
    //     }
    // }
}

int main() {
    std::vector<int> h_a(N, 1);
    std::vector<int> h_b(N);

    int *d_a, *d_b;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));

    cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = BLOCK_SIZE;
    int gridSize = N/blockSize;


    cudaEventRecord(start);

    sumReduce<<<gridSize, blockSize>>>(d_a, d_b, N);
    sumReduce<<<1, blockSize>>>(d_b, d_b, N); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_b.data(), d_b, N * sizeof(int), cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout<< N <<"\n";
    std::cout<<h_b[0]<<"\n";
    std::cout << milliseconds << "\n";
    for(int i=0;i<1024;i++)
    {
        std::cout<<h_b[i]<<"\t";
    }
    std::cout<<"\n";
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}

