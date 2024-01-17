#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

const int N = 1000000;

// CUDA kernel to add two vectors
__global__ void addVectors(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Host vectors
    std::vector<int> h_a(N, 1);
    std::vector<int> h_b(N, 2);
    std::vector<int> h_c(N);

    // Device vectors
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Record the start event
    cudaEventRecord(start);

    // Launch the CUDA kernel
    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Record the stop event
    cudaEventRecord(stop);

    // Synchronize to make sure the kernel is done
    cudaEventSynchronize(stop);

    // Copy the result back to the host
    cudaMemcpy(h_c.data(), d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << "\n";

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
