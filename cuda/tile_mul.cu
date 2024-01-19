#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>
 
const int N = 64;
const int SIZE = 64;

__global__ void matrixMul(const int *a, const int *b, int *c) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_a[SIZE];
  __shared__ int s_b[SIZE];

  int tmp = 0;

  for (int i = 0; i < N; i += blockDim.x) {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
      tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }
    __syncthreads();
  }
  c[row * N + col] = tmp;
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[i * N + k] * b[k * N + j];
      }

      assert(tmp == c[i * N + j]);
    }
  }
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "CPU : "<< ms_double.count()<<"\n";
}

int main() {

  size_t bytes = N * N * sizeof(int);

  std::vector<int> h_a((N * N),1);
  std::vector<int> h_b((N * N),2);
  std::vector<int> h_c(N * N);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int THREADS = 8;
  int BLOCKS = N / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "GPU : "<<milliseconds << "\n";

  verify_result(h_a, h_b, h_c);
  std::cout << "COMPLETED SUCCESSFULLY\n";
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}