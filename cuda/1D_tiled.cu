#include <chrono>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
const int I_WIDTH = 1024;
const int O_WIDTH = 1018; // 1024-2*r

__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m) {

  int tid = blockIdx.x * O_WIDTH + threadIdx.x;
  int r = m / 2;
  int start = tid - r;
  __shared__ int s_array[I_WIDTH];
  int temp = 0;
  if((start>=0)&&start<n)
  {
    s_array[threadIdx.x]=array[start];
  }
  else
  {
    s_array[threadIdx.x]=0;
  }
  __syncthreads();
  if(threadIdx.x<O_WIDTH)
  {
    temp=0;
    for(int i=0;i<m;i++)
    {
        temp+=mask[i]*s_array[i+threadIdx.x];
    }
    result[tid]=temp;
    __syncthreads();
  }
}

void verify_result(int *array, int *mask, int *result, int n, int m) {

  int radius = m / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < m; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]);
  }

}

int main() {

  int n = 1<<18;   //1<<19 doesn't work
  std::cout<<n<<"\n";
  int bytes_n = n * sizeof(int);
  int m = 7;
  int bytes_m = m * sizeof(int);

  
  std::vector<int> h_array(n);
  for(int i=0;i<n;i++)
  {
    h_array[i]=i;
  }

  std::vector<int> h_mask(m);
  for(int i=0;i<m;i++)
  {
    h_mask[i]=i;
  }
  
  std::vector<int> h_result(n);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int *d_array, *d_mask, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_mask, bytes_m);
  cudaMalloc(&d_result, bytes_n);

  cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

  int THREADS = I_WIDTH;
  int GRID = n/O_WIDTH+1;     

  convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaError) << std::endl;
  }

  cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "GPU : "<<milliseconds << "\n";

  verify_result(h_array.data(), h_mask.data(), h_result.data(), n, m);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  cudaFree(d_result);
  cudaFree(d_mask);
  cudaFree(d_array);

  return 0;
}


