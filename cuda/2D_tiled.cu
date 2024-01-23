#include <chrono>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
const int I_WIDTH = 32;
const int O_WIDTH = 30; 

__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m) {

  int col=blockIdx.x*O_WIDTH+threadIdx.x;
  int row=blockIdx.y*O_WIDTH+threadIdx.y;

  int r = m / 2;
  int col_i = col - r;
  int row_i = row - r;

  __shared__ int s_array[I_WIDTH][I_WIDTH];
  int temp = 0;

  if((row_i>=0)&&row_i<n&&col_i>=0&&col_i<n)
  {
    s_array[threadIdx.y][threadIdx.x]=array[row_i*n+col_i];             
  }
  else
  {
    s_array[threadIdx.y][threadIdx.x]=0;
  }
  __syncthreads();


  if(threadIdx.x<O_WIDTH&&threadIdx.y<O_WIDTH&&row<n&&col<n)
  {
    temp=0;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<m;j++)
        {
            temp+=mask[i*m+j]*s_array[i+threadIdx.y][j+threadIdx.x];
        }
    }
    result[row*n+col]=temp;
    __syncthreads();
  }
}

void verify_result(int *array, int *mask, int *result, int n, int m) {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();
  
  int temp;
  int r = m/2;
  int offset_r;
  int offset_c;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      temp = 0;
      for (int k = 0; k < m; k++) {
        offset_r = i - r + k;
        for (int l = 0; l < m; l++) {
          offset_c = j - r + l;
          if (offset_r >= 0 && offset_r < n) {
            if (offset_c >= 0 && offset_c < n) {
              temp += array[offset_r*n + offset_c] * mask[k * m + l];
            }
          }
        }
      }
      assert(result[i * n + j] == temp);
    }
  }
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "CPU : "<< ms_double.count()<<"\n";

}

int main() {

  int n = 512; 
  std::cout<<n<<"\n";
  int bytes_n = (n*n) * sizeof(int);
  int m = 3;
  int bytes_m = (m*m) * sizeof(int);

  
  std::vector<int> h_array(n*n);
  for(int i=0;i<n*n;i++)
  {
    h_array[i]=i;
  }

  std::vector<int> h_mask(m*m);
  for(int i=0;i<m*m;i++)
  {
    h_mask[i]=i;
  }
  
  std::vector<int> h_result(n*n);
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
     
  dim3 threads(I_WIDTH,I_WIDTH,1);
  dim3 grid((n/O_WIDTH)+1,(n/O_WIDTH)+1,1);
  convolution_1d<<<grid, threads>>>(d_array, d_mask, d_result, n, m);

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
