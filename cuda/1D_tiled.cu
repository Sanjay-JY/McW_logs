#include <chrono>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
const int I_WIDTH = 8;
const int O_WIDTH = 4; // 1024-r

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
    std::cout<<result[i];
    std::cout<<i<<"\n";
    assert(temp == result[i]);
  }

}

int main() {

  int n = 1024;
  std::cout<<n<<"\n";
  int bytes_n = n * sizeof(int);
  int m = 5;
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
  int GRID = (n-1)/THREADS+1;

  convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

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


// #include<iostream>
// #include<algorithm>
// #include<vector>
// #include<cassert>
// const int outwidth = 3;
// // GPU kernel
// __global__ void temp(int * input, int *output, int * mask, int width)
// {
//     int tx = threadIdx.x;
//     int index_o = blockIdx.x * outwidth + threadIdx.x;
//     int index_i = index_o -1;
//     int temp1 = 0;
//     __shared__ int input_s[5];
//     if ((index_i) >= 0 && (index_i) < width)
//     {
//         input_s[tx] = input[index_i];
//     }
//     else
//     {
//         input_s[tx] = 0;
//     }
//     __syncthreads();
//     if (tx < outwidth)
//     {
//         // temp1 = 0;
//         for (int j = 0; j < 5; j++)
//             temp1 += mask[j] * input_s[j + tx];
//         output[index_o] = temp1;
//     }
// }
// // Serial code for verification
// void serial_temp(const std::vector<int>& input, const std::vector<int>& mask, std::vector<int>& output, int width)
// {
//     for (int i = 0; i < width; i++)
//     {
//         int temp = 0;
//         for (int j = 0; j < 5; j++)
//         {
//             int index = i - (5 / 2) + j;
//             if (index >= 0 && index < width)
//             {
//                 temp += mask[j] * input[index];
//             }
//         }
//         output[i] = temp;
//     }
// }
// int main()
// {
//     int width = 5;
//     size_t bytes = sizeof(int) * width;
//     // Host data
//     std::vector<int> input;
//     input.reserve(width);
//     std::vector<int> mask;
//     mask.reserve(3);
//     std::vector<int> output;
//     output.reserve(width);
//     // Generate random input and mask
//     for (int i = 0; i < width; i++)
//     {
//         input.push_back(1);
//     }
//     for (int i = 0; i < 3; i++)
//     {
//         mask.push_back(1);
//     }
//     // Allocate device memory
//     int* d_i, *d_m, *d_o;
//     cudaMalloc(&d_i, bytes);
//     cudaMalloc(&d_m, 3 * sizeof(int));
//     cudaMalloc(&d_o, bytes);
//     // Copy data to device
//     cudaMemcpy(d_i, input.data(), bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_m, mask.data(), 3 * sizeof(int), cudaMemcpyHostToDevice);
//     // Launch GPU kernel
//     dim3 dimBlock(5,1,1);
//     dim3 dimGrid((((width-1)/3)+1),1,1);
//     // convolution_1D_tiled<<<dimGrid,dimBlock>>>(device_input,device_Mask,device_output);
//     // int threads = 256;
//     // int blocks = (width + threads - 1) / threads;
//     temp<<<dimGrid,dimBlock>>>(d_i, d_o, d_m, width);
// cudaError_t cudaError = cudaGetLastError();
//     if (cudaError != cudaSuccess) {
//         std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaError) << std::endl;
//     }
//     // Copy result back to host
//     cudaMemcpy(output.data(), d_o, bytes, cudaMemcpyDeviceToHost);
//     // Verify GPU result with serial code and print first 5 elements
//     std::vector<int> serial_output(width);
//     serial_temp(input, mask, serial_output, width);
//     for (int i = 0; i < width; i++)
//     {
//         std::cout << "Element " << i << " - GPU: " << output[i] << " | Serial: " << serial_output[i] << std::endl;
//     }
//     std::cout << "Verification completed." << std::endl;
//     // Free device memory
//     cudaFree(d_i);
//     cudaFree(d_m);
//     cudaFree(d_o);
//     return 0;
// }



// #include<stdio.h>
// #include<cuda.h>
// #include<cuda_runtime_api.h>
// #include<stdlib.h>
// #include<vector>
// #define O_Tile_Width 3
// #define Mask_width 3
// #define width 5
// #define Block_width (O_Tile_Width+(Mask_width-1))
// #define Mask_radius (Mask_width/2)
// __global__ void convolution_1D_tiled(float *N,float *M,float *P)
// {
// int index_out_x=blockIdx.x*O_Tile_Width+threadIdx.x;
// int index_in_x=index_out_x-Mask_radius;
// __shared__ float N_shared[Block_width];
// float Pvalue=0.0;
// //Load Data into shared Memory (into TILE)
// if((index_in_x>=0)&&(index_in_x<width))
// {
//  N_shared[threadIdx.x]=N[index_in_x];
// }
// else
// {
//  N_shared[threadIdx.x]=0.0f;
// }
// __syncthreads();
// //Calculate Convolution (Multiply TILE and Mask Arrays)
// if(threadIdx.x<O_Tile_Width)
// {
//  //Pvalue=0.0f;
//  for(int j=0;j<Mask_width;j++)
//  {
//   Pvalue+=M[j]*N_shared[j+threadIdx.x];
//  }
//  P[index_out_x]=Pvalue;
// }
// }
// void serial_temp(float* input, float* mask, std::vector<float>& output)
// {
//     for (int i = 0; i < width; i++)
//     {
//         int temp = 0;
//         for (int j = 0; j < Mask_width; j++)  // Change here to use Mask_width instead of 5
//         {
//             int index = i - Mask_radius + j;  // Change here to use Mask_radius instead of (5 / 2)
//             if (index >= 0 && index < width)
//             {
//                 temp += mask[j] * input[index];
//             }
//         }
//         output[i] = temp;
//     }
// }

// int main()
// {
//  float * input;
//  float * Mask;
//  float * output;
//  float * device_input;
//  float * device_Mask;
//  float * device_output;
//  input=(float *)malloc(sizeof(float)*width);
//  Mask=(float *)malloc(sizeof(float)*Mask_width);
//  output=(float *)malloc(sizeof(float)*width);
//  for(int i=0;i<width;i++)
//  {
//   input[i]=1.0;
//  }
//  for(int i=0;i<Mask_width;i++)
//  {
//   Mask[i]=1.0;
//  }
//   printf("\nInput:\n");
//   for(int i=0;i<width;i++)
//   {
//    printf(" %0.2f\t",*(input+i));
//   }
//   printf("\nMask:\n");
//    for(int i=0;i<Mask_width;i++)
//    {
//     printf(" %0.2f\t",*(Mask+i));
//    }
//  cudaMalloc((void **)&device_input,sizeof(float)*width);
//  cudaMalloc((void **)&device_Mask,sizeof(float)*Mask_width);
//  cudaMalloc((void **)&device_output,sizeof(float)*width);
//  cudaMemcpy(device_input,input,sizeof(float)*width,cudaMemcpyHostToDevice);
//  cudaMemcpy(device_Mask,Mask,sizeof(float)*Mask_width,cudaMemcpyHostToDevice);
//  dim3 dimBlock(Block_width,1,1);
//  dim3 dimGrid((((width-1)/O_Tile_Width)+1),1,1);
//  convolution_1D_tiled<<<dimGrid,dimBlock>>>(device_input,device_Mask,device_output);
//  cudaMemcpy(output,device_output,sizeof(float)*width,cudaMemcpyDeviceToHost);
//  std::vector<float> serial_output(width);
//  serial_temp(input, Mask, serial_output);
//  printf("\nOutput:\n");
//  for(int i=0;i<width;i++)
//  {
//   printf(" %0.2f %0.2f\t",*(output+i),serial_output[i]);
//  }
//  cudaFree(device_input);
//  cudaFree(device_Mask);
//  cudaFree(device_output);
//  free(input);
//  free(Mask);
//  free(output);
// printf("\n\nNumber of Blocks: %d ",dimGrid.x);
// printf("\n\nNumber of Threads Per Block: %d ",dimBlock.x);
// return 0;
// }