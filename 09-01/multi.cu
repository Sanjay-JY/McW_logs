#include <iostream>
#include <chrono>
#define N 32

__global__ void multiply_vectors(int size,double *a, double *b)
{
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if((Row<size)&&(Col<size))
    {
        double sum=0;
        for(int i=0;i<size;i++)
        {
            sum=sum+a[Row*size+i]*b[Col+i*size];
        }
        b[Row*size+Col]=sum;     
    }
}


int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

	size_t bytes = N*sizeof(double);

	double *A = (double*)malloc(bytes);
	double *B = (double*)malloc(bytes);

	double *d_A, *d_B;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);

	for(int i=0; i<N; i++)
	{
		A[i] = i;
	}

    auto t1 = high_resolution_clock::now();

	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	dim3 grid((N-1)/4 + 1,(N-1)/4 + 1,1);
	dim3 blk(4,4,1);


	multiply_vectors<<< grid, blk >>>(N,d_A, d_A);  


	cudaMemcpy(B, d_B, bytes, cudaMemcpyDeviceToHost);

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "CUDA: "<<ms_double.count() << "ms\n";



    // t1 = high_resolution_clock::now();
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         double sum = 0;
    //         for (int k = 0; k < N; k++)
    //             sum += A[i*N+k] * A[k*N+j];
    //         B[i*N+j] = sum;
    //     }
    // }
    // t2 = high_resolution_clock::now();
    // ms_double = t2 - t1;
    // std::cout << "C++: "<<ms_double.count() << "ms\n";

    for(int i=0,j=0;i<N;i++)
    {
        if(j<N-1){
            std::cout<<B[i]<<"\t";
            j++;
        }
        else{
            std::cout<<B[i]<<"\n";
            j=0;
        }
            
    }
    
}