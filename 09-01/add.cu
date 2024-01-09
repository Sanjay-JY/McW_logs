#include <iostream>
#include <chrono>
#define N 999

__global__ void add_vectors(double *a, double *b, double *c)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
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
	double *C = (double*)malloc(bytes);

	double *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
	}

    auto t1 = high_resolution_clock::now();

	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	int thr_per_blk = 256;
	int blk_in_grid = ceil(float(N) / thr_per_blk);


	add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);


	cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "CUDA: "<<ms_double.count() << "ms\n";



    t1 = high_resolution_clock::now();
    for(int i=0; i<N; i++)
	{
		C[i] = A[i]+B[i];
	}
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;
    std::cout << "C++: "<<ms_double.count() << "ms\n";

    for(int i=0,j=0;i<N;i++)
    {
        if(j<9){
            std::cout<<C[i]<<"\t";
            j++;
        }
        else{
            std::cout<<C[i]<<"\n";
            j=0;
        }
            
    }
    
}