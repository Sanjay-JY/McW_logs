#include <stdio.h>
#include <cooperative_groups.h>

#define CLUSTER_SIZE 4
#define BLOCK_SIZE 32

namespace cg = cooperative_groups;
using mt=int;
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
cluster_reduce_sum(int n, mt *arr, mt *sum)
{
    __shared__ mt shared_mem[BLOCK_SIZE];
    __shared__ mt cluster_sum;

    cluster_sum = 0;

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int cluster_block_rank = cluster.block_rank();
    unsigned int cluster_size = cluster.dim_blocks().x;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;


    shared_mem[threadIdx.x] = 0;
    if (idx < n) {
        shared_mem[threadIdx.x] = arr[idx];
    }

    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset; offset /= 2) {
                if (threadIdx.x < offset) {
                        shared_mem[threadIdx.x] += shared_mem[threadIdx.x + offset];
                }
                __syncthreads();
        }

    cluster.sync();

    if (threadIdx.x == 0) {
                atomicAdd(cluster.map_shared_rank(&cluster_sum, 0), shared_mem[0]);
    }

    cluster.sync();

    if (threadIdx.x == 0 && cluster_block_rank == 0) {
                atomicAdd(sum, cluster_sum);
    }

    cluster.sync();
}

int main(int argc, char* argv[]) {
    int n = 128;

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    mt *h_arr, *h_sum, sum;
    h_arr = (mt*) malloc(n * sizeof(float));
    h_sum = (mt*) malloc(sizeof(float));

    //int upper = 1024, lower = -1024;

    sum = 0;
    for(int i = 0; i < n; i++)
    {
        h_arr[i] = 1;
        sum += h_arr[i];
    }

    mt *d_arr, *d_sum;
    cudaMalloc(&d_arr, n * sizeof(mt));
    cudaMalloc(&d_sum, sizeof(mt));

    cudaMemcpy(d_arr, h_arr, n * sizeof(mt), cudaMemcpyHostToDevice);

    //int num_clusters = ceil ((float)n / (CLUSTER_SIZE * BLOCK_SIZE)) + 1;
    int num_clusters = 1;
    cluster_reduce_sum <<< CLUSTER_SIZE * num_clusters, BLOCK_SIZE >>> (n, d_arr, d_sum);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaMemcpy(h_sum, d_sum, sizeof(mt), cudaMemcpyDeviceToHost);

    if (*h_sum != sum) {
        printf("Kernel incorrect: %f vs %f\n", (float)sum, (float)(*h_sum));
    }

    return 0;
}