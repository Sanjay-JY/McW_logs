/****************************************************************************************
A modified code which uses shared memory to store the accumulating sum and then finally 
writing it into the global memory. It uses small input and computes only one polaristion.

Link: https://bitbucket.org/assessmentmcw/cyclid/src/master/shared_modular.cu

GPU Time: 0.123388ms
*****************************************************************************************/

#include <stdio.h>
#include <complex>
#include <assert.h>
#include<cuda.h>

struct cycfold_struct {
    unsigned ncyc;
    unsigned nlag;
    unsigned nchanPfb; 
    size_t numPhaseBins;
    unsigned numTimeSamplesHfft;
    unsigned nBlocks;
};


int GPU_BLOCK_SIZE = 256*4;

const int NUM_BINS=96;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


__global__ void cyclid_corr_accum_nlag_fast(float2 *in1, float2* in2, size_t size1, size_t size2, float2* out, int nlag, unsigned *phaseBins, int pfbChan, int numPfbChans, int iblock, int numPhaseBins, bool verbose) {
    // index into the data (in2)
    int inIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // index into the lagged data partially (in1)
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // what phase bin does this result go into?
    unsigned phaseBin; 
    int lookupBlockLen = (size2*2) + nlag - 2;
    __shared__ float2 mem[NUM_BINS];
    for(int i=0;i<NUM_BINS;i++)
    {
        mem[i].x=0;
        mem[i].y=0;
    }
    __syncthreads();
    int phaseBinIdx = (iblock * lookupBlockLen) + (2*inIdx) + yIdx;
    //int phaseBinIdx = (iblock * lookupBlockLen) + (2*threadIdx.x) + threadIdx.y;
    phaseBin = phaseBins[phaseBinIdx];
    // and where in our output does the result go?
    int outIdx = (phaseBin * nlag * numPfbChans) + (nlag * pfbChan) + yIdx;
    //int outIdx = (phaseBin * nlag * numPfbChans) + (nlag * pfbChan) + threadIdx.y;
    size_t outSz = numPhaseBins * numPfbChans * nlag;
    float2 tmp;
    if ((inIdx<size2) && (outIdx<outSz) && (yIdx<nlag)) {
        // shift the in1 element, and mutliply by in2 conjugate element
        int j = inIdx+yIdx;

        tmp.x = (in1[j].x * in2[inIdx].x) - (in1[j].y * -1.0 * in2[inIdx].y);
        tmp.y = (in1[j].x * -1.0 * in2[inIdx].y) + (in1[j].y * in2[inIdx].x);
        // we will want to take the C2R FFT of the conjugate of this  
        tmp.y = -1.0 * tmp.y;

        atomicAdd(&mem[threadIdx.x*blockDim.x+threadIdx.y].x, tmp.x);
        atomicAdd(&mem[threadIdx.x*blockDim.x+threadIdx.y].y, tmp.y);
    }
    __syncthreads();

    if((inIdx<size2) && (outIdx<outSz) && (yIdx<nlag)){
        atomicAdd(&out[outIdx].x, mem[threadIdx.x*blockDim.x+threadIdx.y].x);
        atomicAdd(&out[outIdx].y, mem[threadIdx.x*blockDim.x+threadIdx.y].y);
    } 

}


void complexConjMult(float2 in1, float2 in2, float2 *tmp) {
    tmp->x = (in2.x * in1.x) - (in2.y * -1.0 * in1.y);
    tmp->y = (in2.x * -1.0 * in1.y) + (in2.y * in1.x);
    tmp->y = -1.0 * tmp->y;
}


void reference_code(float2 *in, float2 *exp, int inSize2, int nlag, int nchan, int ichan, int nPhaseBins,unsigned *phaseBins, bool verbose)
{
    int phaseBinIdx, phaseBin, expIdx;
    float2 tmp, in1, in2;
    for (int i = 0; i<inSize2; i++) {
        if (verbose)
            printf("\n%d\n", i);
        in1 = in[i];
        for (int ilag=0; ilag<nlag; ilag++) {
            if (verbose)
                printf(" ilag=%d ", ilag);
            in2 = in[i + ilag];
            complexConjMult(in1, in2, &tmp);

            // now accumulate in the right phase bin
            phaseBinIdx = (2*i)+ilag;
            phaseBin = phaseBins[phaseBinIdx];
            expIdx = (phaseBin * nlag * nchan) + (nlag * ichan) + ilag; 
            if (verbose) {
                printf(" pb=%d ",phaseBin);
                printf(" pi=%d ",phaseBinIdx);
                printf(" ei=%d ", expIdx);
            }
            // accumulate (fold)
            exp[expIdx].x += tmp.x;
            exp[expIdx].y += tmp.y;
        }
    
    }

    if (verbose)
        printf("\ncomputed expected results:\n");
    for (int iphase=0; iphase<nPhaseBins; iphase++) {
        if (verbose)
            printf("phase %d\n", iphase);
        for ( ichan=0; ichan<nchan; ichan++) {
            if (verbose)
                printf("chan %d\n", ichan);
            for (int ilag=0; ilag<nlag; ilag++ ) {
                expIdx = (iphase * nlag * nchan) + (nlag * ichan) + ilag; 
                if (verbose)
                    printf(" %f+%fi ", exp[expIdx].x, exp[expIdx].y);
            }
            if (verbose)
                printf("\n");
        }
    }
}


void call_fast_kernel(float2* out,int inSize,int profileSize,int phaseBinLookupSize,int inSize2,int nlag,float2 *in,float2 *iny,unsigned* phaseBins,int nPhaseBins,int nchan,int iblock,int ichan,bool maxOccupancy,bool time,bool verbose)
{
    printf("\n\nFAST KERNEL\n\n");
    float2 *in_gpu, *iny_gpu, *out_gpu, *outyy_gpu, *outxy_gpu, *outyx_gpu; 
    unsigned *phaseBins_gpu;
    cudaMalloc((float2 **)&in_gpu, inSize*sizeof(float2));
    cudaMalloc((float2 **)&iny_gpu, inSize*sizeof(float2));
    cudaMalloc((float2 **)&out_gpu, profileSize*sizeof(float2));
    cudaMalloc((float2 **)&outyy_gpu, profileSize*sizeof(float2));
    cudaMalloc((float2 **)&outxy_gpu, profileSize*sizeof(float2));
    cudaMalloc((float2 **)&outyx_gpu, profileSize*sizeof(float2));
    cudaMalloc((unsigned **)&phaseBins_gpu, phaseBinLookupSize*sizeof(unsigned));
    cudaMemset(out_gpu, 0, profileSize*sizeof(float2));
    cudaMemset(outyy_gpu, 0, profileSize*sizeof(float2));
    cudaMemset(outxy_gpu, 0, profileSize*sizeof(float2));
    cudaMemset(outyx_gpu, 0, profileSize*sizeof(float2));

    cudaMemcpy(in_gpu, in, inSize*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(iny_gpu, iny, inSize*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(phaseBins_gpu, phaseBins, phaseBinLookupSize*sizeof(unsigned), cudaMemcpyHostToDevice);

    int gridX, gridY, threadX, threadY;
    if (inSize <  128) { 
        gridX = 1;
        gridY = 1;
        threadX = inSize2;
        threadY = nlag;
    } else {
        if (maxOccupancy) {
            int gpuGridSize = ((inSize2 + 256) / 256);
            gridX = gpuGridSize;
            gridY = (nlag+4)/4;
            threadX = 256; ///thisGpuBlockSize; //GPU_BLOCK_SIZE / nlag;
            threadY = 4; //nlag;
        } else {
            threadX = GPU_BLOCK_SIZE/nlag;
            threadY = nlag;
            gridX = (inSize2 + threadX) / threadX;
            gridY = 1;
        }

    }

    int numKernelCalls = gridX * gridY * threadX * threadY;

    if (verbose) {
        printf("inSize2=%d nlag=%d\n", inSize2, nlag);
        printf("grid x=%d, y=%d\n", gridX, gridY);
        printf("thread x=%d, y=%d\n", threadX, threadY);
        printf("num kernel calls: %d\n", numKernelCalls);
        printf("num needed: %d\n", inSize2 * nlag);
        float diffPct = ((numKernelCalls - (inSize2*nlag))/numKernelCalls)*100.0;
        printf("num null threads: %d, %f percent\n", numKernelCalls - (inSize2*nlag), diffPct);
    }

    dim3 grids(gridX, gridY, 1);
    dim3 threads(threadX, threadY, 1);
    ichan=0;

    assert(threadX * threadY <= GPU_BLOCK_SIZE);

    cudaEvent_t startEvent, stopEvent;
    float ms;
    if (time) {
        checkCuda( cudaEventCreate(&startEvent) );
        checkCuda( cudaEventCreate(&stopEvent) );  
        checkCuda( cudaEventRecord(startEvent, 0) );
    }

    cyclid_corr_accum_nlag_fast<<<grids, threads>>>(in_gpu, in_gpu, inSize,  inSize2, out_gpu, nlag, phaseBins_gpu, ichan, nchan, iblock, nPhaseBins, verbose);
    
    if (time) {
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) ); 
        printf("%f\n", ms);
    }

    cudaMemcpy(out, out_gpu, profileSize*sizeof(float2), cudaMemcpyDeviceToHost);
    
    cudaFree(in_gpu);
    cudaFree(iny_gpu);
    cudaFree(out_gpu);
    cudaFree(outyy_gpu);
    cudaFree(outxy_gpu);
    cudaFree(outyx_gpu);
    cudaFree(phaseBins_gpu);

}

int validate_results(int nPhaseBins,int nchan,int nlag,float2 *out,float2 *exp,bool verbose){
    int expIdx;
    if (verbose)
        printf("\nresults:\n");
    for (int iphase=0; iphase<nPhaseBins; iphase++) {
        if (verbose)
            printf("phase %d\n", iphase);
        for (int ichan=0; ichan<nchan; ichan++) {
            if (verbose)
                printf("chan %d\n", ichan);
            for (int ilag=0; ilag<nlag; ilag++ ) {
                expIdx = (iphase * nlag * nchan) + (nlag * ichan) + ilag; 
                if (verbose)
                    printf(" %f+%fi ", out[expIdx].x, out[expIdx].y);
                float diffx = abs(out[expIdx].x - exp[expIdx].x);    
                float diffy = abs(out[expIdx].y - exp[expIdx].y);    
                float tol = 1e2;
                if ((diffx > tol) || (diffy > tol)) {
                    printf("out[%d]=%f + %fi != exp[%d]=%f + %fi\n", expIdx, out[expIdx].x, out[expIdx].y, expIdx, exp[expIdx].x, exp[expIdx].y);
                    printf("diff x=%f y=%f\n", diffx, diffy);
                    return 1;
                }
            }
            if (verbose)
                printf("\n");
        }
    }
    printf("test_cyclid_corr_accum passed\n");
    return 1;
}
        

int main() {
    printf("Sample_data_cyclid_gpu\n");

    cycfold_struct cs;

    cs.ncyc = 4;
    cs.nlag = (cs.ncyc/2) + 1; 
    cs.numTimeSamplesHfft = 16;
    cs.nBlocks = 1;
    cs.nchanPfb = 1;
    cs.numPhaseBins = 4;

    int phaseBinLookupSize = (2*cs.numTimeSamplesHfft) + cs.nlag - 2;
    unsigned *phaseBins;
    phaseBins = (unsigned *)malloc(phaseBinLookupSize *sizeof(unsigned));
    memset(phaseBins, 0, phaseBinLookupSize*sizeof(int));

    bool time = true;
    bool maxOccupancy = false;
    bool verbose = false;

    int nlag = cs.nlag; 
    size_t inSize = cs.numTimeSamplesHfft;
    size_t inSize2 = inSize - nlag - 1;
    phaseBinLookupSize = (2*inSize) + nlag - 2;
    int nPhaseBins = cs.numPhaseBins;
    int nchan = cs.nchanPfb;
    int ichan = 0;
    int iblock = 0;
    
    float2 *in, *iny;
    in = (float2 *)malloc(inSize * sizeof(float2));
    iny = (float2 *)malloc(inSize * sizeof(float2));
    memset(in, 0, inSize*sizeof(float2));
    memset(iny, 0, inSize*sizeof(float2));
    int maxValue = 127; 
    int value = 0;
    float fvalue = 0.5;
    float imgDiv = 2.0;

    for (int i = 0; i<inSize; i++) {
        in[i].x = ((float)value) + fvalue;
        in[i].y = ( (float)((float)value)/imgDiv) + fvalue;
        iny[i].x = in[i].x;
        iny[i].y = in[i].y;
        value++;
        if (value + fvalue>maxValue)
            value=0;
    }
        
    if (verbose) {
        printf("inSize=%ld, inSize2=%ld\n", inSize, inSize2);
        printf("phaseBinLookupSize=%d\n", phaseBinLookupSize);
        for (int i = 0; i<inSize; i++) 
            printf("in[%d]=%f+%fi\n", i, in[i].x, in[i].y);
    }


    float2 *exp;
    size_t profileSize = nPhaseBins * nchan * nlag;
    exp = (float2 *)malloc(profileSize * sizeof(float2));
    memset(exp, 0, profileSize*sizeof(float2));

    reference_code(in,exp,inSize2,nlag,nchan,ichan,nPhaseBins,phaseBins,verbose);

    float2 *out;
    out = (float2 *)malloc(profileSize*sizeof(float2));

    call_fast_kernel(out,inSize,profileSize,phaseBinLookupSize,inSize2,nlag,in,iny,phaseBins,nPhaseBins,nchan,iblock,ichan,maxOccupancy,time,verbose);

    validate_results(nPhaseBins,nchan,nlag,out,exp,verbose);

    free(phaseBins);
    free(in);
    free(iny);
    free(exp);
    free(out);

}