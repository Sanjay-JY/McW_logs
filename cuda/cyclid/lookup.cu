// Copyright (C) 2023 Associated Universities, Inc. Washington DC, USA.
// This stand alone module is an adaptation of a unit test of a specific kernel in the
// Cyclid package (a package for Cyclic Spectroscopy processing)
// To build using CUDA 11.6:
// nvcc -c cyclid_gpu.cu -o cyclid_gpu.o -dc
// nvcc -o cyclid_gpu cyclid_gpu.o
#include <stdio.h>
#include <complex>
#include <assert.h>
#include <time.h>
#include<cuda.h>
// part of a larger structure that plays a larger
// role in the full pipeline
struct cycfold_struct {
    unsigned ncyc;
    unsigned nlag;
    unsigned nchanPfb;
    size_t numPhaseBins;
    unsigned numTimeSamplesHfft;
    unsigned nBlocks;
};
// TBF: constant for this gpu?  Maybe query?
int GPU_BLOCK_SIZE = 256*4;
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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

__global__ void lookuptable_kernel(float2 *in1, float2 *in2, int2 *d_lookup, float2 *d_result) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if the thread index is within the valid range (less than or equal to 16640)
    if (rowIdx < 16640) {
        float2 sum;
        sum.x = 0.0f;
        sum.y = 0.0f;
        for (int colIdx = 1; colIdx < 1003; colIdx++) {
            int2 current = d_lookup[rowIdx * 1003 + colIdx];
                if(current.x==-1) break;
            float2 product;
            product.x = in1[current.y].x * in2[current.x].x - in1[current.y].y *-1.0* in2[current.x].y;
            product.y = in1[current.y].x * -1.0*in2[current.x].y + in1[current.y].y * in2[current.x].x;
              //printf("%f\t%f\t%d\n",product.x,product.y,rowIdx);
            product.y = -1.0*product.y;
            sum.x += product.x;
            sum.y += product.y;
            //printf("%d\t%d\t%d\n",threadIdx.x,current.x,current.y);
        }
        d_result[rowIdx].x = sum.x;
        d_result[rowIdx].y = sum.y;
    }
}

void complexConjMult(float2 in1, float2 in2, float2 *tmp) {
    tmp->x = (in2.x * in1.x) - (in2.y * -1.0 * in1.y);
    tmp->y = (in2.x * -1.0 * in1.y) + (in2.y * in1.x);
    tmp->y = -1.0 * tmp->y;
}

// woarker function for tests of the corr_accum kernel
int test_cyclid_corr_accum(cycfold_struct *cs, unsigned *phaseBins, bool maxOccupancy, bool time) {
    printf("test_cyclid_corr_accum\n");
        fflush(stdout);
    bool verbose = false;
    int nlag = cs->nlag; //(ncyc/2) + 1;
    size_t inSize = cs->numTimeSamplesHfft; //8;
    size_t inSize2 = inSize - nlag - 1;
    int phaseBinLookupSize = (2*inSize) + nlag - 2;
    int nPhaseBins = cs->numPhaseBins;
    int nchan = cs->nchanPfb;
    int ichan = 0;
    int iblock = 0;
    // init input
    float2 *in, *iny;
    in = (float2 *)malloc(inSize * sizeof(float2));
    iny = (float2 *)malloc(inSize * sizeof(float2));
    memset(in, 0, inSize*sizeof(float2));
    memset(iny, 0, inSize*sizeof(float2));
    int maxValue = 127; //255 causes overflow problems?;
    int value = 0;
    // setting the fractional parts to anything but 0
    // keeps these tests from failing!
    // TBF: I think this is the fact that atomicAdd is not
    // reproducible - floating point errors are not associative?
    float fvalue = 0.5;
    float imgDiv = 2.0;
    for (int i = 0; i<inSize; i++) {
        in[i].x = ((float)value) + fvalue;
        in[i].y = ( (float)((float)value)/imgDiv) + fvalue;
        // unimaginative population of the second polarization
        iny[i].x = in[i].x;
        iny[i].y = in[i].y;
        value++;
        if (value + fvalue>maxValue)
            value=0;
    }
    //printf("test_cyclid_corr_accum\n");
    fflush(stdout);
    // compute expected results
    int phaseBinIdx, phaseBin, expIdx;
    float2 tmp, in1, in2;
    float2 *exp;
    size_t profileSize = nPhaseBins * nchan * nlag;
    exp = (float2 *)malloc(profileSize * sizeof(float2));
    memset(exp, 0, profileSize*sizeof(float2));
    int2 *lookuptable;
    lookuptable= (int2 *)malloc(16640*1003* sizeof(int2));
    //printf("%d",sizeof(int2));
        fflush(stdout);
    for(int c=0;c<16640*1003;c++)
    {
        lookuptable[c].x=-1;
    }
    //printf("test_cyclid_corr_accum3\n");
    fflush(stdout);
    //clock_t start_time = clock();
    for (int i = 0; i<inSize2; i++) {
        in1 = in[i];
        for (int ilag=0; ilag<nlag; ilag++) {
            in2 = in[i + ilag];
            complexConjMult(in1, in2, &tmp);
            phaseBinIdx = (2*i)+ilag;
            phaseBin = phaseBins[phaseBinIdx];
            expIdx = (phaseBin * nlag * nchan) + (nlag * ichan) + ilag;
            //printf("exp->%d\n",expIdx);
            int index =(lookuptable[expIdx*1003].x);
            if(index==-1)
            {
                lookuptable[expIdx*1003].x=1;
                lookuptable[expIdx*1003 + (lookuptable[expIdx*1003].x)].x = i;
                lookuptable[expIdx*1003 + (lookuptable[expIdx*1003].x)].y = i+ilag;
                lookuptable[expIdx*1003].x=(lookuptable[expIdx*1003].x)+1;
            }
            else
            {
                lookuptable[expIdx*1003 +(lookuptable[expIdx*1003].x)].x=i;
                lookuptable[expIdx*1003 + (lookuptable[expIdx*1003].x)].y=i+ilag;
                lookuptable[expIdx*1003].x= (lookuptable[expIdx*1003].x)+1;
            }
            exp[expIdx].x += tmp.x;
            exp[expIdx].y += tmp.y;
            //printf("%d\t%d\t%d\n",expIdx,i,i+ilag);
            fflush(stdout);
        }
    }
    // clock_t end_time = clock();
    // double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    // printf("Elapsed time: %.6f seconds\n", elapsed_time);
    // printf("\n");
    // printf("\n");
    /*for(int i=0;i<3;i++)
    {
        for(int j=1;j<=12;j++)
        {
            int a =lookuptable[i*1003+j].x;
            int b =lookuptable[i*1003+j].y;
            printf("%d\t%d\t%d\n",i,a,b);
        }
    }*/
    //printf("test_cyclid_corr_accum4\n");
    fflush(stdout);
    int2 *d_lookup;
    float2 *d_result,*result;
    result= (float2 *)malloc(16640*sizeof(struct float2));
    for(int i=0;i<16640;i++)
    {
        result[i].x=0.0f;
        result[i].y=0.0f;
    }
    cudaMalloc((void**)&d_lookup,16640*1003 * sizeof(int2));
    cudaMalloc((void**)&d_result, 16640* sizeof(float2));
    cudaMemcpy(d_lookup,lookuptable, 16640*1003 * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result,result, 16640* sizeof(float2), cudaMemcpyHostToDevice);
    float2 *in_gpu, *iny_gpu, *out_gpu, *outyy_gpu, *outxy_gpu, *outyx_gpu;
    unsigned *phaseBins_gpu;
    cudaMalloc((float2 **)&in_gpu, inSize*sizeof(float2));
    cudaMalloc((float2 **)&iny_gpu, inSize*sizeof(float2));
    cudaMemcpy(in_gpu, in, inSize*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(iny_gpu, iny, inSize*sizeof(float2), cudaMemcpyHostToDevice);
    int NUM_THREADS =1024;
    int NUM_BLOCKS = (16640+ NUM_THREADS-1) / NUM_THREADS;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record start event
    cudaEventRecord(start);
    lookuptable_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(in_gpu,iny_gpu,d_lookup, d_result);
    cudaEventRecord(stop);
    // Synchronize to make sure all the streams have finished
    cudaEventSynchronize(stop);
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%f\n",milliseconds);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_result, 16640 * sizeof(float2), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 10; i++) {
    //     printf("%f+%fi\n", exp[i].x, exp[i].y);
    // }
    // for (int i = 0; i < 10; i++) {
    //     printf("%f+%fi\n", result[i].x, result[i].y);
    // }
    for(int i=0;i<16640;i++)
    {
        // printf("%d\t",i);
        // printf("%f\n",result[i].x);
        // printf()
        assert(exp[i].x==result[i].x);
        assert(exp[i].y==result[i].y);
    }
    printf("test_cyclid_corr_accum passed\n");
    return 0;
}


int test_cyclid_corr_accum1() {
    cycfold_struct cs;
    // realistice size data set
    cs.ncyc = 128;
    cs.nlag = (cs.ncyc/2) + 1;
    cs.numTimeSamplesHfft = 256250;
    //int inSize2 = cs.numTimeSamplesHfft - cs.nlag - 1;
    cs.nBlocks = 1;
    cs.nchanPfb = 1;
    // init phase bins: not many phase bins, all samples use 0 but a few
    cs.numPhaseBins = 256;
    int phaseBinLookupSize = (2*cs.numTimeSamplesHfft) + cs.nlag - 2;
    unsigned *phaseBins;
    phaseBins = (unsigned *)malloc(phaseBinLookupSize *sizeof(unsigned));
    memset(phaseBins, 0, phaseBinLookupSize*sizeof(int));
    // spread out the phaseBins equally
    int phaseStep = phaseBinLookupSize / cs.numPhaseBins;
    //int phaseStepRem = phaseBinLookupSize % cs.numPhaseBins;
    for (int iphase=0; iphase<cs.numPhaseBins; iphase++) {
        int start = iphase*phaseStep;
        int end = (iphase+1)*phaseStep;
        for (int j=start; j<end; j++)
            phaseBins[j] = iphase;
    }
    // double check phase counts make sense
    int phaseCnts[cs.numPhaseBins];
    memset(phaseCnts, 0, cs.numPhaseBins*sizeof(int));
    for (int i = 0; i < cs.numTimeSamplesHfft; i++)
        for (int ilag = 0; ilag < cs.nlag; ilag++)
            phaseCnts[phaseBins[(2*i) + ilag]]++;
    // add them up
    int phaseCntTotal = 0;
    for (int i=0; i<cs.numPhaseBins; i++) {
        //printf("%d\n", phaseCnts[i]);
        phaseCntTotal += phaseCnts[i];
    }
    assert(phaseCntTotal == cs.numTimeSamplesHfft * cs.nlag);
    bool time = false;
    bool maxOccupancy = false;
    int rv = test_cyclid_corr_accum(&cs, phaseBins, maxOccupancy, time);
    free(phaseBins);
    return rv;
}


int test_cyclid_corr_accum2() {
    cycfold_struct cs;
    // very small data set
    cs.ncyc = 4;
    cs.nlag = (cs.ncyc/2) + 1;
    cs.numTimeSamplesHfft = 16;
    cs.nBlocks = 1;
    cs.nchanPfb = 1;
    // init phase bins: not many phase bins, all samples use 0
    cs.numPhaseBins = 4;
    int phaseBinLookupSize = (2*cs.numTimeSamplesHfft) + cs.nlag - 2;
    unsigned *phaseBins;
    phaseBins = (unsigned *)malloc(phaseBinLookupSize *sizeof(unsigned));
    memset(phaseBins, 0, phaseBinLookupSize*sizeof(int));
    bool time = false;
    bool maxOccupancy = false; // ignored for small data sets anyways
    int rv = test_cyclid_corr_accum(&cs, phaseBins, time, maxOccupancy);
    free(phaseBins);
    return rv;
}


int main() {
    printf("Lookup nlag\n");
    fflush(stdout);
    test_cyclid_corr_accum1();
    //test_cyclid_corr_accum2();
}


