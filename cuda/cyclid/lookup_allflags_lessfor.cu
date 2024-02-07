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

__global__ void lookuptable_kernel(float2 *in1, float2 *in2, int2 *d_lookup, float2 *d_xxresult, float2 *d_yyresult, float2 *d_xyresult, float2 *d_yxresult) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if the thread index is within the valid range (less than or equal to 16640)
    if (rowIdx < 16640) {
        float2 sumxx,sumyy,sumxy,sumyx;
        sumxx.x = 0.0f;
        sumxx.y = 0.0f;
        sumyy.x = 0.0f;
        sumyy.y = 0.0f;
        sumxy.x = 0.0f;
        sumxy.y = 0.0f;
        sumyx.x = 0.0f;
        sumyx.y = 0.0f;
        for (int colIdx = 1; colIdx < 1003; colIdx++) {
            int2 current = d_lookup[rowIdx * 1003 + colIdx];

            if(current.x==-1) break;

            int current_x=current.x;
            int current_y=current.y;

            float in1y_x=in1[current_y].x;
            float in1x_x=in1[current_x].x;
            float in1y_y=in1[current_y].y;
            float in1x_y=in1[current_x].y;

            float in2y_x=in2[current_y].x;
            float in2x_x=in2[current_x].x;
            float in2y_y=in2[current_y].y;
            float in2x_y=in2[current_x].y;

            float2 product;

            //XX Corelation
            product.x = (in1y_x * in1x_x) - (in1y_y *-1.0* in1x_y);
            product.y = (in1y_x * -1.0*in1x_y) + (in1y_y * in1x_x);
            product.y = -1.0*product.y;
            sumxx.x += product.x;
            sumxx.y += product.y;

            //YY Corelation
            product.x = (in2y_x * in2x_x) - (in2y_y *-1.0* in2x_y);
            product.y = (in2y_x * -1.0*in2x_y) + (in2y_y * in2x_x);
            product.y = -1.0*product.y;
            sumyy.x += product.x;
            sumyy.y += product.y;

            //XY Corelation
            product.x = (in1y_x * in2x_x) - (in1y_y *-1.0* in2x_y);
            product.y = (in1y_x * -1.0*in2x_y) + (in1y_y * in2x_x);
            product.y = -1.0*product.y;
            sumxy.x += product.x;
            sumxy.y += product.y;

            //YX Corelation
            product.x = (in2y_x * in1x_x) - (in2y_y *-1.0* in1x_y);
            product.y = (in2y_x * -1.0*in1x_y) + (in2y_y * in1x_x);
            product.y = -1.0*product.y;
            sumyx.x += product.x;
            sumyx.y += product.y;
        }

        d_xxresult[rowIdx].x = sumxx.x;
        d_xxresult[rowIdx].y = sumxx.y;
        d_yyresult[rowIdx].x = sumyy.x;
        d_yyresult[rowIdx].y = sumyy.y;
        d_xyresult[rowIdx].x = sumxy.x;
        d_xyresult[rowIdx].y = sumxy.y;
        d_yxresult[rowIdx].x = sumyx.x;
        d_yxresult[rowIdx].y = sumyx.y;

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
    
    // compute expected results
    int phaseBinIdx, phaseBin, expIdx;
    float2 tmp, in1, in2;
    float2 *exp;
    size_t profileSize = nPhaseBins * nchan * nlag;
    exp = (float2 *)malloc(profileSize * sizeof(float2));
    memset(exp, 0, profileSize*sizeof(float2));

    //Reference Code
    for (int i = 0; i<inSize2; i++) {
        in1 = in[i];
        for (int ilag=0; ilag<nlag; ilag++) {
            in2 = in[i + ilag];
            complexConjMult(in1, in2, &tmp);
            phaseBinIdx = (2*i)+ilag;
            phaseBin = phaseBins[phaseBinIdx];
            expIdx = (phaseBin * nlag * nchan) + (nlag * ichan) + ilag;
            exp[expIdx].x += tmp.x;
            exp[expIdx].y += tmp.y;
        }
    }

    //Lookup Table Creation
    int2 *lookuptable;
    lookuptable= (int2 *)malloc(16640*1003* sizeof(int2));
    fflush(stdout);
    for(int c=0;c<16640*1003;c++)
    {
        lookuptable[c].x=-1;
    }

    //clock_t start_time = clock();
    for (int i = 0; i<inSize2; i++) {
        for (int ilag=0; ilag<nlag; ilag++) {
            phaseBinIdx = (2*i)+ilag;
            phaseBin = phaseBins[phaseBinIdx];
            expIdx = (phaseBin * nlag * nchan) + (nlag * ichan) + ilag;
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
        }
    }
    // clock_t end_time = clock();
    // double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    // printf("Elapsed time: %.6f seconds\n", elapsed_time);
    
    int2 *d_lookup;
    float2 *d_xxresult,*d_yyresult,*d_xyresult,*d_yxresult,*resultxx,*resultyy,*resultxy,*resultyx;
    resultxx= (float2 *)malloc(16640*sizeof(struct float2));
    resultyy= (float2 *)malloc(16640*sizeof(struct float2));
    resultxy= (float2 *)malloc(16640*sizeof(struct float2));
    resultyx= (float2 *)malloc(16640*sizeof(struct float2));
    for(int i=0;i<16640;i++)
    {
        resultxx[i].x=0.0f;
        resultxx[i].y=0.0f;
        resultyy[i].x=0.0f;
        resultyy[i].y=0.0f;
        resultxy[i].x=0.0f;
        resultxy[i].y=0.0f;
        resultyx[i].x=0.0f;
        resultyx[i].y=0.0f;
    }

    cudaMalloc((void**)&d_lookup,16640*1003 * sizeof(int2));
    cudaMalloc((void**)&d_xxresult, 16640* sizeof(float2));
    cudaMalloc((void**)&d_yyresult, 16640* sizeof(float2));
    cudaMalloc((void**)&d_xyresult, 16640* sizeof(float2));
    cudaMalloc((void**)&d_yxresult, 16640* sizeof(float2));
    
    cudaMemcpy(d_lookup,lookuptable, 16640*1003 * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xxresult,resultxx, 16640* sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yyresult,resultyy, 16640* sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xyresult,resultxy, 16640* sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yxresult,resultyx, 16640* sizeof(float2), cudaMemcpyHostToDevice);

    float2 *in_gpu, *iny_gpu;
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

    lookuptable_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(in_gpu,iny_gpu,d_lookup, d_xxresult,d_yyresult,d_xyresult,d_yxresult);

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

    cudaMemcpy(resultxx, d_xxresult, 16640 * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultyy, d_yyresult, 16640 * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultxy, d_xyresult, 16640 * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultyx, d_yxresult, 16640 * sizeof(float2), cudaMemcpyDeviceToHost);
    
    cudaFree(d_lookup);
    cudaFree(d_xxresult);
    cudaFree(d_yyresult);
    cudaFree(d_xyresult);
    cudaFree(d_yxresult);
    cudaFree(in_gpu);
    cudaFree(iny_gpu);

    for(int i=0;i<16640;i++)
    {
        assert(exp[i].x==resultxx[i].x);
        assert(exp[i].y==resultxx[i].y);
        assert(exp[i].x==resultyy[i].x);
        assert(exp[i].y==resultyy[i].y);
        assert(exp[i].x==resultxy[i].x);
        assert(exp[i].y==resultxy[i].y);
        assert(exp[i].x==resultyx[i].x);
        assert(exp[i].y==resultyx[i].y);
    }
    printf("test_cyclid_corr_accum passed\n");


    free(in);
    free(iny);
    free(exp);
    free(lookuptable);
    free(resultxx);
    free(resultyy);
    free(resultxy);
    free(resultyx);
    
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
    printf("Lookup for all polarisation\n");
    fflush(stdout);
    test_cyclid_corr_accum1();
    //test_cyclid_corr_accum2();
}


