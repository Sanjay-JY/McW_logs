// Copyright (C) 2023 Associated Universities, Inc. Washington DC, USA.

// This stand alone module is an adaptation of a unit test of a specific kernel in the
// Cyclid package (a package for Cyclic Spectroscopy processing)

// To build using CUDA 11.6:
// nvcc -c cyclid_gpu.cu -o cyclid_gpu.o -dc
// nvcc -o cyclid_gpu cyclid_gpu.o

#include <stdio.h>
#include <complex>
#include <assert.h>

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

// The kernel for combining the correlation and folding (accumulation) steps
__global__ void cyclid_corr_accum_nlag_fast(float2 *in1, float2* in2, size_t size1, size_t size2, float2* out, int nlag, unsigned *phaseBins, int pfbChan, int numPfbChans, int iblock, int numPhaseBins, bool verbose) {
    // index into the data (in2)
    int inIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // index into the lagged data partially (in1)
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    // what phase bin does this result go into?
    unsigned phaseBin; 
    int lookupBlockLen = (size2*2) + nlag - 2;
    int phaseBinIdx = (iblock * lookupBlockLen) + (2*inIdx) + yIdx;
    phaseBin = phaseBins[phaseBinIdx];
    // and where in our output does the result go?
    int outIdx = (phaseBin * nlag * numPfbChans) + (nlag * pfbChan) + yIdx;
    size_t outSz = numPhaseBins * numPfbChans * nlag;
    float2 tmp;
    if ((inIdx<size2) && (outIdx<outSz) && (yIdx<nlag)) {
        // shift the in1 element, and mutliply by in2 conjugate element
        int j = inIdx+yIdx;

        tmp.x = (in1[j].x * in2[inIdx].x) - (in1[j].y * -1.0 * in2[inIdx].y);
        tmp.y = (in1[j].x * -1.0 * in2[inIdx].y) + (in1[j].y * in2[inIdx].x);
        // we will want to take the C2R FFT of the conjugate of this  
        tmp.y = -1.0 * tmp.y;

        atomicAdd(&out[outIdx].x, tmp.x);
        atomicAdd(&out[outIdx].y, tmp.y);
    } 
}

// The kernel for combining the correlation and folding (accumulation) steps
// but for ALL polarizations
__global__ void cyclid_corr_accum_all_pols(float2 *in1, float2* in2, size_t size1, size_t size2, float2* outXX, float2* outYY, float2 *outXY, float2 *outYX, unsigned *phaseBins, int numPhaseBins, int numPfbChans, int nlag , int iblock, int pfbChan, size_t outSz, int lookupBlockLen, bool verbose) {

    // index into the data (in2)
    int inIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // index into the lagged data partially (in1)
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned phaseBin ; //= phaseBins[phaseBinIdx];
    lookupBlockLen = (size2*2) + nlag - 2;
    int phaseBinIdx = (iblock * lookupBlockLen) + (2*inIdx) + yIdx;
    phaseBin = phaseBins[phaseBinIdx];
    int outIdx = (phaseBin * nlag * numPfbChans) + (nlag * pfbChan) + yIdx;
    float2 tmp;
    if ((inIdx<size2) && (outIdx<outSz) && (yIdx<nlag)) {
        // shift the in1 element, and mutliply by in2 conjugate element
        int j = inIdx+yIdx;

        // XX correlation
        tmp.x = (in1[j].x * in1[inIdx].x) - (in1[j].y * -1.0 * in1[inIdx].y);
        tmp.y = (in1[j].x * -1.0 * in1[inIdx].y) + (in1[j].y * in1[inIdx].x);
        // we will want to take the C2R FFT of the conjugate of this  
        tmp.y = -1.0 * tmp.y;

        atomicAdd(&outXX[outIdx].x, tmp.x);
        atomicAdd(&outXX[outIdx].y, tmp.y);

        // YY correlation
        tmp.x = (in2[j].x * in2[inIdx].x) - (in2[j].y * -1.0 * in2[inIdx].y);
        tmp.y = (in2[j].x * -1.0 * in2[inIdx].y) + (in2[j].y * in2[inIdx].x);
        // we will want to take the C2R FFT of the conjugate of this  
        tmp.y = -1.0 * tmp.y;

        atomicAdd(&outYY[outIdx].x, tmp.x);
        atomicAdd(&outYY[outIdx].y, tmp.y);

        // XY correlation
        tmp.x = (in1[j].x * in2[inIdx].x) - (in1[j].y * -1.0 * in2[inIdx].y);
        tmp.y = (in1[j].x * -1.0 * in2[inIdx].y) + (in1[j].y * in2[inIdx].x);
        // we will want to take the C2R FFT of the conjugate of this  
        tmp.y = -1.0 * tmp.y;
        atomicAdd(&outXY[outIdx].x, tmp.x);
        atomicAdd(&outXY[outIdx].y, tmp.y);

        // YX correlation
        tmp.x = (in2[j].x * in1[inIdx].x) - (in2[j].y * -1.0 * in1[inIdx].y);
        tmp.y = (in2[j].x * -1.0 * in1[inIdx].y) + (in2[j].y * in1[inIdx].x);
        // we will want to take the C2R FFT of the conjugate of this  
        tmp.y = -1.0 * tmp.y;
        atomicAdd(&outYX[outIdx].x, tmp.x);
        atomicAdd(&outYX[outIdx].y, tmp.y);

    } 
}



// returns conjugate of complex multiplication
void complexConjMult(float2 in1, float2 in2, float2 *tmp) {
    tmp->x = (in2.x * in1.x) - (in2.y * -1.0 * in1.y);
    tmp->y = (in2.x * -1.0 * in1.y) + (in2.y * in1.x);
    tmp->y = -1.0 * tmp->y;
}

// worker function for tests of the corr_accum kernel
int test_cyclid_corr_accum(cycfold_struct *cs, unsigned *phaseBins, bool maxOccupancy, bool time) {

    printf("test_cyclid_corr_accum\n");

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
        
    if (verbose) {
        printf("inSize=%d, inSize2=%d\n", inSize, inSize2);
        printf("phaseBinLookupSize=%d\n", phaseBinLookupSize);
        for (int i = 0; i<inSize; i++) 
            printf("in[%d]=%f+%fi\n", i, in[i].x, in[i].y);
    }

    // compute expected results
    int phaseBinIdx, phaseBin, expIdx;
    float2 tmp, in1, in2;
    float2 *exp;
    size_t profileSize = nPhaseBins * nchan * nlag;
    exp = (float2 *)malloc(profileSize * sizeof(float2));
    memset(exp, 0, profileSize*sizeof(float2));

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
    // print exp result:
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


    // move data to GPU
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

    //CHECK_CUDA("prepared gpu output\n");

    // determine how to run the kernel
    int gridX, gridY, threadX, threadY;
    if (inSize <  128) { //thisGpuBlockSize) {
        gridX = 1;
        gridY = 1;
        threadX = inSize2;
        threadY = nlag;
    } else {
        // two styles for calling the kernel
        if (maxOccupancy) {
            // the current style wastes a LOT of threads
            int gpuGridSize = ((inSize2 + 256) / 256);
            gridX = gpuGridSize;
            gridY = (nlag+4)/4;
            threadX = 256; ///thisGpuBlockSize; //GPU_BLOCK_SIZE / nlag;
            threadY = 4; //nlag;
        } else {
            // new style wastes fewer threads BUT IS SLOWER!?
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
    //bool v = false;
    ichan=0;

    // if these were to fail siltenlty, we'd get CUDA errors later
    assert(threadX * threadY <= GPU_BLOCK_SIZE);

    // for timing
    cudaEvent_t startEvent, stopEvent;
    float ms;
    if (time) {
        // record the time
        // events for timing
        checkCuda( cudaEventCreate(&startEvent) );
        checkCuda( cudaEventCreate(&stopEvent) );  
        checkCuda( cudaEventRecord(startEvent, 0) );
    }

    // FINALLY, actually call the kernel!
    //cyclid_corr_accum_nlag_fast<<<grids, threads>>>(in_gpu, in_gpu, inSize,  inSize2, out_gpu, nlag, phaseBins_gpu, ichan, nchan, iblock, nPhaseBins, verbose);
    cyclid_corr_accum_all_pols<<<grids,threads>>>(in_gpu, iny_gpu, inSize, inSize2, out_gpu, outyy_gpu, outxy_gpu, outyx_gpu, phaseBins_gpu, nPhaseBins, nchan, nlag, iblock, ichan, profileSize, phaseBinLookupSize, false);

    //CHECK_CUDA("ran gpu kernel\n");

    if (time) {
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) ); 
        printf("%f\n", ms);
    }

    //CHECK_CUDA("ran gpu kernel timing\n");

    // move data to CPU
    float2 *out;
    out = (float2 *)malloc(profileSize*sizeof(float2));
    cudaMemcpy(out, out_gpu, profileSize*sizeof(float2), cudaMemcpyDeviceToHost);

    // check result:
    //verbose = true;
    if (verbose)
        printf("\nresults:\n");
    for (int iphase=0; iphase<nPhaseBins; iphase++) {
        if (verbose)
            printf("phase %d\n", iphase);
        for ( ichan=0; ichan<nchan; ichan++) {
            if (verbose)
                printf("chan %d\n", ichan);
            for (int ilag=0; ilag<nlag; ilag++ ) {
                expIdx = (iphase * nlag * nchan) + (nlag * ichan) + ilag; 
                if (verbose)
                    printf(" %f+%fi ", out[expIdx].x, out[expIdx].y);
                float diffx = abs(out[expIdx].x - exp[expIdx].x);    
                float diffy = abs(out[expIdx].y - exp[expIdx].y);    
                float tol = 1e2;
                // TBF: exact matches do work if fvalue above is zero
                //if ((out[expIdx].x != exp[expIdx].x) || (out[expIdx].y != exp[expIdx].y)) {
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

    return 0;

}

// A test of the corr_accum kernel that uses fake data that
// is of a realistic size, processed using a realistic number of params
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

    bool time = true;
    bool maxOccupancy = false;
    int rv = test_cyclid_corr_accum(&cs, phaseBins, maxOccupancy, time);

    free(phaseBins);

    return rv;
}

// A test of the corr_accum kernel that uses data on a small
// enough scale to allow visualization of what's going on
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

    bool time = true;
    bool maxOccupancy = false; // ignored for small data sets anyways

    int rv = test_cyclid_corr_accum(&cs, phaseBins, maxOccupancy, time);

    free(phaseBins);

    return rv;
}
int main() {
    printf("cyclid_gpu\n");
    //test_cyclid_corr_accum1();
    test_cyclid_corr_accum2();
}

