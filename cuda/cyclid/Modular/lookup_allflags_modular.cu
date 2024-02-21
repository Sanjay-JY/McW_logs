/************************************************************************************
A modified code which computes lookuptable in the CPU and threads are mapped to index 
in the output array. Each thread adds the value in the lookup table and writes in the 
output array. It uses input which is similar to the real-world data and computes all 
polarisations

Link: https://bitbucket.org/assessmentmcw/cyclid/src/master/lookup_allflags_modular.cu

GPU Time: 14.5694 ms
CPU Time: 142.603 ms
***************************************************************************************/

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


int call_all_polarisation_kernel(float2 *out,int inSize,int profileSize,int phaseBinLookupSize,int inSize2,int nlag,float2 *in,float2 *iny,unsigned* phaseBins,int nPhaseBins,int nchan,int iblock,int ichan,bool maxOccupancy,bool time,bool verbose,bool validate)
{
    printf("\n\nALL POLARISATION KERNEL\n\n");

    int phaseBinIdx, phaseBin, expIdx;

    clock_t start_time = clock();

    int2 *lookuptable;
    lookuptable= (int2 *)malloc(16640*1003*sizeof(int2));
    fflush(stdout);
    for(int c=0;c<16640*1003;c++)
    {
        lookuptable[c].x=-1;
    }

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
    
    
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    //printf("%.6f\n", elapsed_time*1000);


    int2 *d_lookup;
    float2 *d_xxresult,*d_yyresult,*d_xyresult,*d_yxresult,*resultxx,*resultyy,*resultxy,*resultyx;
    
    
    resultxx= (float2 *)malloc(16640*sizeof(float2));
    resultyy= (float2 *)malloc(16640*sizeof(float2));
    resultxy= (float2 *)malloc(16640*sizeof(float2));
    resultyx= (float2 *)malloc(16640*sizeof(float2));

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
    cudaMalloc((float2 **)&in_gpu, inSize*sizeof(float2));
    cudaMalloc((float2 **)&iny_gpu, inSize*sizeof(float2));
    cudaMemcpy(in_gpu, in, inSize*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(iny_gpu, iny, inSize*sizeof(float2), cudaMemcpyHostToDevice);


    int NUM_THREADS =1024;
    int NUM_BLOCKS = (16640+ NUM_THREADS-1) / NUM_THREADS;

    cudaEvent_t startEvent, stopEvent;
    float ms;
    if (time) {
        checkCuda( cudaEventCreate(&startEvent) );
        checkCuda( cudaEventCreate(&stopEvent) );  
        checkCuda( cudaEventRecord(startEvent, 0) );
    }

    lookuptable_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(in_gpu,iny_gpu,d_lookup, d_xxresult,d_yyresult,d_xyresult,d_yxresult);
    
    if (time) {
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) ); 
        printf("%f\n", ms);
    }

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

    if(validate)
    {   
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
                        printf(" %f+%fi ", resultxx[expIdx].x, resultxx[expIdx].y);
                    float diffx = abs(resultxx[expIdx].x - exp[expIdx].x);    
                    float diffy = abs(resultxx[expIdx].y - exp[expIdx].y);    
                    float tol = 1e2;
                    if ((diffx > tol) || (diffy > tol)) {
                        printf("out[%d]=%f + %fi != exp[%d]=%f + %fi\n", expIdx, resultxx[expIdx].x, resultxx[expIdx].y, expIdx, exp[expIdx].x, exp[expIdx].y);
                        printf("diff x=%f y=%f\n", diffx, diffy);
                        return 1;
                    }
                }
                if (verbose)
                    printf("\n");
            }
        }
        printf("test_cyclid_corr_accum passed\n");
    }
    return 1;
}


int main() {
    printf("Realworld_data_cyclid_gpu\n");

    cycfold_struct cs;

    cs.ncyc = 128;
    cs.nlag = (cs.ncyc/2) + 1; 
    cs.numTimeSamplesHfft = 256250;
    cs.nBlocks = 1;
    cs.nchanPfb = 1;
    cs.numPhaseBins = 256;

    int phaseBinLookupSize = (2*cs.numTimeSamplesHfft) + cs.nlag - 2;
    unsigned *phaseBins;
    phaseBins = (unsigned *)malloc(phaseBinLookupSize *sizeof(unsigned));
    memset(phaseBins, 0, phaseBinLookupSize*sizeof(int));

    int phaseStep = phaseBinLookupSize / cs.numPhaseBins;
    for (int iphase=0; iphase<cs.numPhaseBins; iphase++) {
        int start = iphase*phaseStep;
        int end = (iphase+1)*phaseStep;
        for (int j=start; j<end; j++)
            phaseBins[j] = iphase;
    }

    int phaseCnts[cs.numPhaseBins];
    memset(phaseCnts, 0, cs.numPhaseBins*sizeof(int));
    for (int i = 0; i < cs.numTimeSamplesHfft; i++)
        for (int ilag = 0; ilag < cs.nlag; ilag++)
            phaseCnts[phaseBins[(2*i) + ilag]]++;
    
    int phaseCntTotal = 0;
    for (int i=0; i<cs.numPhaseBins; i++) {
        phaseCntTotal += phaseCnts[i];
    }    
    assert(phaseCntTotal == cs.numTimeSamplesHfft * cs.nlag);    

    bool time = true;
    bool maxOccupancy = false;
    bool verbose = false;
    bool validate = true;

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
    float fvalue = 1.0;
    float imgDiv = 1.0;

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

    call_all_polarisation_kernel(out,inSize,profileSize,phaseBinLookupSize,inSize2,nlag,in,iny,phaseBins,nPhaseBins,nchan,iblock,ichan,maxOccupancy,time,verbose,validate);

    free(phaseBins);
    free(in);
    free(iny);
    free(exp);
    free(out);

}