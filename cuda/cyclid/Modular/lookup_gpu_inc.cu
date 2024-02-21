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

int GPU_BLOCK_SIZE = 256 * 4;
const int LOOP_SIZE = 64;
const int BLOCK_SIZE = 64;
const int Y_THREADS = 16;

inline
cudaError_t checkCuda(cudaError_t result) {
  #if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  #endif
  return result;
}
__global__ void lookuptable_kernel(unsigned int * d_last_index, float2 * in1, float2 * in2, int2 * d_lookup, float2 * d_xxresult, float2 * d_yyresult, float2 * d_xyresult, float2 * d_yxresult) {

  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int threadrow = threadIdx.y;
  __shared__ float2 memxx[BLOCK_SIZE][Y_THREADS];
  __shared__ float2 memyy[BLOCK_SIZE][Y_THREADS];
  __shared__ float2 memxy[BLOCK_SIZE][Y_THREADS];
  __shared__ float2 memyx[BLOCK_SIZE][Y_THREADS];

  if(rowIdx<16640)
  {
    float2 sumxx, sumyy, sumxy, sumyx;
    sumxx.x = 0.0f;
    sumxx.y = 0.0f;
    sumyy.x = 0.0f;
    sumyy.y = 0.0f;
    sumxy.x = 0.0f;
    sumxy.y = 0.0f;
    sumyx.x = 0.0f;
    sumyx.y = 0.0f;

    int start = (threadrow * LOOP_SIZE);
    int end = start + LOOP_SIZE;

    if (end >= 1003) end = 1002;

    for (int colIdx = start; colIdx < end; colIdx++) {

        int2 current = d_lookup[rowIdx * 1003 + colIdx];

        if (d_last_index[rowIdx]==colIdx) break;

        int current_x = current.x;
        int current_y = current.y;

        float in1y_x = in1[current_y].x;
        float in1x_x = in1[current_x].x;
        float in1y_y = in1[current_y].y;
        float in1x_y = in1[current_x].y;

        float in2y_x = in2[current_y].x;
        float in2x_x = in2[current_x].x;
        float in2y_y = in2[current_y].y;
        float in2x_y = in2[current_x].y;

        float2 product;

        product.x = (in1y_x * in1x_x) + (in1y_y * in1x_y);
        product.y = (in1y_y * in1x_x) - (in1y_x * in1x_y);

        sumxx.x += product.x;
        sumxx.y -= product.y;

        product.x = (in2y_x * in2x_x) + (in2y_y * in2x_y);
        product.y = (in2y_y * in2x_x) - (in2y_x * in2x_y);

        sumyy.x += product.x;
        sumyy.y -= product.y;

        product.x = (in1y_x * in2x_x) + (in1y_y * in2x_y);
        product.y = (in1y_y * in2x_x) - (in1y_x * in2x_y);

        sumxy.x += product.x;
        sumxy.y -= product.y;

        product.x = (in2y_x * in1x_x) + (in2y_y * in1x_y);
        product.y = (in2y_y * in1x_x) - (in2y_x * in1x_y);

        sumyx.x += product.x;
        sumyx.y -= product.y;

    }

    memxx[threadIdx.x][threadIdx.y].x = sumxx.x;
    memxx[threadIdx.x][threadIdx.y].y = sumxx.y;

    memyy[threadIdx.x][threadIdx.y].x = sumyy.x;
    memyy[threadIdx.x][threadIdx.y].y = sumyy.y;

    memxy[threadIdx.x][threadIdx.y].x = sumxy.x;
    memxy[threadIdx.x][threadIdx.y].y = sumxy.y;

    memyx[threadIdx.x][threadIdx.y].x = sumyx.x;
    memyx[threadIdx.x][threadIdx.y].y = sumyx.y;

    __syncthreads();

    float2 xxfinal_sum, yyfinal_sum, xyfinal_sum, yxfinal_sum;
    xxfinal_sum.x = 0;
    xxfinal_sum.y = 0;
    yyfinal_sum.x = 0;
    yyfinal_sum.y = 0;
    xyfinal_sum.x = 0;
    xyfinal_sum.y = 0;
    yxfinal_sum.x = 0;
    yxfinal_sum.y = 0;

    if (threadIdx.y == 0) {
      for (int i = 0; i < Y_THREADS; i++) {
        xxfinal_sum.x += memxx[threadIdx.x][i].x;
        xxfinal_sum.y += memxx[threadIdx.x][i].y;
        yyfinal_sum.x += memyy[threadIdx.x][i].x;
        yyfinal_sum.y += memyy[threadIdx.x][i].y;
        xyfinal_sum.x += memxy[threadIdx.x][i].x;
        xyfinal_sum.y += memxy[threadIdx.x][i].y;
        yxfinal_sum.x += memyx[threadIdx.x][i].x;
        yxfinal_sum.y += memyx[threadIdx.x][i].y;
      }

      d_xxresult[rowIdx].x = xxfinal_sum.x;
      d_xxresult[rowIdx].y = xxfinal_sum.y;

      d_yyresult[rowIdx].x = yyfinal_sum.x;
      d_yyresult[rowIdx].y = yyfinal_sum.y;

      d_xyresult[rowIdx].x = xyfinal_sum.x;
      d_xyresult[rowIdx].y = xyfinal_sum.y;

      d_yxresult[rowIdx].x = yxfinal_sum.x;
      d_yxresult[rowIdx].y = yxfinal_sum.y;
    }
  }
}

__global__ void generate_lookup(unsigned int * d_last_index, int2 * lookuptable, float2 * in1, float2 * in2, size_t size1, size_t size2, float2 * outXX, float2 * outYY, float2 * outXY, float2 * outYX, unsigned * phaseBins, int numPhaseBins, int numPfbChans, int nlag, int iblock, int pfbChan, size_t outSz, int lookupBlockLen, bool verbose, int maxvar) {
  int inIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned phaseBin;
  int phaseBinIdx = (iblock * lookupBlockLen) + (2 * inIdx) + yIdx;
  phaseBin = phaseBins[phaseBinIdx];

  int expIdx = (phaseBin * nlag * numPfbChans) + (nlag * pfbChan) + yIdx;

  if ((inIdx < size2) && (expIdx < outSz) && (yIdx < nlag)) {
    int j = inIdx + yIdx;
    unsigned int index = atomicInc( &d_last_index[expIdx], 1003);
    int lookupIdx = expIdx * 1003 + index;
    lookuptable[lookupIdx].x = inIdx;
    lookuptable[lookupIdx].y = j;
    
  }
}

void complexConjMult(float2 in1, float2 in2, float2 * tmp) {
  tmp -> x = (in2.x * in1.x) - (in2.y * -1.0 * in1.y);
  tmp -> y = (in2.x * -1.0 * in1.y) + (in2.y * in1.x);
  tmp -> y = -1.0 * tmp -> y;
}

void reference_code(float2 * in, float2 * exp, int inSize2, int nlag, int nchan, int ichan, int nPhaseBins, unsigned * phaseBins, bool verbose) {
  int phaseBinIdx, phaseBin, expIdx;
  float2 tmp, in1, in2;
  for (int i = 0; i < inSize2; i++) {
    if (verbose)
      printf("\n%d\n", i);
    in1 = in [i];
    for (int ilag = 0; ilag < nlag; ilag++) {
      if (verbose)
        printf(" ilag=%d ", ilag);
      in2 = in [i + ilag];
      complexConjMult(in1, in2, & tmp);

      // now accumulate in the right phase bin
      phaseBinIdx = (2 * i) + ilag;
      phaseBin = phaseBins[phaseBinIdx];
      expIdx = (phaseBin * nlag * nchan) + (nlag * ichan) + ilag;
      if (verbose) {
        printf(" pb=%d ", phaseBin);
        printf(" pi=%d ", phaseBinIdx);
        printf(" ei=%d ", expIdx);
      }
      // accumulate (fold)
      exp[expIdx].x += tmp.x;
      exp[expIdx].y += tmp.y;
    }

  }

  if (verbose)
    printf("\ncomputed expected results:\n");
  for (int iphase = 0; iphase < nPhaseBins; iphase++) {
    if (verbose)
      printf("phase %d\n", iphase);
    for (ichan = 0; ichan < nchan; ichan++) {
      if (verbose)
        printf("chan %d\n", ichan);
      for (int ilag = 0; ilag < nlag; ilag++) {
        expIdx = (iphase * nlag * nchan) + (nlag * ichan) + ilag;
        if (verbose)
          printf(" %f+%fi ", exp[expIdx].x, exp[expIdx].y);
      }
      if (verbose)
        printf("\n");
    }
  }
}

void call_all_polarisation_kernel(float2 * out, int inSize, int profileSize, int phaseBinLookupSize, int inSize2, int nlag, float2 * in, float2 * iny, unsigned * phaseBins, int nPhaseBins, int nchan, int iblock, int ichan, bool maxOccupancy, bool time, bool verbose) {
  int phaseBinIdx, phaseBin, expIdx;
  int maxvar = 0;
  int2 *lookuptable;
  lookuptable = (int2 * ) malloc(16640 * 1003 * sizeof(int2));

  for (int c = 0; c < 16640 * 1003; c++) {
    lookuptable[c].x = -1;
  }
  for (int i = 0; i < inSize2; i++) {
    for (int ilag = 0; ilag < nlag; ilag++) {
      phaseBinIdx = (2 * i) + ilag;
      phaseBin = phaseBins[phaseBinIdx];
      expIdx = (phaseBin * nlag * nchan) + (nlag * ichan) + ilag;
      int index = (lookuptable[expIdx * 1003].x);
      if (index == -1) {
        lookuptable[expIdx * 1003].x = 1;
        lookuptable[expIdx * 1003 + (lookuptable[expIdx * 1003].x)].x = i;
        lookuptable[expIdx * 1003 + (lookuptable[expIdx * 1003].x)].y = i + ilag;
        lookuptable[expIdx * 1003].x = (lookuptable[expIdx * 1003].x) + 1;
      } else {
        lookuptable[expIdx * 1003 + (lookuptable[expIdx * 1003].x)].x = i;
        lookuptable[expIdx * 1003 + (lookuptable[expIdx * 1003].x)].y = i + ilag;
        lookuptable[expIdx * 1003].x = (lookuptable[expIdx * 1003].x) + 1;
      }
    }
  }
  // FILE *cpu;
  // cpu=fopen("cpu.txt","w");
  // for(int i=0;i<1000;i++)
  // {
  //   printf("%f\t%f\n",lookuptable[i].x,lookuptable[i].y);
  // }
  // fclose(cpu);
  int2 * d_lookup;
  printf("\n\nALL POLARISATION KERNEL\n\n");
  float2 * in_gpu, * iny_gpu, * out_gpu, * outyy_gpu, * outxy_gpu, * outyx_gpu;
  unsigned * phaseBins_gpu;
  unsigned int * d_last_index;

  cudaMalloc((float2 ** ) & in_gpu, inSize * sizeof(float2));
  cudaMalloc((float2 ** ) & iny_gpu, inSize * sizeof(float2));

  cudaMalloc((float2 ** ) & out_gpu, profileSize * sizeof(float2));
  cudaMalloc((float2 ** ) & outyy_gpu, profileSize * sizeof(float2));
  cudaMalloc((float2 ** ) & outxy_gpu, profileSize * sizeof(float2));
  cudaMalloc((float2 ** ) & outyx_gpu, profileSize * sizeof(float2));

  cudaMalloc((unsigned ** ) & phaseBins_gpu, phaseBinLookupSize * sizeof(unsigned));

  cudaMemset(out_gpu, 0, profileSize * sizeof(float2));
  cudaMemset(outyy_gpu, 0, profileSize * sizeof(float2));
  cudaMemset(outxy_gpu, 0, profileSize * sizeof(float2));
  cudaMemset(outyx_gpu, 0, profileSize * sizeof(float2));

  cudaMemcpy(in_gpu, in, inSize * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(iny_gpu, iny, inSize * sizeof(float2), cudaMemcpyHostToDevice);

  cudaMemcpy(phaseBins_gpu, phaseBins, phaseBinLookupSize * sizeof(unsigned), cudaMemcpyHostToDevice);

  int gridX, gridY, threadX, threadY;
  if (inSize < 128) {
    gridX = 1;
    gridY = 1;
    threadX = inSize2;
    threadY = nlag;
  } else {
    if (maxOccupancy) {
      int gpuGridSize = ((inSize2 + 256) / 256);
      gridX = gpuGridSize;
      gridY = (nlag + 4) / 4;
      threadX = 256; ///thisGpuBlockSize; //GPU_BLOCK_SIZE / nlag;
      threadY = 4; //nlag;
    } else {
      threadX = GPU_BLOCK_SIZE / nlag;
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
    float diffPct = ((numKernelCalls - (inSize2 * nlag)) / numKernelCalls) * 100.0;
    printf("num null threads: %d, %f percent\n", numKernelCalls - (inSize2 * nlag), diffPct);
  }

  dim3 grids(gridX, gridY, 1);
  dim3 threads(threadX, threadY, 1);
  ichan = 0;

  dim3 NUM_THREADS(BLOCK_SIZE, Y_THREADS);
  dim3 NUM_BLOCKS(((16640 * Y_THREADS) / 1024), 1, 1);
  assert(threadX * threadY <= GPU_BLOCK_SIZE);

  cudaEvent_t startEvent, stopEvent;
  float ms;
  if (time) {
    checkCuda(cudaEventCreate( & startEvent));
    checkCuda(cudaEventCreate( & stopEvent));
    checkCuda(cudaEventRecord(startEvent, 0));
  }

  cudaMalloc((int2 ** ) & d_lookup, 16640 * 1003 * sizeof(int2));

  cudaMalloc((unsigned int ** ) & d_last_index, 16640 * sizeof(unsigned int));
  cudaMemset(d_last_index, 0, 16640 * sizeof(unsigned int));
  
  generate_lookup <<< grids, threads >>> (d_last_index, d_lookup, in_gpu, iny_gpu, inSize, inSize2, out_gpu, outyy_gpu, outxy_gpu, outyx_gpu, phaseBins_gpu, nPhaseBins, nchan, nlag, iblock, ichan, profileSize, phaseBinLookupSize, verbose, maxvar);
  //cudaDeviceSynchronize();
  lookuptable_kernel << < NUM_BLOCKS, NUM_THREADS >>> (d_last_index, in_gpu, iny_gpu, d_lookup, out_gpu, outyy_gpu, outxy_gpu, outyx_gpu);

  if (time) {
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime( & ms, startEvent, stopEvent));
    printf("%f\n", ms);
  }
  int2 *gpulookuptable;
  gpulookuptable = (int2 * ) malloc(16640 * 1003 * sizeof(int2));
  cudaMemcpy(gpulookuptable, d_lookup, 16640 * sizeof(int2) * 1003, cudaMemcpyDeviceToHost);
  cudaMemcpy(out, out_gpu, profileSize * sizeof(float2), cudaMemcpyDeviceToHost);
  // FILE *gpu;
  // gpu=fopen("gpu.txt","w");
  // for(int i=0;i<16640*1003;i++)
  // {
  //   fprintf(gpu,"%f\t%f\n",gpulookuptable[i].x,gpulookuptable[i].y);
  // }
  // fclose(gpu);
  
  cudaFree(in_gpu);
  cudaFree(iny_gpu);
  cudaFree(out_gpu);
  cudaFree(outyy_gpu);
  cudaFree(outxy_gpu);
  cudaFree(outyx_gpu);
  cudaFree(phaseBins_gpu);
  cudaFree(d_lookup);
  cudaFree(d_last_index);

}

int validate_results(int nPhaseBins, int nchan, int nlag, float2 * out, float2 * exp, bool verbose) {
  int expIdx;
  float max_diffx=0;
  float max_diffy=0;
  if (verbose)
    printf("\nresults:\n");
  for (int iphase = 0; iphase < nPhaseBins; iphase++) {
    if (verbose)
      printf("phase %d\n", iphase);
    for (int ichan = 0; ichan < nchan; ichan++) {
      if (verbose)
        printf("chan %d\n", ichan);
      for (int ilag = 0; ilag < nlag; ilag++) {
        expIdx = (iphase * nlag * nchan) + (nlag * ichan) + ilag;
        if (verbose)
          printf(" %f+%fi ", out[expIdx].x, out[expIdx].y);
        float diffx = abs(out[expIdx].x - exp[expIdx].x);
        float diffy = abs(out[expIdx].y - exp[expIdx].y);
        float tol = 1e2;
        if(diffx>max_diffx)
        {
            max_diffx=diffx;
        }
        if(diffy>max_diffy)
        {
            max_diffy=diffy;
        }
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
  printf("Max X:%f\tMax Y:%f\n",max_diffx,max_diffy);
  return 1;
}

int main() {
  printf("Realworld_data_cyclid_gpu\n");

  cycfold_struct cs;

  cs.ncyc = 128;
  cs.nlag = (cs.ncyc / 2) + 1;
  cs.numTimeSamplesHfft = 256250;
  cs.nBlocks = 1;
  cs.nchanPfb = 1;
  cs.numPhaseBins = 256;

  int phaseBinLookupSize = (2 * cs.numTimeSamplesHfft) + cs.nlag - 2;
  unsigned * phaseBins;
  phaseBins = (unsigned * ) malloc(phaseBinLookupSize * sizeof(unsigned));
  memset(phaseBins, 0, phaseBinLookupSize * sizeof(int));

  int phaseStep = phaseBinLookupSize / cs.numPhaseBins;
  for (int iphase = 0; iphase < cs.numPhaseBins; iphase++) {
    int start = iphase * phaseStep;
    int end = (iphase + 1) * phaseStep;
    for (int j = start; j < end; j++)
      phaseBins[j] = iphase;
  }

  int phaseCnts[cs.numPhaseBins];
  memset(phaseCnts, 0, cs.numPhaseBins * sizeof(int));
  for (int i = 0; i < cs.numTimeSamplesHfft; i++)
    for (int ilag = 0; ilag < cs.nlag; ilag++)
      phaseCnts[phaseBins[(2 * i) + ilag]]++;

  int phaseCntTotal = 0;
  for (int i = 0; i < cs.numPhaseBins; i++) {
    phaseCntTotal += phaseCnts[i];
  }
  assert(phaseCntTotal == cs.numTimeSamplesHfft * cs.nlag);

  bool time = true;
  bool maxOccupancy = true;
  bool verbose = false;

  int nlag = cs.nlag;
  size_t inSize = cs.numTimeSamplesHfft;
  size_t inSize2 = inSize - nlag - 1;
  phaseBinLookupSize = (2 * inSize) + nlag - 2;
  int nPhaseBins = cs.numPhaseBins;
  int nchan = cs.nchanPfb;
  int ichan = 0;
  int iblock = 0;

  float2 * in, * iny;
  in = (float2 * ) malloc(inSize * sizeof(float2));
  iny = (float2 * ) malloc(inSize * sizeof(float2));
  memset(in, 0, inSize * sizeof(float2));
  memset(iny, 0, inSize * sizeof(float2));
  int maxValue = 127;
  int value = 0;
  float fvalue = 1.0;
  float imgDiv = 1.0;

  for (int i = 0; i < inSize; i++) {
    in [i].x = ((float) value) + fvalue;
    in [i].y = ((float)((float) value) / imgDiv) + fvalue;
    iny[i].x = in [i].x;
    iny[i].y = in [i].y;
    value++;
    if (value + fvalue > maxValue)
      value = 0;
  }

  if (verbose) {
    printf("inSize=%ld, inSize2=%ld\n", inSize, inSize2);
    printf("phaseBinLookupSize=%d\n", phaseBinLookupSize);
    for (int i = 0; i < inSize; i++)
      printf("in[%d]=%f+%fi\n", i, in [i].x, in [i].y);
  }

  float2 * exp;
  size_t profileSize = nPhaseBins * nchan * nlag;
  exp = (float2 * ) malloc(profileSize * sizeof(float2));
  memset(exp, 0, profileSize * sizeof(float2));

  reference_code(in, exp, inSize2, nlag, nchan, ichan, nPhaseBins, phaseBins, verbose);

  float2 * out;
  out = (float2 * ) malloc(profileSize * sizeof(float2));

  call_all_polarisation_kernel(out, inSize, profileSize, phaseBinLookupSize, inSize2, nlag, in, iny, phaseBins, nPhaseBins, nchan, iblock, ichan, maxOccupancy, time, verbose);

  validate_results(nPhaseBins, nchan, nlag, out, exp, verbose);

  free(phaseBins);
  free(in);
  free(iny);
  free(exp);
  free(out);

}
