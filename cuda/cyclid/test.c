#include <stdio.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

struct cycfold_struct {
    unsigned ncyc;
    unsigned nlag;
    unsigned nchanPfb; 
    size_t numPhaseBins;
    unsigned numTimeSamplesHfft;
    unsigned nBlocks;
};


struct float2 {
    float x;
    float y;
};

struct int2 {
    int x;
    int y;
};

void complexConjMult(struct float2 in1, struct float2 in2, struct float2 *tmp) {
    tmp->x = (in2.x * in1.x) - (in2.y * -1.0 * in1.y);
    tmp->y = (in2.x * -1.0 * in1.y) + (in2.y * in1.x);
    tmp->y = -1.0 * tmp->y;
}

int main()
{
    struct cycfold_struct cs;
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
    printf("phaselookup->%d\n",phaseBinLookupSize);
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

    bool verbose = false;
    int nlag = cs.nlag; //(ncyc/2) + 1;
    size_t inSize = cs.numTimeSamplesHfft; //8;
    size_t inSize2 = inSize - nlag - 1;
    //int phaseBinLookupSize = (2*inSize) + nlag - 2;
    int nPhaseBins = cs.numPhaseBins;
    int nchan = cs.nchanPfb;
    int ichan = 0;
    int iblock = 0;
    
    // init input
    struct float2 *in, *iny;
    in = (struct float2 *)malloc(inSize * sizeof(struct float2));
    iny = (struct float2 *)malloc(inSize * sizeof(struct float2));
    memset(in, 0, inSize*sizeof(struct float2));
    memset(iny, 0, inSize*sizeof(struct float2));
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
    // printf("\nin = ");
    // for(int i=0;i<inSize;i++)
    // {
    //     printf("(%.2f,%.2f)",in[i].x,in[i].y);
    // }
    // printf("\niny = ");
    // for(int i=0;i<inSize;i++)
    // {
    //     printf("(%.2f,%.2f)",iny[i].x,iny[i].y);
    // }
    // printf("\n");
    if (verbose) {
        printf("inSize=%ld, inSize2=%ld\n", inSize, inSize2);
        printf("phaseBinLookupSize=%d\n", phaseBinLookupSize);
        for (int i = 0; i<inSize; i++) 
            printf("in[%d]=%f+%fi\n", i, in[i].x, in[i].y);
    }


    int phaseBinIdx, phaseBin, expIdx;
    struct float2 tmp, in1, in2;
    struct float2 *exp;
    size_t profileSize = nPhaseBins * nchan * nlag;
    exp = (struct float2 *)malloc(profileSize * sizeof(struct float2));
    memset(exp, 0, profileSize*sizeof(struct float2));

    struct int2 *lookuptable;
    lookuptable= (struct int2 *)malloc(16640*1003* sizeof(struct int2));
    for(int c=0;c<16640*1003;c++)
    {
        lookuptable[c].x=-1;
    }

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
            //printf("binIdx->%d\n",phaseBinIdx);
            phaseBin = phaseBins[phaseBinIdx];
            expIdx = (phaseBin * nlag * nchan) + (nlag * ichan) + ilag; 
            //printf("exp->%d\n",expIdx);
            if (verbose) {
                printf(" pb=%d ",phaseBin);
                printf(" pi=%d ",phaseBinIdx);
                printf(" ei=%d ", expIdx);
            }
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
                // if(expIdx==16444)
                // {
                //     //fprintf(filePointer, "exp->%d\tlast->%d\tcidx->%d\tsize->%d\n",expIdx,(lookuptable[expIdx*1003].x),expIdx*1003 +(lookuptable[expIdx*1003].x),16440*1003* sizeof(int2));
                //     printf("exp->%d\tlast->%d\tcidx->%d\tsize->%d\n",expIdx,(lookuptable[expIdx*1003].x),expIdx*1003 +(lookuptable[expIdx*1003].x),16640*1003* sizeof(struct int2));
                // }
                lookuptable[expIdx*1003 +(lookuptable[expIdx*1003].x)].x=i;
                lookuptable[expIdx*1003 + (lookuptable[expIdx*1003].x)].y=i+ilag;
                lookuptable[expIdx*1003].x= (lookuptable[expIdx*1003].x)+1;
            }

            float valx=tmp.x;
            float valy=tmp.y;
            // accumulate (fold)
            exp[expIdx].x += tmp.x;
            exp[expIdx].y += tmp.y;
            float outx=exp[expIdx].x;
            float outy=exp[expIdx].y;
        }
    
    }

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
    return 0;
}