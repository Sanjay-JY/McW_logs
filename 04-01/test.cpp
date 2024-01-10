#include<iostream>
#include<immintrin.h>
#include<chrono>
int main()
{
    double r1[4][4]={{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    double ans[4][4]={0};
    double res[4][4]={0};

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto tym1 = high_resolution_clock::now();

    __m256d row1,row2,row3,row4;
    __m256d trow1,trow2,trow3,trow4;
    __m256d t1,t2,t3;

    row1=_mm256_loadu_pd(&r1[0][0]);
    row2=_mm256_loadu_pd(&r1[1][0]);
    row3=_mm256_loadu_pd(&r1[2][0]);
    row4=_mm256_loadu_pd(&r1[3][0]);

    t1=_mm256_unpacklo_pd(row1,row2);       // t1 = 1 5 3 7
    t2=_mm256_unpacklo_pd(row3,row4);       // t2 = 9 13 11 15
    t3=_mm256_permute2f128_pd(t2,t2,1);     // t3 = 11 15 9 13
    trow1=_mm256_blend_pd(t1,t3,0b1100);    // trow1 = 1 5 9 13

    t3=_mm256_permute2f128_pd(t1,t1,1);     // t3 = 3 7 1 5
    trow3=_mm256_blend_pd(t3,t2,0b1100);    // trow3 = 3 7 11 15

    t1=_mm256_unpackhi_pd(row1,row2);       // t1 = 2 6 4 8
    t2=_mm256_unpackhi_pd(row3,row4);       // t2 = 10 14 12 16  
    t3=_mm256_permute2f128_pd(t2,t2,1);     // t3 = 12 16 10 14
    trow2=_mm256_blend_pd(t1,t3,0b1100);    // trow2 = 2 6 10 14

    t3=_mm256_permute2f128_pd(t1,t1,1);     // t3 = 4 8 2 6
    trow4=_mm256_blend_pd(t3,t2,0b1100);    // trow4 = 4 8 12 16

    for(int i=0;i<4;i++)
    {   
        __m256d row;
        __m256d add1;
        __m256d add2;
        __m256d add3;
        __m256d add4;
        switch (i)
        {
        case 0:
            row=row1;
            break;
        case 1:
            row=row2;
            break;
        case 2:
            row=row3;
            break;
        case 3:
            row=row4;
            break;
        }   
        __m256d add;                                                     
        for(int j=0;j<4;j++)
        {
            __m256d sum,col;
            switch (j)
            {
            case 0:
                col=trow1;
                break;
            case 1:
                col=trow2;
                break;
            case 2:
                col=trow3;
                break;
            case 3:
                col=trow4;
                break;
            }
            double temp[4];
            sum=_mm256_set_pd(0,0,0,0);
            sum=_mm256_mul_pd(row,col);
            sum = _mm256_hadd_pd(sum,_mm256_permute2f128_pd(sum,sum,1));
            switch (j)
            {
            case 0:
                add = _mm256_blend_pd(add,_mm256_hadd_pd(sum,sum),0b0001);
            case 1:
                add = _mm256_blend_pd(add,_mm256_hadd_pd(sum,sum),0b0010);
            case 2:
                add = _mm256_blend_pd(add,_mm256_hadd_pd(sum,sum),0b0100);
            case 3:
                add = _mm256_blend_pd(add,_mm256_hadd_pd(sum,sum),0b1000);
            }
        }
        _mm256_storeu_pd(&res[i][0],add);
        // switch (i)
        // {
        // case 0:
        //     add1=_mm256_set_pd(add);
        //     break;
        // case 1:
        //     row=row2;
        //     break;
        // case 2:
        //     row=row3;
        //     break;
        // case 3:
        //     row=row4;
        //     break;
        // }
    }
    std::cout<<"\n";
    auto tym2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = tym2 - tym1;
    std::cout << ms_double.count() << "ms\n";

    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            std::cout<<res[i][j]<<"\t";
        }
        std::cout<<"\n";
    }    
    std::cout<<"\n";
}