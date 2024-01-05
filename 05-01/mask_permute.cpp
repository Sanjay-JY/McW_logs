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
    __m256d t1,t2,t3;

    row1=_mm256_loadu_pd(&r1[0][0]);
    row2=_mm256_loadu_pd(&r1[1][0]);
    row3=_mm256_loadu_pd(&r1[2][0]);
    row4=_mm256_loadu_pd(&r1[3][0]);

    t1=_mm256_unpacklo_pd(row1,row2);
    t2=_mm256_unpacklo_pd(row3,row4);
    t3=_mm256_permute2f128_pd(t2,t2,1);
    t3=_mm256_blend_pd(t1,t3,0b1100);
    _mm256_storeu_pd(&ans[0][0],t3);

    t3=_mm256_permute2f128_pd(t1,t1,1);
    t3=_mm256_blend_pd(t3,t2,0b1100);
    _mm256_storeu_pd(&ans[2][0],t3);

    t1=_mm256_unpackhi_pd(row1,row2);
    t2=_mm256_unpackhi_pd(row3,row4);
    t3=_mm256_permute2f128_pd(t2,t2,1);
    t3=_mm256_blend_pd(t1,t3,0b1100);
    _mm256_storeu_pd(&ans[1][0],t3);

    t3=_mm256_permute2f128_pd(t1,t1,1);
    t3=_mm256_blend_pd(t3,t2,0b1100);
    _mm256_storeu_pd(&ans[3][0],t3);

    for(int i=0;i<4;i++)
    {
        __m256d row=_mm256_loadu_pd(&r1[i][0]);
        for(int j=0;j<4;j++)
        {
            __m256d sum,col,mask;
            double temp[4];
            sum=_mm256_set_pd(0,0,0,0);
            col=_mm256_loadu_pd(&ans[j][0]);
            sum=_mm256_fmadd_pd(row,col,sum);
            __m256d add = _mm256_hadd_pd(sum,_mm256_permute2f128_pd(sum,sum,1));
            add = _mm256_hadd_pd(add,add);
            _mm256_storeu_pd(&temp[0],add);
            res[i][j]=temp[0];
        }
    }
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