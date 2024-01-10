#include<iostream>
#include<immintrin.h>
#include<chrono>
#include<vector>
struct blocks{
    std::vector<__m256d> matrix;
    std::vector<__m256d> transpose_matrix;
    std::vector<__m256d> res_matrix;
    std::vector<__m256d> answer;
};

int main()
{
    double r1[16][16];
    int x=1;
    for(int i=0;i<16;i++)
    {
        for(int j=0;j<16;j++)
        {
            r1[i][j]=x;
            x++;
        }
    }
    for(int i=0;i<16;i++)
    {
        for(int j=0;j<16;j++)
        {
            std::cout<<r1[i][j]<<"\t";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
    struct blocks block[16];
    
    double ans[4][4]={0};

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto tym1 = high_resolution_clock::now();

    __m256d row1,row2,row3,row4;
    __m256d trow1,trow2,trow3,trow4;
    __m256d t1,t2,t3;
    int k=0;
    for(int i=0;i<16;i+=4)
    {
        for(int j=0;j<16;j+=4)
        {
            __m256d row;
            __m256d trow1,trow2;
            __m256d t1,t2,t3;
            row=_mm256_loadu_pd(&r1[i][j]);
            block[k].matrix.push_back(row);
            row=_mm256_loadu_pd(&r1[i+1][j]);
            block[k].matrix.push_back(row);
            row=_mm256_loadu_pd(&r1[i+2][j]);
            block[k].matrix.push_back(row);
            row=_mm256_loadu_pd(&r1[i+3][j]);
            block[k].matrix.push_back(row);

            t1=_mm256_unpacklo_pd(block[k].matrix[0],block[k].matrix[1]);       // t1 = 1 5 3 7
            t2=_mm256_unpacklo_pd(block[k].matrix[2],block[k].matrix[3]);       // t2 = 9 13 11 15
            t3=_mm256_permute2f128_pd(t2,t2,1);     // t3 = 11 15 9 13
            trow1=_mm256_blend_pd(t1,t3,0b1100);    // trow1 = 1 5 9 13
            block[k].transpose_matrix.push_back(trow1);
            
            t3=_mm256_permute2f128_pd(t1,t1,1);     // t3 = 3 7 1 5
            trow2=_mm256_blend_pd(t3,t2,0b1100);    // trow3 = 3 7 11 15

            t1=_mm256_unpackhi_pd(block[k].matrix[0],block[k].matrix[1]);       // t1 = 2 6 4 8
            t2=_mm256_unpackhi_pd(block[k].matrix[2],block[k].matrix[3]);       // t2 = 10 14 12 16  
            t3=_mm256_permute2f128_pd(t2,t2,1);     // t3 = 12 16 10 14
            trow1=_mm256_blend_pd(t1,t3,0b1100);     // trow2 = 2 6 10 14
            block[k].transpose_matrix.push_back(trow1);
            block[k].transpose_matrix.push_back(trow2);

            t3=_mm256_permute2f128_pd(t1,t1,1);     // t3 = 4 8 2 6
            trow1=_mm256_blend_pd(t3,t2,0b1100);    // trow4 = 4 8 12 16
            block[k].transpose_matrix.push_back(trow1);

            k++;
        }
    }
    for(int i=0;i<4;i++)    
    {
        for(int j=0;j<4;j++)    
        {
            __m256d row,add;
            row=block[i].matrix[j];
            for(int k=0;k<4;k++)
            {
                __m256d sum,col;
                col=block[i+3*i].transpose_matrix[k];
                sum=_mm256_set_pd(0,0,0,0);
                sum=_mm256_mul_pd(row,col);
                sum = _mm256_hadd_pd(sum,_mm256_permute2f128_pd(sum,sum,1));
                switch (k)
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
            block[i].res_matrix.push_back(add);
        }
    }
    for(int i=0;i<4;i++)
    {
        __m256d row1;
        row1=_mm256_add_pd(block[0].res_matrix[i],block[1].res_matrix[i]);
        row1=_mm256_add_pd(row1,block[2].res_matrix[i]);
        row1=_mm256_add_pd(row1,block[3].res_matrix[i]);
        block[0].answer.push_back(row1);
    }

    std::cout<<"\n";
    auto tym2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = tym2 - tym1;
    std::cout << ms_double.count() << "ms\n";

    std::cout<<"test\n";
    double temp1[4]={0};
    double temp2[4]={0};
    double temp3[4]={0};
    double temp4[4]={0};
    _mm256_storeu_pd(&temp1[0],block[0].answer[0]);
    _mm256_storeu_pd(&temp2[0],block[0].answer[1]);
    _mm256_storeu_pd(&temp3[0],block[0].answer[2]);
    _mm256_storeu_pd(&temp4[0],block[0].answer[3]);

    for(int i=0;i<4;i++)
    {
        std::cout<<temp1[i]<<"\t";
    }
    std::cout<<"\n";
    for(int i=0;i<4;i++)
    {
        std::cout<<temp2[i]<<"\t";
    }
    std::cout<<"\n";
    for(int i=0;i<4;i++)
    {
        std::cout<<temp3[i]<<"\t";
    }
    std::cout<<"\n";
    for(int i=0;i<4;i++)
    {
        std::cout<<temp4[i]<<"\t";
    }
    std::cout<<"\n";

}