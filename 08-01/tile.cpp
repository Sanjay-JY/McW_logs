#include <immintrin.h>
#include <xmmintrin.h>
#include <iostream>
#include <chrono>

using namespace std;

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    float transpose[8][8] = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
        {17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
        {25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0},
        {33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0},
        {41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0},
        {49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0},
        {57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0}
    };

    float matrix[8][8] = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
        {17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
        {25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0},
        {33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0},
        {41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0},
        {49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0},
        {57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0}
    };

    auto t1 = high_resolution_clock::now();

    __m256 row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13;

    row0 = _mm256_loadu_ps(&transpose[0][0]);
    row1 = _mm256_loadu_ps(&transpose[1][0]);
    row2 = _mm256_loadu_ps(&transpose[2][0]);
    row3 = _mm256_loadu_ps(&transpose[3][0]);

    row5 = _mm256_unpacklo_ps(row0, row1);
    row6 = _mm256_unpacklo_ps(row2, row3);

    row0 = _mm256_unpacklo_ps(row5, row6);
    row1 = _mm256_unpackhi_ps(row5, row6);

    __m256i permuteOrder = _mm256_set_epi32(7, 5, 6, 4, 3, 1, 2, 0);

    row0 = _mm256_permutevar8x32_ps(row0, permuteOrder);
    row1 = _mm256_permutevar8x32_ps(row1, permuteOrder);

    permuteOrder = _mm256_set_epi32(3, 1, 2, 0, 7, 6, 5, 4);
    row2 = _mm256_permutevar8x32_ps(row0, permuteOrder);
    row3 = _mm256_permutevar8x32_ps(row1, permuteOrder);

    row4 = _mm256_loadu_ps(&transpose[4][0]);
    row5 = _mm256_loadu_ps(&transpose[5][0]);
    row6 = _mm256_loadu_ps(&transpose[6][0]);
    row7 = _mm256_loadu_ps(&transpose[7][0]);

    row4 = _mm256_unpacklo_ps(row4, row5);
    row5 = _mm256_unpacklo_ps(row6, row7);

    row6 = _mm256_unpacklo_ps(row4, row5);
    row7 = _mm256_unpackhi_ps(row4, row5);

    permuteOrder = _mm256_set_epi32(7, 5, 6, 4, 3, 1, 2, 0);

    row4 = _mm256_permutevar8x32_ps(row6, permuteOrder);
    row5 = _mm256_permutevar8x32_ps(row7, permuteOrder);

    row0 = _mm256_permute2f128_ps(row0, row4, 0x20);
    row1 = _mm256_permute2f128_ps(row1, row5, 0x20);
    row2 = _mm256_permute2f128_ps(row2, row4, 0x30);
    row3 = _mm256_permute2f128_ps(row3, row5, 0x30);

    row4 = _mm256_loadu_ps(&transpose[0][0]);
    row5 = _mm256_loadu_ps(&transpose[1][0]);
    row6 = _mm256_loadu_ps(&transpose[2][0]);
    row7 = _mm256_loadu_ps(&transpose[3][0]);

    row4 = _mm256_unpackhi_ps(row4, row5);
    row5 = _mm256_unpackhi_ps(row6, row7);

    row6 = _mm256_unpacklo_ps(row4, row5);
    row7 = _mm256_unpackhi_ps(row4, row5);

    permuteOrder = _mm256_set_epi32(7, 5, 6, 4, 3, 1, 2, 0);

    row4 = _mm256_permutevar8x32_ps(row6, permuteOrder);
    row5 = _mm256_permutevar8x32_ps(row7, permuteOrder);

    permuteOrder = _mm256_set_epi32(3, 1, 2, 0, 7, 6, 5, 4);
    row6 = _mm256_permutevar8x32_ps(row4, permuteOrder);
    row7 = _mm256_permutevar8x32_ps(row5, permuteOrder);

    row8 = _mm256_loadu_ps(&transpose[4][0]);
    row9 = _mm256_loadu_ps(&transpose[5][0]);
    row10 = _mm256_loadu_ps(&transpose[6][0]);
    row11 = _mm256_loadu_ps(&transpose[7][0]);

    row12 = _mm256_unpackhi_ps(row8, row9);
    row13 = _mm256_unpackhi_ps(row10, row11);

    row8 = _mm256_unpacklo_ps(row12, row13);
    row9 = _mm256_unpackhi_ps(row12, row13);

    permuteOrder = _mm256_set_epi32(7, 5, 6, 4, 3, 1, 2, 0);

    row12 = _mm256_permutevar8x32_ps(row8, permuteOrder);
    row13 = _mm256_permutevar8x32_ps(row9, permuteOrder);

    row4 = _mm256_permute2f128_ps(row4, row12, 0x20);
    row5 = _mm256_permute2f128_ps(row5, row13, 0x20);
    row6 = _mm256_permute2f128_ps(row6, row12, 0x30);
    row7 = _mm256_permute2f128_ps(row7, row13, 0x30);

    for (int i = 0; i < 8; i++) {
        row8 = _mm256_loadu_ps(&matrix[i][0]);
        row9 = _mm256_mul_ps(row0, row8);

        __m128 sum128 = _mm256_extractf128_ps(row9, 0);
        sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float ans;
        _mm_store_ss(&ans, sum128);
        transpose[i][0] = ans;
    }
    for(int i=0;i<8;i++){
       row8= _mm256_loadu_ps (&matrix[i][0]);
       row9=_mm256_mul_ps(row1,row8);
       __m128 sum128 = _mm256_extractf128_ps(row9, 0);
            sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            
            float ans;
            _mm_store_ss(&ans, sum128);
           transpose[i][1]=ans;

    }
    for(int i=0;i<8;i++)
    {
        row8= _mm256_loadu_ps (&matrix[i][0]);
        row9=_mm256_mul_ps(row4,row8);
        __m128 sum128 = _mm256_extractf128_ps(row9, 0);
                sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float ans;
                _mm_store_ss(&ans, sum128);
            transpose[i][2]=ans;

    }
    for(int i=0;i<8;i++)
    {
        row8= _mm256_loadu_ps (&matrix[i][0]);
        row9=_mm256_mul_ps(row5,row8);
        __m128 sum128 = _mm256_extractf128_ps(row9, 0);
                sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float ans;
                _mm_store_ss(&ans, sum128);
            transpose[i][3]=ans;

    }
    for(int i=0;i<8;i++)
    {
        row8= _mm256_loadu_ps (&matrix[i][0]);
        row9=_mm256_mul_ps(row2,row8);
        __m128 sum128 = _mm256_extractf128_ps(row9, 0);
                sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float ans;
                _mm_store_ss(&ans, sum128);
            transpose[i][4]=ans;

    }
    for(int i=0;i<8;i++)
    {
        row8= _mm256_loadu_ps (&matrix[i][0]);
        row9=_mm256_mul_ps(row3,row8);
        __m128 sum128 = _mm256_extractf128_ps(row9, 0);
                sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float ans;
                _mm_store_ss(&ans, sum128);
            transpose[i][5]=ans;

    }
    for(int i=0;i<8;i++)
    {
        row8= _mm256_loadu_ps (&matrix[i][0]);
        row9=_mm256_mul_ps(row6,row8);
        __m128 sum128 = _mm256_extractf128_ps(row9, 0);
                sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float ans;
                _mm_store_ss(&ans, sum128);
            transpose[i][6]=ans;

    }
    for(int i=0;i<8;i++)
    {
        row8= _mm256_loadu_ps (&matrix[i][0]);
        row9=_mm256_mul_ps(row7,row8);
        __m128 sum128 = _mm256_extractf128_ps(row9, 0);
                sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(row9, 1));
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float ans;
                _mm_store_ss(&ans, sum128);
            transpose[i][7]=ans;

    }

    auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        std::cout << ms_double.count() << "ms\n";
        
    for(int i=0;i<8;i++)
    {
        for(int j=0;j<8;j++)
        {
            cout<<transpose[i][j]<<"\t";
        }
        cout<<endl;
    }

}