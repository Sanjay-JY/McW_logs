#include <iostream>
#include <immintrin.h>
#include <chrono>

int main()
{
    int size=65535;

    double vector1[size];
    double vector2[size];
    double ans[size]={0};
    for(int i=0;i<size;i++)
    {
        vector1[i]=i;
        vector2[i]=i;
    }

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    __m256d a,b,sum;
    for(int i=0;i<size;i+=4){
        a = _mm256_load_pd(&vector1[i]);
        b = _mm256_load_pd(&vector2[i]);
        sum = _mm256_add_pd(a,b);
        _mm256_store_pd(&ans[i],sum);
    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "\n";
}