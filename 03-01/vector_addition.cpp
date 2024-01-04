#include <iostream>
#include <immintrin.h>
#include <chrono>

int main()
{
    int input=8;
    int size=8;
    // int input,size;

    // std::cin>>input;

    if(input%4!=0){
        size=input+(4-(input%4));
    }
    else{
        size=input;
    }

    double vector1[size]={1,2,3,4,5,6,7,8};
    double vector2[size]={1,2,3,4,5,6,7,8};
    double ans[size]={0};

    // double vector1[size]={0};
    // double vector2[size]={0};
    // double ans[size]={0};


    // std::cout<<"Vector1: \n";
    // for(int i=0;i<input;i++){
    //     std::cin>>vector1[i];
    // }

    // std::cout<<"Vector2: \n";
    // for(int i=0;i<input;i++){
    //     std::cin>>vector2[i];
    // }
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
    std::cout << ms_double.count() << "ms\n";

    for(int i=0;i<input;i++){
        std::cout<<ans[i]<<"\t";
    }
    std::cout<<"\n";
}