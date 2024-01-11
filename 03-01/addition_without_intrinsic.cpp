#include<iostream>
#include<chrono>
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
    for(int i=0;i<size;i++){
        ans[i]=vector1[i]+vector2[i];
    }
    
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count()<<"\n";
}