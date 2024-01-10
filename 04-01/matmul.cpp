#include<iostream>
#include<chrono>
using namespace std;
int main()
{
    double r1[16][16];
    double ans[16][16]={0};
    double x=1;
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
            cout<<r1[i][j]<<"\t";
        }
        cout<<"\n";
    }
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    for(int i=0;i<16;i++)
    {
        for(int j=0;j<16;j++)
        {
            for(int k=0;k<16;k++)
            {
                ans[i][j]+=r1[i][k]*r1[k][j];
            }
        }
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
        for(int i=0;i<16;i++)
    {
        for(int j=0;j<16;j++)
        {
            cout<<ans[i][j]<<"\t";
        }
        cout<<"\n";
    }
}