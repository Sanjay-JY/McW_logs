#include<iostream>
#include<chrono>
using namespace std;
int main()
{
    double r1[4][4]={{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    double ans[4][4]={0};
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            {
                ans[i][j]+=r1[i][k]*r1[k][j];
            }
        }
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            cout<<ans[i][j]<<"\t";
        }
        cout<<"\n";
    }
}