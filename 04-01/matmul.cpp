#include<iostream>
#include<chrono>
using namespace std;
int main()
{
    float a[8][8] = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                     {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                     {17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
                     {25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0},
                     {33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0},
                     {41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0},
                     {49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0},
                     {57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0}};
    float b[8][8] = {{1, 2, 3, 4, 5, 6, 7, 8},
                     {1, 2, 3, 4, 5, 6, 7, 8},
                     {1, 2, 3, 4, 5, 6, 7, 8},
                     {1, 2, 3, 4, 5, 6, 7, 8},
                     {1, 2, 3, 4, 5, 6, 7, 8},
                     {1, 2, 3, 4, 5, 6, 7, 8},
                     {1, 2, 3, 4, 5, 6, 7, 8},
                     {1, 2, 3, 4, 5, 6, 7, 8}};
    float ans[8][8] = {0};
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    for(int i=0;i<8;i++)
    {
        for(int j=0;j<8;j++)
        {
            for(int k=0;k<8;k++)
            {
                ans[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    for(int i=0;i<8;i++)
    {
        for(int j=0;j<8;j++)
        {
            cout<<ans[i][j]<<"\t";
        }
        cout<<"\n";
    }
}