#include <iostream>
#include <vector>
#include <chrono>

const int N = 1000000;

void addVectors(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Host vectors
    std::vector<int> h_a(N, 1);
    std::vector<int> h_b(N, 2);
    std::vector<int> h_c(N);

    // Measure the time for CPU computation
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform vector addition on the CPU
    addVectors(h_a, h_b, h_c);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout <<duration.count() << "\n";

    // Print the result or further process the result as needed
    // ...

    return 0;
}
