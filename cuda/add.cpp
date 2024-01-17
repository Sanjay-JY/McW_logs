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
    std::vector<int> h_a(N, 1);
    std::vector<int> h_b(N, 2);
    std::vector<int> h_c(N);

    auto start_time = std::chrono::high_resolution_clock::now();

    addVectors(h_a, h_b, h_c);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout <<duration.count() << "\n";

    return 0;
}
