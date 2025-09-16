#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

static inline int idx(int i, int j, int N) { return i * N + j; }

static void fill_rand(std::vector<float>& M, unsigned seed=123) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : M) x = dist(gen);
}

static void fill_identity(std::vector<float>& M, int N) {
    std::fill(M.begin(), M.end(), 0.f);
    for (int i = 0; i < N; ++i) M[idx(i,i,N)] = 1.f;
}

static float max_abs_diff(const std::vector<float>& A, const std::vector<float>& B) {
    float m = 0.f;
    for (size_t k = 0; k < A.size(); ++k) m = std::max(m, std::fabs(A[k] - B[k]));
    return m;
}

// Algoritmo naïve: 3 bucles (orden i-k-j)
static void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A[idx(i,k,N)];
            for (int j = 0; j < N; ++j) {
                C[idx(i,j,N)] += a * B[idx(k,j,N)];
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = 64;                          // por defecto
    if (argc > 1) N = std::atoi(argv[1]);

    std::vector<float> A(N*N), B(N*N), C(N*N, 0.f);

    // --- Verificación rápida con identidad: A * I = A ---
    fill_rand(A, 123);
    fill_identity(B, N);
    std::fill(C.begin(), C.end(), 0.f);
    matmul_naive(A.data(), B.data(), C.data(), N);
    float diff_id = max_abs_diff(A, C);
    std::cout << "N=" << N << "  check(A*I≈A) max_abs_diff=" << diff_id << "\n";

    // --- Benchmark simple con B aleatoria ---
    fill_rand(B, 456);
    std::fill(C.begin(), C.end(), 0.f);
    // warmup
    matmul_naive(A.data(), B.data(), C.data(), N);
    // medir
    std::fill(C.begin(), C.end(), 0.f);
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_naive(A.data(), B.data(), C.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "time_ms=" << ms << "\n";
    return 0;
}
