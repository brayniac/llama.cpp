#pragma once

#include "common.cuh"

// Ada Lovelace benchmark header
namespace ggml_cuda_ada_benchmark {

// Benchmark function declarations
float benchmark_matmul_ada(int M, int N, int K, int num_iterations = 10);
float benchmark_attention_ada(int batch_size, int seq_len, int head_dim, int num_iterations = 10);
void run_ada_benchmark_suite();

// Performance comparison utilities
struct benchmark_result {
    float performance;
    float efficiency;
    const char* description;
};

benchmark_result compare_ada_vs_baseline(int M, int N, int K);

} // namespace ggml_cuda_ada_benchmark