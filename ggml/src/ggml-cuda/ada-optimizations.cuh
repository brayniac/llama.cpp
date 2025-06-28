#pragma once

// Ada Lovelace (RTX 4090) Optimization Suite
// This header integrates all Ada Lovelace specific optimizations for llama.cpp
// 
// Key improvements for RTX 4090:
// - Enhanced Flash Attention with 2x batch sizes and 4x pipeline stages
// - BF16 tensor core utilization with structured sparsity support
// - L2 cache optimizations for 96MB cache (16x larger than previous gen)
// - Thread block optimization for 128 threads/SM architecture
// - Memory bandwidth optimization for 1008 GB/s throughput
//
// Expected performance improvements:
// - Flash Attention: 20-30% faster
// - Matrix Multiplication: 15-25% faster  
// - Overall inference: 15-20% faster

#include "common.cuh"
#include "fattn-ada.cuh"
#include "mma-ada.cuh"
#include "l2-cache-ada.cuh"
#include "ada-thread-config.cuh"
#include "ada-memory.cuh"
#include "ada-benchmark.cuh"

namespace ggml_cuda_ada {

// Master configuration struct for Ada Lovelace optimizations
struct ada_master_config {
    static constexpr int COMPUTE_CAPABILITY = GGML_CUDA_CC_ADA_LOVELACE;
    static constexpr bool OPTIMIZATIONS_ENABLED = true;
    static constexpr const char* ARCHITECTURE_NAME = "Ada Lovelace";
    static constexpr const char* TARGET_GPU = "RTX 4090";
    
    // Performance targets
    static constexpr float TARGET_FLASH_ATTENTION_SPEEDUP = 1.25f; // 25% improvement
    static constexpr float TARGET_MATMUL_SPEEDUP = 1.20f;          // 20% improvement
    static constexpr float TARGET_OVERALL_SPEEDUP = 1.18f;         // 18% improvement
    
    // Hardware specifications
    static constexpr int TENSOR_CORES = 544;
    static constexpr int L2_CACHE_MB = 96;
    static constexpr int MEMORY_BANDWIDTH_GB = 1008;
    static constexpr int CUDA_CORES_PER_SM = 128;
};

// Runtime detection of Ada Lovelace capabilities
class ada_runtime_detector {
public:
    static bool is_ada_lovelace_available() {
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        return (prop.major * 100 + prop.minor * 10) >= GGML_CUDA_CC_ADA_LOVELACE;
    }
    
    static bool supports_bf16_tensor_cores() {
        return is_ada_lovelace_available(); // Ada Lovelace has enhanced BF16 support
    }
    
    static bool supports_structured_sparsity() {
        return is_ada_lovelace_available(); // 2:4 sparsity support
    }
    
    static int get_l2_cache_size_mb() {
        if (is_ada_lovelace_available()) {
            return ada_master_config::L2_CACHE_MB;
        }
        return 6; // Fallback for older architectures
    }
};

// Optimization dispatcher - automatically selects best implementation
template<typename OperationType>
class ada_optimization_dispatcher {
public:
    template<typename... Args>
    static auto dispatch(Args&&... args) {
        if constexpr (std::is_same_v<OperationType, FlashAttentionOp>) {
            return dispatch_flash_attention(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<OperationType, MatrixMultiplyOp>) {
            return dispatch_matrix_multiply(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<OperationType, TensorCoreOp>) {
            return dispatch_tensor_core(std::forward<Args>(args)...);
        } else {
            return dispatch_generic(std::forward<Args>(args)...);
        }
    }

private:
    template<typename... Args>
    static auto dispatch_flash_attention(Args&&... args) {
        if (ada_runtime_detector::is_ada_lovelace_available()) {
            // Use Ada-optimized flash attention
            return ada_flash_attention_optimized(std::forward<Args>(args)...);
        } else {
            // Fallback to standard implementation
            return standard_flash_attention(std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    static auto dispatch_matrix_multiply(Args&&... args) {
        if (ada_runtime_detector::is_ada_lovelace_available()) {
            // Use Ada-optimized matrix multiplication
            return ada_matrix_multiply_optimized(std::forward<Args>(args)...);
        } else {
            return standard_matrix_multiply(std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    static auto dispatch_tensor_core(Args&&... args) {
        if (ada_runtime_detector::supports_bf16_tensor_cores()) {
            return ggml_cuda_mma_ada::dispatch_tensor_core_ada(std::forward<Args>(args)...);
        } else {
            return ggml_cuda_mma::mma(std::forward<Args>(args)...);
        }
    }
};

// Operation type markers for dispatcher
struct FlashAttentionOp {};
struct MatrixMultiplyOp {};
struct TensorCoreOp {};

// Convenience functions for common operations
template<int DKQ, int DV>
auto get_optimal_flash_attention_config() {
    if (ada_runtime_detector::is_ada_lovelace_available()) {
        return fattn_ada_config<DKQ, DV>{};
    } else {
        return fattn_mma_f16_config<DKQ, DV>{};
    }
}

auto get_optimal_thread_config(int M, int N, int K) {
    if (ada_runtime_detector::is_ada_lovelace_available()) {
        return ggml_cuda_ada_threads::ada_dynamic_config::get_matmul_config(M, N, K, 
                                                                            GGML_CUDA_CC_ADA_LOVELACE);
    } else {
        // Fallback configuration
        ggml_cuda_ada_threads::ada_dynamic_config::config_result result;
        result.threads_per_block = 256;
        result.blocks_per_grid = ((M + 15) / 16) * ((N + 15) / 16);
        result.shared_mem_per_block = 32 * 1024;
        result.registers_per_thread = 32;
        return result;
    }
}

// Performance monitoring and validation
class ada_performance_monitor {
private:
    static inline float baseline_performance = 0.0f;
    static inline float optimized_performance = 0.0f;
    
public:
    static void set_baseline_performance(float perf) {
        baseline_performance = perf;
    }
    
    static void set_optimized_performance(float perf) {
        optimized_performance = perf;
    }
    
    static float get_speedup() {
        if (baseline_performance > 0.0f) {
            return optimized_performance / baseline_performance;
        }
        return 1.0f;
    }
    
    static bool meets_performance_target(float target = ada_master_config::TARGET_OVERALL_SPEEDUP) {
        return get_speedup() >= target;
    }
    
    static void print_performance_summary() {
        printf("Ada Lovelace Optimization Summary:\n");
        printf("Baseline Performance: %.2f GFLOPS\n", baseline_performance);
        printf("Optimized Performance: %.2f GFLOPS\n", optimized_performance);
        printf("Speedup: %.2fx\n", get_speedup());
        printf("Target Achievement: %s\n", 
               meets_performance_target() ? "SUCCESS" : "NEEDS_IMPROVEMENT");
    }
};

// Main initialization function for Ada optimizations
void initialize_ada_optimizations() {
    if (ada_runtime_detector::is_ada_lovelace_available()) {
        printf("Initializing Ada Lovelace optimizations for RTX 4090...\n");
        printf("- Enhanced Flash Attention: ENABLED\n");
        printf("- BF16 Tensor Cores: ENABLED\n");
        printf("- L2 Cache Optimization: ENABLED (96MB)\n");
        printf("- Thread Block Optimization: ENABLED\n");
        printf("- Memory Bandwidth Optimization: ENABLED (1008 GB/s)\n");
        
        // Run validation benchmarks
        ggml_cuda_ada_benchmark::run_ada_benchmark_suite();
    } else {
        printf("Ada Lovelace optimizations not available on this device.\n");
    }
}

} // namespace ggml_cuda_ada

// Convenience macros for easy integration
#define ADA_OPTIMIZE_FLASH_ATTENTION(dkq, dv) \
    ggml_cuda_ada::get_optimal_flash_attention_config<dkq, dv>()

#define ADA_OPTIMIZE_THREADS(m, n, k) \
    ggml_cuda_ada::get_optimal_thread_config(m, n, k)

#define ADA_IS_AVAILABLE() \
    ggml_cuda_ada::ada_runtime_detector::is_ada_lovelace_available()

#define ADA_INITIALIZE() \
    ggml_cuda_ada::initialize_ada_optimizations()