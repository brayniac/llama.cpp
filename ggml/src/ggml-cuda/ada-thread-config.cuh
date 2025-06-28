#pragma once

#include "common.cuh"

// Ada Lovelace (RTX 4090) optimized thread block configurations
// Leverages RTX 4090's specific SM architecture:
// - 128 CUDA cores per SM (vs 64 on Ampere)
// - 65,536 32-bit registers per SM  
// - 164KB shared memory per block
// - Enhanced occupancy characteristics

namespace ggml_cuda_ada_threads {

// Ada Lovelace SM characteristics
struct ada_sm_config {
    static constexpr int CUDA_CORES_PER_SM = 128;
    static constexpr int MAX_THREADS_PER_SM = 2048;
    static constexpr int MAX_BLOCKS_PER_SM = 16;
    static constexpr int REGISTERS_PER_SM = 65536;
    static constexpr int SHARED_MEM_PER_SM = 164 * 1024; // 164KB
    static constexpr int MAX_WARPS_PER_SM = 64;
    static constexpr int WARP_SIZE = 32;
};

// Optimal thread block configurations for different kernel types
struct ada_thread_config {
    // Flash Attention optimal configuration
    struct flash_attention {
        static constexpr int threads_per_block = 256;  // 8 warps, optimal for Ada
        static constexpr int warps_per_block = 8;
        static constexpr int blocks_per_sm = 8;        // High occupancy
        static constexpr int shared_mem_per_block = 48 * 1024; // 48KB
    };
    
    // Matrix Multiplication optimal configuration  
    struct matrix_mul {
        static constexpr int threads_per_block = 512;  // 16 warps for heavy compute
        static constexpr int warps_per_block = 16;
        static constexpr int blocks_per_sm = 4;        // Balanced occupancy
        static constexpr int shared_mem_per_block = 96 * 1024; // 96KB
    };
    
    // Quantized Matrix Multiplication
    struct quantized_mmq {
        static constexpr int threads_per_block = 256;  // Good balance
        static constexpr int warps_per_block = 8;
        static constexpr int blocks_per_sm = 6;        // High throughput
        static constexpr int shared_mem_per_block = 32 * 1024; // 32KB
    };
    
    // Element-wise operations
    struct elementwise {
        static constexpr int threads_per_block = 1024; // Maximum threads
        static constexpr int warps_per_block = 32;
        static constexpr int blocks_per_sm = 2;        // Memory bound
        static constexpr int shared_mem_per_block = 8 * 1024; // 8KB
    };
};

// Dynamic thread configuration based on problem size and Ada characteristics
class ada_dynamic_config {
public:
    struct config_result {
        int threads_per_block;
        int blocks_per_grid;
        int shared_mem_per_block;
        int registers_per_thread;
    };
    
    // Calculate optimal configuration for matrix multiplication
    static __host__ config_result get_matmul_config(
        const int M, const int N, const int K, const int cc) {
        
        config_result result;
        
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            // Ada Lovelace specific optimization
            if (M * N >= 1024 * 1024) {
                // Large matrices: maximize throughput
                result.threads_per_block = 512;
                result.shared_mem_per_block = 96 * 1024;
                result.registers_per_thread = 64;
            } else if (M * N >= 64 * 64) {
                // Medium matrices: balance occupancy and cache
                result.threads_per_block = 256;
                result.shared_mem_per_block = 48 * 1024;
                result.registers_per_thread = 48;
            } else {
                // Small matrices: maximize occupancy
                result.threads_per_block = 128;
                result.shared_mem_per_block = 24 * 1024;
                result.registers_per_thread = 32;
            }
            
            // Calculate grid size
            const int tile_size = result.threads_per_block == 512 ? 32 : 16;
            const int blocks_m = (M + tile_size - 1) / tile_size;
            const int blocks_n = (N + tile_size - 1) / tile_size;
            result.blocks_per_grid = blocks_m * blocks_n;
        } else {
            // Fallback for older architectures
            result.threads_per_block = 256;
            result.blocks_per_grid = ((M + 15) / 16) * ((N + 15) / 16);
            result.shared_mem_per_block = 32 * 1024;
            result.registers_per_thread = 32;
        }
        
        return result;
    }
    
    // Calculate optimal configuration for attention
    static __host__ config_result get_attention_config(
        const int batch_size, const int seq_len, const int head_dim, const int cc) {
        
        config_result result;
        
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            // Ada Lovelace attention optimization
            if (seq_len >= 2048) {
                // Long sequences: optimize for memory bandwidth
                result.threads_per_block = 256;
                result.shared_mem_per_block = 64 * 1024;
                result.registers_per_thread = 56;
            } else if (seq_len >= 512) {
                // Medium sequences: balance compute and memory
                result.threads_per_block = 256;
                result.shared_mem_per_block = 48 * 1024;
                result.registers_per_thread = 48;
            } else {
                // Short sequences: maximize occupancy
                result.threads_per_block = 128;
                result.shared_mem_per_block = 32 * 1024;
                result.registers_per_thread = 40;
            }
            
            // Grid configuration for attention
            const int blocks_per_head = (seq_len + 127) / 128;
            result.blocks_per_grid = batch_size * blocks_per_head;
        } else {
            // Fallback configuration
            result.threads_per_block = 128;
            result.blocks_per_grid = batch_size * ((seq_len + 63) / 64);
            result.shared_mem_per_block = 32 * 1024;
            result.registers_per_thread = 32;
        }
        
        return result;
    }
};

// Ada Lovelace occupancy calculator
class ada_occupancy_calculator {
public:
    struct occupancy_result {
        int theoretical_occupancy;
        int achieved_occupancy;
        int limiting_factor; // 0: threads, 1: registers, 2: shared_mem
        float efficiency;
    };
    
    static __host__ occupancy_result calculate_occupancy(
        const int threads_per_block, const int registers_per_thread,
        const int shared_mem_per_block) {
        
        occupancy_result result;
        
        // Calculate theoretical limits
        const int max_blocks_by_threads = ada_sm_config::MAX_THREADS_PER_SM / threads_per_block;
        const int max_blocks_by_registers = ada_sm_config::REGISTERS_PER_SM / 
            (threads_per_block * registers_per_thread);
        const int max_blocks_by_shared_mem = ada_sm_config::SHARED_MEM_PER_SM / 
            shared_mem_per_block;
        
        // Find limiting factor
        result.achieved_occupancy = min({max_blocks_by_threads, 
                                        max_blocks_by_registers,
                                        max_blocks_by_shared_mem,
                                        ada_sm_config::MAX_BLOCKS_PER_SM});
        
        result.theoretical_occupancy = ada_sm_config::MAX_THREADS_PER_SM / threads_per_block;
        
        // Determine limiting factor
        if (result.achieved_occupancy == max_blocks_by_threads) {
            result.limiting_factor = 0; // threads
        } else if (result.achieved_occupancy == max_blocks_by_registers) {
            result.limiting_factor = 1; // registers  
        } else {
            result.limiting_factor = 2; // shared memory
        }
        
        result.efficiency = float(result.achieved_occupancy * threads_per_block) / 
                           float(ada_sm_config::MAX_THREADS_PER_SM);
        
        return result;
    }
};

// Enhanced MMQ configuration for Ada Lovelace
template<int X_TILE_SIZE, int Y_TILE_SIZE>
struct ada_mmq_config {
    // Optimized for Ada's enhanced SM characteristics
    static constexpr int ADA_MMQ_NWARPS = 8;           // 2x increase for Ada
    static constexpr int ADA_MMQ_MAX_BATCH_SIZE = 128; // 2x increase for Ada
    static constexpr int ADA_MMQ_ITER_K = 512;         // 2x increase for better L2 utilization
    
    static __host__ __device__ int get_threads_per_block() {
        return ADA_MMQ_NWARPS * WARP_SIZE; // 256 threads
    }
    
    static __host__ __device__ int get_shared_memory_size() {
        // Calculate based on tile sizes and pipeline stages
        constexpr int pipeline_stages = 4; // Ada can handle more stages
        return (X_TILE_SIZE + Y_TILE_SIZE) * pipeline_stages * sizeof(float);
    }
    
    static __host__ __device__ int get_registers_per_thread() {
        // Estimate based on tile size and accumulation requirements
        return 32 + (X_TILE_SIZE * Y_TILE_SIZE) / (ADA_MMQ_NWARPS * WARP_SIZE);
    }
};

// Adaptive configuration selector
template<typename KernelType>
__host__ auto get_ada_config(const int M, const int N, const int K, const int cc) {
    if constexpr (std::is_same_v<KernelType, ada_thread_config::flash_attention>) {
        return ada_dynamic_config::get_attention_config(1, M, N, cc);
    } else if constexpr (std::is_same_v<KernelType, ada_thread_config::matrix_mul>) {
        return ada_dynamic_config::get_matmul_config(M, N, K, cc);
    } else {
        // Default configuration
        ada_dynamic_config::config_result result;
        result.threads_per_block = 256;
        result.blocks_per_grid = (M * N + 255) / 256;
        result.shared_mem_per_block = 32 * 1024;
        result.registers_per_thread = 32;
        return result;
    }
}

} // namespace ggml_cuda_ada_threads