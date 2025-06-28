#pragma once

#include "common.cuh"

// Ada Lovelace (RTX 4090) memory bandwidth optimization
// Optimizes for RTX 4090's enhanced memory subsystem:
// - 1008 GB/s memory bandwidth (24GB GDDR6X)
// - Enhanced memory controllers
// - 128-byte cache lines
// - Improved memory latency hiding

namespace ggml_cuda_ada_memory {

// Ada Lovelace memory subsystem characteristics
struct ada_memory_config {
    static constexpr size_t MEMORY_BANDWIDTH = 1008ULL * 1024 * 1024 * 1024; // 1008 GB/s
    static constexpr size_t CACHE_LINE_SIZE = 128; // bytes
    static constexpr int MEMORY_CONTROLLERS = 12;  // RTX 4090 memory controllers
    static constexpr size_t OPTIMAL_TRANSFER_SIZE = 512; // bytes per thread
    static constexpr int MEMORY_LATENCY_CYCLES = 400; // approximate
};

// Enhanced vectorized memory operations for Ada Lovelace
template<typename T>
struct ada_vectorized_ops {
    // Optimal vector sizes for Ada's memory subsystem
    using vector_t = typename std::conditional_t<
        sizeof(T) == 2, uint4,  // half -> uint4 (16 bytes)
        typename std::conditional_t<
            sizeof(T) == 4, float4, // float -> float4 (16 bytes)  
            uint4                   // fallback
        >
    >;
    
    static constexpr int elements_per_vector = sizeof(vector_t) / sizeof(T);
    
    // Optimized coalesced load for Ada Lovelace
    __device__ __forceinline__ static void coalesced_load(
        T* dst, const T* src, const int count, const int stride = 1) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int total_threads = blockDim.x * gridDim.x;
        
        // Calculate optimal load pattern for Ada's memory controllers
        const int vectors_per_thread = (count + total_threads * elements_per_vector - 1) / 
                                      (total_threads * elements_per_vector);
        
        #pragma unroll 4
        for (int v = 0; v < vectors_per_thread; ++v) {
            const int vector_idx = tid + v * total_threads;
            const int element_idx = vector_idx * elements_per_vector;
            
            if (element_idx + elements_per_vector <= count) {
                // Use vectorized loads for maximum bandwidth utilization
                const vector_t* src_vec = reinterpret_cast<const vector_t*>(src + element_idx * stride);
                vector_t* dst_vec = reinterpret_cast<vector_t*>(dst + element_idx);
                
                *dst_vec = *src_vec;
            } else if (element_idx < count) {
                // Handle remaining elements
                #pragma unroll
                for (int i = 0; i < elements_per_vector && element_idx + i < count; ++i) {
                    dst[element_idx + i] = src[(element_idx + i) * stride];
                }
            }
        }
#endif
    }
    
    // Optimized coalesced store for Ada Lovelace
    __device__ __forceinline__ static void coalesced_store(
        T* dst, const T* src, const int count, const int stride = 1) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int total_threads = blockDim.x * gridDim.x;
        
        const int vectors_per_thread = (count + total_threads * elements_per_vector - 1) / 
                                      (total_threads * elements_per_vector);
        
        #pragma unroll 4
        for (int v = 0; v < vectors_per_thread; ++v) {
            const int vector_idx = tid + v * total_threads;
            const int element_idx = vector_idx * elements_per_vector;
            
            if (element_idx + elements_per_vector <= count) {
                const vector_t* src_vec = reinterpret_cast<const vector_t*>(src + element_idx);
                vector_t* dst_vec = reinterpret_cast<vector_t*>(dst + element_idx * stride);
                
                *dst_vec = *src_vec;
            } else if (element_idx < count) {
                #pragma unroll
                for (int i = 0; i < elements_per_vector && element_idx + i < count; ++i) {
                    dst[(element_idx + i) * stride] = src[element_idx + i];
                }
            }
        }
#endif
    }
};

// Advanced memory access patterns for Ada Lovelace
template<typename T, int TILE_SIZE>
struct ada_memory_patterns {
    // Optimized strided access for matrix operations
    __device__ __forceinline__ static void load_strided_optimized(
        T* smem_dst, const T* global_src, const int global_stride,
        const int rows, const int cols) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        const int tid = threadIdx.x;
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        
        // Use Ada's enhanced memory hierarchy
        constexpr int VECTORS_PER_WARP = TILE_SIZE / ada_vectorized_ops<T>::elements_per_vector;
        
        #pragma unroll
        for (int row = warp_id; row < rows; row += blockDim.x / WARP_SIZE) {
            #pragma unroll
            for (int vec = 0; vec < VECTORS_PER_WARP; vec += WARP_SIZE) {
                const int col_vec = vec + lane_id;
                const int col_base = col_vec * ada_vectorized_ops<T>::elements_per_vector;
                
                if (col_base < cols) {
                    const auto* src_vec = reinterpret_cast<const typename ada_vectorized_ops<T>::vector_t*>(
                        global_src + row * global_stride + col_base);
                    auto* dst_vec = reinterpret_cast<typename ada_vectorized_ops<T>::vector_t*>(
                        smem_dst + row * TILE_SIZE + col_base);
                    
                    *dst_vec = *src_vec;
                }
            }
        }
#endif
    }
    
    // Optimized transpose with coalescing for Ada
    __device__ __forceinline__ static void transpose_coalesced(
        T* dst, const T* src, const int rows, const int cols) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        const int tid = threadIdx.x;
        
        // Use tile-based transpose optimized for Ada's cache hierarchy
        constexpr int TRANSPOSE_TILE = 32;
        
        for (int tile_row = 0; tile_row < rows; tile_row += TRANSPOSE_TILE) {
            for (int tile_col = 0; tile_col < cols; tile_col += TRANSPOSE_TILE) {
                // Process tiles to maximize cache reuse
                const int row = tile_row + (tid / TRANSPOSE_TILE);
                const int col = tile_col + (tid % TRANSPOSE_TILE);
                
                if (row < rows && col < cols) {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
        }
#endif
    }
};

// Asynchronous memory operations for Ada Lovelace
template<typename T>
struct ada_async_memory {
    // Double buffering with enhanced pipeline depth
    __device__ __forceinline__ static void async_load_pipeline(
        T* stage0, T* stage1, const T* global_src, const int size, const int stage) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        const int tid = threadIdx.x;
        
        // Ada supports deeper pipelines
        if (stage % 2 == 0) {
            // Load into stage0
            ada_vectorized_ops<T>::coalesced_load(stage0 + tid, global_src + tid, size);
        } else {
            // Load into stage1
            ada_vectorized_ops<T>::coalesced_load(stage1 + tid, global_src + tid, size);
        }
        
        // Enhanced memory fence for Ada's memory subsystem
        __threadfence_block();
#endif
    }
    
    // Cooperative memory operations
    __device__ __forceinline__ static void cooperative_load(
        T* smem_dst, const T* global_src, const int total_size) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        const int tid = threadIdx.x;
        const int block_size = blockDim.x;
        
        // Distribute load across all threads for maximum bandwidth
        const int elements_per_thread = (total_size + block_size - 1) / block_size;
        const int start_idx = tid * elements_per_thread;
        const int end_idx = min(start_idx + elements_per_thread, total_size);
        
        // Use vectorized loads where possible
        ada_vectorized_ops<T>::coalesced_load(
            smem_dst + start_idx, global_src + start_idx, end_idx - start_idx);
#endif
    }
};

// Memory bandwidth utilization monitor for Ada Lovelace
struct ada_bandwidth_monitor {
    __device__ __forceinline__ static float calculate_efficiency(
        const size_t bytes_transferred, const int cycles_elapsed) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        // Calculate achieved vs theoretical bandwidth
        const float theoretical_bytes_per_cycle = 
            float(ada_memory_config::MEMORY_BANDWIDTH) / 1.5e9f; // Assuming 1.5 GHz
        const float achieved_bytes_per_cycle = 
            float(bytes_transferred) / float(cycles_elapsed);
        
        return achieved_bytes_per_cycle / theoretical_bytes_per_cycle;
#else
        return 0.0f;
#endif
    }
};

// Cache-aware memory layout optimization
template<typename T, int ALIGNMENT = 128>
struct ada_memory_layout {
    // Optimized for Ada's cache line size and memory controllers
    static constexpr int ELEMENTS_PER_CACHE_LINE = ALIGNMENT / sizeof(T);
    
    __host__ __device__ static int get_padded_stride(const int width) {
        // Ensure cache line alignment and avoid bank conflicts
        const int aligned_width = ((width + ELEMENTS_PER_CACHE_LINE - 1) / 
                                  ELEMENTS_PER_CACHE_LINE) * ELEMENTS_PER_CACHE_LINE;
        return aligned_width;
    }
    
    __host__ __device__ static size_t get_padded_size(const int height, const int width) {
        return size_t(height) * get_padded_stride(width) * sizeof(T);
    }
};

// Optimized GEMM memory access for Ada Lovelace
template<int TILE_M, int TILE_N, int TILE_K>
struct ada_gemm_memory {
    using half_vector = ada_vectorized_ops<half>::vector_t;
    using float_vector = ada_vectorized_ops<float>::vector_t;
    
    __device__ __forceinline__ static void load_A_tile(
        half* smem_A, const half* global_A, const int lda, 
        const int tile_row, const int tile_k) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        ada_memory_patterns<half, TILE_M>::load_strided_optimized(
            smem_A, global_A + tile_row * lda + tile_k * TILE_K, lda, TILE_M, TILE_K);
#endif
    }
    
    __device__ __forceinline__ static void load_B_tile(
        half* smem_B, const half* global_B, const int ldb,
        const int tile_k, const int tile_col) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        ada_memory_patterns<half, TILE_K>::load_strided_optimized(
            smem_B, global_B + tile_k * ldb + tile_col * TILE_N, ldb, TILE_K, TILE_N);
#endif
    }
    
    __device__ __forceinline__ static void store_C_tile(
        float* global_C, const float* smem_C, const int ldc,
        const int tile_row, const int tile_col) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        ada_memory_patterns<float, TILE_M>::transpose_coalesced(
            global_C + tile_row * ldc + tile_col * TILE_N, smem_C, TILE_M, TILE_N);
#endif
    }
};

} // namespace ggml_cuda_ada_memory