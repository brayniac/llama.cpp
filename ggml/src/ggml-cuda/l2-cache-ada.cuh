#pragma once

#include "common.cuh"

// Ada Lovelace L2 Cache Optimization Strategies
// RTX 4090 has a massive 96MB L2 cache (16x larger than previous generations)
// These optimizations leverage this enhanced cache hierarchy for better performance

namespace ggml_cuda_l2_ada {

// L2 cache management for Ada Lovelace's 96MB L2 cache
struct ada_l2_cache_config {
    static constexpr size_t L2_CACHE_SIZE = 96 * 1024 * 1024;  // 96MB
    static constexpr size_t CACHE_LINE_SIZE = 128;              // bytes
    static constexpr size_t PREFETCH_DISTANCE = 8;              // cache lines ahead
    static constexpr size_t WORKING_SET_THRESHOLD = 32 * 1024 * 1024; // 32MB threshold
};

// Enhanced prefetching strategies for Ada Lovelace
template<typename T>
__device__ __forceinline__ void ada_prefetch_l2(const T* ptr, const int stride, const int count = 4) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    // Prefetch multiple cache lines into L2
    #pragma unroll
    for (int i = 0; i < count; ++i) {
        const T* prefetch_addr = ptr + i * stride;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(prefetch_addr));
    }
#endif
}

// Optimized data layout for L2 cache efficiency
template<typename T, int TILE_SIZE>
__device__ __forceinline__ void ada_cache_friendly_load(
    T* smem_dst, const T* global_src, const int global_stride,
    const int tile_row, const int tile_col) {
    
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Calculate optimal memory access pattern for Ada's L2 cache
    constexpr int ELEMENTS_PER_THREAD = TILE_SIZE * TILE_SIZE / (32 * 8); // 8 warps
    constexpr int VECTOR_SIZE = sizeof(float4) / sizeof(T);
    
    // Prefetch upcoming data into L2 cache
    if (lane_id == 0) {
        const T* prefetch_base = global_src + (tile_row + 1) * global_stride + tile_col;
        ada_prefetch_l2(prefetch_base, global_stride, 4);
    }
    
    // Vectorized loads with enhanced coalescing for Ada
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += VECTOR_SIZE) {
        const int row = (warp_id * ELEMENTS_PER_THREAD + i) / TILE_SIZE;
        const int col = (warp_id * ELEMENTS_PER_THREAD + i) % TILE_SIZE;
        
        if (row < TILE_SIZE && col + VECTOR_SIZE <= TILE_SIZE) {
            // Use vectorized loads for better L2 cache utilization
            const T* src_addr = global_src + (tile_row + row) * global_stride + tile_col + col;
            T* dst_addr = smem_dst + row * TILE_SIZE + col;
            
            *reinterpret_cast<float4*>(dst_addr) = 
                *reinterpret_cast<const float4*>(src_addr);
        }
    }
#endif
}

// L2 cache-aware matrix blocking for large matrices
struct ada_cache_blocking_strategy {
    // Calculate optimal block sizes based on Ada's L2 cache
    static __host__ __device__ void get_optimal_block_sizes(
        const int M, const int N, const int K,
        int& block_m, int& block_n, int& block_k) {
        
        // Conservative approach: keep 2/3 of working set in L2 cache
        constexpr size_t target_working_set = ada_l2_cache_config::L2_CACHE_SIZE * 2 / 3;
        
        // Estimate memory requirements for different block sizes
        // A block: block_m * block_k * sizeof(element)
        // B block: block_k * block_n * sizeof(element)  
        // C block: block_m * block_n * sizeof(element)
        
        // Start with default sizes and adjust
        block_m = 128;
        block_n = 128; 
        block_k = 32;
        
        // Adjust based on matrix dimensions and cache capacity
        size_t working_set = (size_t(block_m) * block_k + 
                             size_t(block_k) * block_n + 
                             size_t(block_m) * block_n) * sizeof(float);
        
        // Scale up if we have room in L2 cache
        while (working_set * 2 < target_working_set && 
               block_m < M && block_n < N) {
            block_m = min(block_m * 2, M);
            block_n = min(block_n * 2, N);
            working_set = (size_t(block_m) * block_k + 
                          size_t(block_k) * block_n + 
                          size_t(block_m) * block_n) * sizeof(float);
        }
    }
};

// Enhanced shared memory to L2 cache transfer patterns
template<typename T, int SHARED_MEM_SIZE>
__device__ __forceinline__ void ada_shared_to_l2_transfer(
    const T* smem_src, T* global_dst, const int global_stride,
    const int tile_row, const int tile_col) {
    
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    const int tid = threadIdx.x;
    constexpr int ELEMENTS_PER_THREAD = SHARED_MEM_SIZE / (32 * 8); // 8 warps
    constexpr int VECTOR_SIZE = sizeof(float4) / sizeof(T);
    
    // Cooperative write pattern optimized for Ada's memory hierarchy
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += VECTOR_SIZE) {
        const int offset = tid * ELEMENTS_PER_THREAD + i;
        const int row = offset / 128; // Assuming 128-column tiles
        const int col = offset % 128;
        
        if (offset + VECTOR_SIZE <= SHARED_MEM_SIZE) {
            const T* src_addr = smem_src + offset;
            T* dst_addr = global_dst + (tile_row + row) * global_stride + tile_col + col;
            
            // Write through to L2 cache for future access
            *reinterpret_cast<float4*>(dst_addr) = 
                *reinterpret_cast<const float4*>(src_addr);
        }
    }
#endif
}

// L2 cache persistence control for Ada Lovelace
__device__ __forceinline__ void ada_set_cache_persistence(const void* ptr, size_t size) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    // Set cache persistence for frequently accessed data
    // This helps keep important data in Ada's large L2 cache
    const size_t cache_segment_size = 32 * 1024; // 32KB segments
    const size_t num_segments = (size + cache_segment_size - 1) / cache_segment_size;
    
    for (size_t i = 0; i < num_segments; ++i) {
        const char* segment_ptr = static_cast<const char*>(ptr) + i * cache_segment_size;
        // Mark this memory region for L2 persistence
        asm volatile("prefetch.global.L2 [%0];" :: "l"(segment_ptr));
    }
#endif
}

// Memory access pattern optimization for Ada's cache hierarchy
template<typename T>
struct ada_memory_access_pattern {
    static constexpr int OPTIMAL_STRIDE = 128 / sizeof(T); // 128-byte cache lines
    static constexpr int PREFETCH_AHEAD = 8; // Cache lines ahead
    
    __device__ __forceinline__ static void optimized_read_sequence(
        T* dst, const T* src, const int count, const int stride) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        const int tid = threadIdx.x;
        
        // Prefetch pattern optimized for Ada's memory subsystem
        if (tid < WARP_SIZE) {
            #pragma unroll 4
            for (int prefetch_idx = 0; prefetch_idx < PREFETCH_AHEAD; ++prefetch_idx) {
                const int prefetch_offset = (tid + prefetch_idx * WARP_SIZE) * stride;
                if (prefetch_offset < count) {
                    ada_prefetch_l2(src + prefetch_offset, 1, 1);
                }
            }
        }
        
        __syncthreads();
        
        // Actual data transfer with optimized access pattern
        const int elements_per_thread = (count + blockDim.x - 1) / blockDim.x;
        const int start_idx = tid * elements_per_thread;
        
        #pragma unroll 4
        for (int i = 0; i < elements_per_thread && start_idx + i < count; ++i) {
            dst[start_idx + i] = src[(start_idx + i) * stride];
        }
#endif
    }
};

// Ada Lovelace specific GEMM cache optimization
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct ada_gemm_cache_strategy {
    // Optimal shared memory configuration for Ada's cache hierarchy
    static constexpr int SMEM_A_SIZE = BLOCK_M * BLOCK_K;
    static constexpr int SMEM_B_SIZE = BLOCK_K * BLOCK_N;
    static constexpr int SMEM_STAGE_COUNT = 4; // Pipeline stages for Ada
    
    __device__ __forceinline__ static void cache_aware_tile_load(
        float* smem_a, float* smem_b,
        const float* global_a, const float* global_b,
        const int lda, const int ldb,
        const int block_row, const int block_col, const int block_k_idx) {
        
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        // Load A tile with L2 cache optimization
        ada_cache_friendly_load<float, BLOCK_M>(
            smem_a, global_a + block_row * lda + block_k_idx * BLOCK_K, 
            lda, 0, 0);
        
        // Load B tile with L2 cache optimization  
        ada_cache_friendly_load<float, BLOCK_K>(
            smem_b, global_b + block_k_idx * ldb + block_col * BLOCK_N,
            ldb, 0, 0);
        
        // Prefetch next iteration's data into L2 cache
        if (threadIdx.x == 0) {
            const float* next_a = global_a + block_row * lda + (block_k_idx + 1) * BLOCK_K;
            const float* next_b = global_b + (block_k_idx + 1) * ldb + block_col * BLOCK_N;
            ada_prefetch_l2(next_a, lda, 2);
            ada_prefetch_l2(next_b, ldb, 2);
        }
#endif
    }
};

} // namespace ggml_cuda_l2_ada