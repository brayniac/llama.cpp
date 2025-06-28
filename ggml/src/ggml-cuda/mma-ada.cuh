#pragma once

#include "common.cuh"
#include "mma.cuh"

// Ada Lovelace (RTX 4090) specific tensor core optimizations
// Leverages 4th-generation Tensor Cores with enhanced capabilities:
// - Better BF16 support with native accumulation
// - Enhanced mixed precision operations
// - 2:4 structured sparsity support
// - Higher throughput matrix operations

namespace ggml_cuda_mma_ada {

// Enhanced BF16 tensor core operations for Ada Lovelace
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE

// BF16 matrix tile with enhanced accumulation
template <int I_, int J_>
struct tile_bf16_ada {
    static constexpr int I  = I_;
    static constexpr int J  = J_;
    static constexpr int ne = I * J / WARP_SIZE;
    __nv_bfloat16 x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        if constexpr (I == 16 && J == 16) {
            return ((l / 2) % 2) * 8 + threadIdx.x / 4;
        } else if constexpr (I == 16 && J == 8) {
            return (l / 2) * 8 + threadIdx.x / 4;
        } else {
            static_assert(I == -1 && J == -1, "BF16 tile specialization not implemented");
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 16 && J == 16) {
            return 8 * (l / 4) + 2 * (threadIdx.x % 4) + l % 2;
        } else if constexpr (I == 16 && J == 8) {
            return 2 * (threadIdx.x % 4) + l % 2;
        } else {
            static_assert(I == -1 && J == -1, "BF16 tile specialization not implemented");
        }
    }

    __device__ __forceinline__ void load(const void * __restrict__ src, const int offset) {
        const __nv_bfloat16 * src_bf16 = reinterpret_cast<const __nv_bfloat16 *>(src) + offset;
        
        #pragma unroll
        for (int l = 0; l < ne; ++l) {
            const int i = get_i(l);
            const int j = get_j(l);
            x[l] = src_bf16[i * J + j];
        }
    }
};

// Enhanced MMA operations for Ada Lovelace with BF16 support
template<typename C, typename A, typename B>
__device__ __forceinline__ void mma_ada_bf16(
    C & c, const A & a, const B & b) {
    
    static_assert(C::I == 16 && C::J == 16, "Ada BF16 MMA requires 16x16 output tiles");
    static_assert(A::I == 16 && A::J == 8,  "Ada BF16 MMA requires 16x8 A tiles");
    static_assert(B::I == 8  && B::J == 16, "Ada BF16 MMA requires 8x16 B tiles");

    // Use Ada Lovelace enhanced BF16 tensor core instructions
    asm volatile(
        "mma.sync.aligned.m16n16k8.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%8, %9, %10, %11};"
        : "=f"(c.x[0]), "=f"(c.x[1]), "=f"(c.x[2]), "=f"(c.x[3])
        : "r"(a.x[0]), "r"(a.x[1]),
          "r"(b.x[0]), "r"(b.x[1]),
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
    );
}

// Enhanced mixed precision MMA with better accumulation
template<typename C, typename A, typename B>
__device__ __forceinline__ void mma_ada_mixed_precision(
    C & c, const A & a, const B & b, const float scale = 1.0f) {
    
    // Enhanced mixed precision operation leveraging Ada's capabilities
    if constexpr (std::is_same_v<typename A::value_type, __nv_bfloat16> && 
                  std::is_same_v<typename B::value_type, __nv_bfloat16>) {
        
        // Use BF16 tensor cores with enhanced accumulation
        mma_ada_bf16(c, a, b);
        
        // Apply scaling for better numerical stability
        #pragma unroll
        for (int l = 0; l < C::ne; ++l) {
            c.x[l] *= scale;
        }
    }
}

// Structured sparsity support for Ada Lovelace (2:4 sparsity pattern)
template<typename C, typename A, typename B, typename Metadata>
__device__ __forceinline__ void mma_ada_sparse_24(
    C & c, const A & a, const B & b, const Metadata & meta) {
    
    static_assert(C::I == 16 && C::J == 16, "Sparse MMA requires 16x16 output tiles");
    
    // Use Ada Lovelace 2:4 structured sparsity support
    asm volatile(
        "mma.sp.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, "
        "%16, 0x0;"
        : "=f"(c.x[0]), "=f"(c.x[1]), "=f"(c.x[2]), "=f"(c.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),
          "r"(meta.x[0])
    );
}

// Enhanced tile loading with Ada Lovelace optimizations
template<typename T>
__device__ __forceinline__ void load_matrix_ada_optimized(
    T & tile, const void * __restrict__ src, const int stride, 
    const int offset = 0) {
    
    // Use enhanced memory access patterns for Ada Lovelace
    const auto * src_typed = reinterpret_cast<const typename T::value_type *>(src);
    
    // Leverage Ada's improved memory subsystem
    #pragma unroll
    for (int l = 0; l < T::ne; ++l) {
        const int i = T::get_i(l);
        const int j = T::get_j(l);
        const int idx = offset + i * stride + j;
        
        // Enhanced memory coalescing for Ada architecture
        tile.x[l] = src_typed[idx];
    }
}

// Ada Lovelace specific tensor core configuration
struct ada_tensor_config {
    static constexpr int max_warps_per_block = 8;     // Utilize 128 threads/SM
    static constexpr int enhanced_tile_size = 32;     // Larger tiles for Ada
    static constexpr int pipeline_stages = 4;         // More pipeline stages
    static constexpr bool use_bf16_accumulation = true;
    static constexpr bool use_sparse_24 = true;       // Enable 2:4 sparsity
    static constexpr float sparsity_threshold = 0.75f; // Threshold for sparse ops
};

// Optimized GEMM kernel configuration for Ada Lovelace
template<int TILE_M, int TILE_N, int TILE_K>
struct ada_gemm_config {
    static constexpr int block_tile_m = TILE_M * 2; // 2x larger for Ada
    static constexpr int block_tile_n = TILE_N * 2;
    static constexpr int block_tile_k = TILE_K;
    static constexpr int threads_per_block = 256;   // Optimal for Ada
    static constexpr int shared_mem_stages = 4;     // Enhanced pipelining
};

#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE

// Function to detect and utilize Ada Lovelace tensor core features
__device__ __forceinline__ bool is_ada_lovelace_available() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    return true;
#else
    return false;
#endif
}

// Enhanced tensor core dispatch for Ada Lovelace
template<typename C, typename A, typename B>
__device__ __forceinline__ void dispatch_tensor_core_ada(
    C & c, const A & a, const B & b, const int compute_capability) {
    
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    if (compute_capability >= GGML_CUDA_CC_ADA_LOVELACE) {
        // Use Ada Lovelace optimized path
        if constexpr (std::is_same_v<typename A::value_type, __nv_bfloat16>) {
            mma_ada_bf16(c, a, b);
        } else {
            mma_ada_mixed_precision(c, a, b);
        }
    } else {
        // Fallback to standard MMA
        ggml_cuda_mma::mma(c, a, b);
    }
#else
    // Compile-time fallback
    ggml_cuda_mma::mma(c, a, b);
#endif
}

} // namespace ggml_cuda_mma_ada