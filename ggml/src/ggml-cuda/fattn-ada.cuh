#pragma once

#include "common.cuh"

// Forward declarations instead of including fattn-mma-f16.cuh to avoid duplicate definitions
struct ggml_backend_cuda_context;
struct ggml_tensor;

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Ada Lovelace (RTX 4090) optimized Flash Attention configurations
// These configurations leverage RTX 4090's specific capabilities:
// - 544 4th-gen Tensor Cores
// - 96MB L2 cache (16x larger than previous gen)
// - 128 threads per SM (vs 64 on Ampere)
// - Enhanced memory subsystem with 1008 GB/s bandwidth
// - 164KB shared memory per block

namespace ggml_cuda_ada {

// Enhanced configurations for Ada Lovelace architecture
template <int DKQ, int DV>
struct fattn_ada_config;

// RTX 4090 optimized configuration for 64x64 attention
template <>
struct fattn_ada_config<64, 64> {
    static constexpr int  nbatch_fa      = 128;  // 2x increase for better occupancy
    static constexpr int  nwarps_max     = 8;    // Utilize 128 threads/SM
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 4;    // More pipeline stages for Ada's memory subsystem
    static constexpr bool use_l2_prefetch = true; // Leverage 96MB L2 cache
    static constexpr bool use_enhanced_coalescing = true;

    static int get_nbatch_K2_host(const int cc, const int ncols) {
        GGML_UNUSED(ncols);
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return 64; // 2x increase for Ada Lovelace
        }
        return 32; // Fallback for older architectures
    }

    static constexpr __device__ int get_nbatch_K2_device(int ncols) {
        GGML_UNUSED(ncols);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        return 64;
#else
        return 32;
#endif
    }

    static int get_nbatch_V2_host(const int cc, const int ncols) {
        GGML_UNUSED(ncols);
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return 64;
        }
        return 32;
    }

    static constexpr __device__ int get_nbatch_V2_device(int ncols) {
        GGML_UNUSED(ncols);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        return 64;
#else
        return 32;
#endif
    }

    static int get_nbatch_combine_host(const int cc, const int ncols) {
        GGML_UNUSED(ncols);
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return 64;
        }
        return 32;
    }

    static constexpr __device__ int get_nbatch_combine_device(int ncols) {
        GGML_UNUSED(ncols);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        return 64;
#else
        return 32;
#endif
    }
};

// RTX 4090 optimized configuration for 128x128 attention (most common)
template <>
struct fattn_ada_config<128, 128> {
    static constexpr int  nbatch_fa      = 256;  // 4x increase for high occupancy
    static constexpr int  nwarps_max     = 8;    // Maximum warps for Ada Lovelace
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 4;    // Enhanced pipelining
    static constexpr bool use_l2_prefetch = true;
    static constexpr bool use_enhanced_coalescing = true;
    static constexpr bool use_tensor_core_bf16 = true; // Leverage BF16 tensor cores

    static int get_nbatch_K2_host(const int cc, const int ncols) {
        GGML_UNUSED(ncols);
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return 128; // Maximum batch size for Ada
        }
        return 64;
    }

    static constexpr __device__ int get_nbatch_K2_device(int ncols) {
        GGML_UNUSED(ncols);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        return 128;
#else
        return 64;
#endif
    }

    static int get_nbatch_V2_host(const int cc, const int ncols) {
        GGML_UNUSED(ncols);
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return 128;
        }
        return 64;
    }

    static constexpr __device__ int get_nbatch_V2_device(int ncols) {
        GGML_UNUSED(ncols);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        return 128;
#else
        return 64;
#endif
    }

    static int get_nbatch_combine_host(const int cc, const int ncols) {
        GGML_UNUSED(ncols);
        if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return 128;
        }
        return 64;
    }

    static constexpr __device__ int get_nbatch_combine_device(int ncols) {
        GGML_UNUSED(ncols);
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
        return 128;
#else
        return 64;
#endif
    }
};

// Enhanced memory coalescing for Ada Lovelace
template<typename T, int ELEMENTS_PER_THREAD = 8>
__device__ __forceinline__ void ada_coalesced_load(T* dst, const T* src, int n) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    // Optimized for Ada's cache line size and memory subsystem
    constexpr int VECTOR_SIZE = sizeof(float4) / sizeof(T);
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += VECTOR_SIZE) {
        if (threadIdx.x * ELEMENTS_PER_THREAD + i + VECTOR_SIZE <= n) {
            // Use vectorized loads for better memory bandwidth utilization
            *reinterpret_cast<float4*>(&dst[i]) = 
                *reinterpret_cast<const float4*>(&src[threadIdx.x * ELEMENTS_PER_THREAD + i]);
        }
    }
#else
    // Fallback for older architectures
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (threadIdx.x * ELEMENTS_PER_THREAD + i < n) {
            dst[i] = src[threadIdx.x * ELEMENTS_PER_THREAD + i];
        }
    }
#endif
}

// L2 cache prefetching for Ada Lovelace's 96MB L2 cache
template<typename T>
__device__ __forceinline__ void ada_l2_prefetch(const T* ptr) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    // Prefetch data into L2 cache
    __builtin_assume_aligned(ptr, 16);
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
#endif
}

// Enhanced shared memory configuration for Ada Lovelace
template<int TILE_SIZE_M, int TILE_SIZE_K, int TILE_SIZE_N>
struct ada_shared_memory_config {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
    // Use larger tiles for Ada Lovelace's enhanced shared memory
    static constexpr int enhanced_tile_m = TILE_SIZE_M * 2;
    static constexpr int enhanced_tile_k = TILE_SIZE_K * 2;
    static constexpr int enhanced_tile_n = TILE_SIZE_N * 2;
    static constexpr int pipeline_stages = 4;
#else
    static constexpr int enhanced_tile_m = TILE_SIZE_M;
    static constexpr int enhanced_tile_k = TILE_SIZE_K;
    static constexpr int enhanced_tile_n = TILE_SIZE_N;
    static constexpr int pipeline_stages = 2;
#endif
};

// Ada Lovelace specific case functions that use our optimized configurations
template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case_ada(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // Use Ada-optimized configuration instead of standard config
    typedef fattn_ada_config<DKQ, DV> c;

    printf("Ada Lovelace Flash Attention: DKQ=%d, DV=%d, batch_fa=%d, nwarps=%d, nstages=%d\n", 
           DKQ, DV, c::nbatch_fa, c::nwarps_max, c::nstages_target);

    // IMPORTANT: The standard implementation uses fattn_mma_f16_config, not our Ada config
    // So our optimized parameters aren't actually being used in the kernel
    // This is why you're not seeing performance improvements
    
    printf("WARNING: Ada config defined but not used by kernel - no performance gain expected\n");
    printf("TODO: Modify kernel to actually use Ada parameters for real speedup\n");
    
    // Call the standard implementation (which ignores our Ada config)
    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, ncols1, ncols2>(ctx, dst);
}

// Ada Lovelace optimized dispatch functions
template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_ada(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case_ada<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case_ada<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case_ada<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    // Use larger batch sizes for Ada Lovelace
    ggml_cuda_flash_attn_ext_mma_f16_case_ada<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2_ada(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_ada<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_ada<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_ada<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_ada<DKQ, DV, 1>(ctx, dst);
}

} // namespace ggml_cuda_ada