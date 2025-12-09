#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>
#include <type_traits>

// ============================================================================
// Scalar row-wise kernel (generic, works for any T / shape)
// x:   [rows, cols]
// idx: [index_count]
// out: [index_count, cols]
// ============================================================================
template <typename T>
__global__ void index_select_rows_scalar(
    const T* __restrict__ x,
    const uint32_t* __restrict__ idx,
    T* __restrict__ out,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    // Each block works on one or more output rows in a grid-stride loop.
    uint32_t j = blockIdx.x;
    const uint32_t j_stride = gridDim.x;

    for (; j < index_count; j += j_stride) {
        const uint32_t r = idx[j];
        // Assumes indices are valid: r < rows

        const T* __restrict__ src_row = x   + (size_t)r * cols;
        T*       __restrict__ dst_row = out + (size_t)j * cols;

        // Threads in the block walk columns contiguously => coalesced
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
            dst_row[c] = src_row[c];
        }
    }
}

// ============================================================================
// Warp-specialized FP16 path (__half2)
// Each warp handles one row at a time, copying using __half2 vector loads/stores.
// This is the main fast path for large FP16 2D/flattened ND tensors.
// ============================================================================
template <int WARPS_PER_BLOCK>
__global__ void index_select_rows_half2_warp(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ idx,
    __half* __restrict__ out,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    constexpr int WARP_SIZE = 32;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    if (threadIdx.x >= THREADS_PER_BLOCK) return;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    const uint32_t rows_per_block = WARPS_PER_BLOCK;
    const uint32_t cols_vec = cols / 2; // __half2

    const __half2* __restrict__ x_vec   = reinterpret_cast<const __half2*>(x);
    __half2*       __restrict__ out_vec = reinterpret_cast<__half2*>(out);

    // Each warp starts at its own row j
    uint32_t j = blockIdx.x * rows_per_block + warp_id;

    // Grid-stride over rows
    while (j < index_count) {
        const uint32_t r = idx[j];
        // Assumes r < rows

        const __half2* __restrict__ src_row =
            x_vec   + (size_t)r * cols_vec;
        __half2*       __restrict__ dst_row =
            out_vec + (size_t)j * cols_vec;

        // Vectorized copy: each lane handles a strided subset of the row,
        // unrolling by 2 chunks per loop to reduce loop overhead.
        for (uint32_t c_vec = lane_id; c_vec < cols_vec; c_vec += WARP_SIZE * 2) {
            uint32_t i0 = c_vec;
            uint32_t i1 = c_vec + WARP_SIZE;
            if (i0 < cols_vec) dst_row[i0] = src_row[i0];
            if (i1 < cols_vec) dst_row[i1] = src_row[i1];
        }

        j += gridDim.x * rows_per_block;
    }
}

// ============================================================================
// Warp-specialized FP32 path (float4)
// Same idea as FP16, but vector width = 4 floats.
// ============================================================================
template <int WARPS_PER_BLOCK>
__global__ void index_select_rows_float4_warp(
    const float* __restrict__ x,
    const uint32_t* __restrict__ idx,
    float* __restrict__ out,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    constexpr int WARP_SIZE = 32;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    if (threadIdx.x >= THREADS_PER_BLOCK) return;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    const uint32_t rows_per_block = WARPS_PER_BLOCK;
    const uint32_t cols_vec = cols / 4; // float4

    const float4* __restrict__ x_vec   = reinterpret_cast<const float4*>(x);
    float4*       __restrict__ out_vec = reinterpret_cast<float4*>(out);

    uint32_t j = blockIdx.x * rows_per_block + warp_id;

    while (j < index_count) {
        const uint32_t r = idx[j];

        const float4* __restrict__ src_row =
            x_vec   + (size_t)r * cols_vec;
        float4*       __restrict__ dst_row =
            out_vec + (size_t)j * cols_vec;

        // Same pattern: 2 chunks per iteration per lane.
        for (uint32_t c_vec = lane_id; c_vec < cols_vec; c_vec += WARP_SIZE * 2) {
            uint32_t i0 = c_vec;
            uint32_t i1 = c_vec + WARP_SIZE;
            if (i0 < cols_vec) dst_row[i0] = src_row[i0];
            if (i1 < cols_vec) dst_row[i1] = src_row[i1];
        }

        j += gridDim.x * rows_per_block;
    }
}

// ============================================================================
// Launch config helpers
// ============================================================================

// Scalar row-wise launcher config: one block per row (up to mp * factor)
static inline void compute_launch_config_scalar(
    uint32_t index_count,
    int multi_processor_count,
    dim3& grid,
    dim3& block
) {
    const int threads = 256;
    int max_blocks = (multi_processor_count > 0) ? (multi_processor_count * 8) : 1024;

    int blocks = (int)index_count;
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    grid  = dim3((unsigned int)blocks, 1, 1);
    block = dim3((unsigned int)threads, 1, 1);
}

// Warp-specialized config: each block owns WARPS_PER_BLOCK rows at a time
static inline void compute_launch_config_warp(
    uint32_t index_count,
    int warps_per_block,
    int multi_processor_count,
    dim3& grid,
    dim3& block
) {
    const int threads = warps_per_block * 32;
    const uint64_t rows_per_block = (uint64_t)warps_per_block;

    int max_blocks = (multi_processor_count > 0) ? (multi_processor_count * 8) : 1024;

    uint64_t grid_x = (index_count + rows_per_block - 1) / rows_per_block;
    if (grid_x > (uint64_t)max_blocks) grid_x = max_blocks;
    if (grid_x < 1) grid_x = 1;

    grid  = dim3((unsigned int)grid_x, 1, 1);
    block = dim3((unsigned int)threads, 1, 1);
}

// ============================================================================
// Type-specific launcher: chooses best kernel for T and shape
// ============================================================================

template <typename T>
static void launch_index_select(
    const void* x,
    const uint32_t* idx,
    void* dst,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count,
    int multi_processor_count
) {
    if (index_count == 0 || rows == 0 || cols == 0) {
        return;
    }

    const T* x_t   = static_cast<const T*>(x);
    T*       dst_t = static_cast<T*>(dst);

    // ================== FP16 fast path (half2 warp kernel) ==================
    if constexpr (std::is_same<T, __half>::value) {
        // Good for large-ish tensors where cols is even (so we can use half2).
        if ((cols % 2u) == 0 && cols >= 64 && index_count >= 64) {
            constexpr int WARPS_PER_BLOCK = 4; // 4 warps -> 128 threads
            dim3 grid, block;
            compute_launch_config_warp(index_count, WARPS_PER_BLOCK,
                                       multi_processor_count, grid, block);

            index_select_rows_half2_warp<WARPS_PER_BLOCK><<<grid, block>>>(
                x_t, idx, dst_t, rows, cols, index_count
            );
#ifdef DEBUG
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("index_select_rows_half2_warp launch failed: %s\n",
                       cudaGetErrorString(err));
            }
#endif
            return;
        }
    }

    // ================== FP32 fast path (float4 warp kernel) ==================
    if constexpr (std::is_same<T, float>::value) {
        if ((cols % 4u) == 0 && cols >= 64 && index_count >= 64) {
            constexpr int WARPS_PER_BLOCK = 4;
            dim3 grid, block;
            compute_launch_config_warp(index_count, WARPS_PER_BLOCK,
                                       multi_processor_count, grid, block);

            index_select_rows_float4_warp<WARPS_PER_BLOCK><<<grid, block>>>(
                x_t, idx, dst_t, rows, cols, index_count
            );
#ifdef DEBUG
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("index_select_rows_float4_warp launch failed: %s\n",
                       cudaGetErrorString(err));
            }
#endif
            return;
        }
    }

    // ================== Scalar fallback (any T, any shape) ==================
    {
        dim3 grid, block;
        compute_launch_config_scalar(index_count, multi_processor_count, grid, block);

        index_select_rows_scalar<T><<<grid, block>>>(
            x_t, idx, dst_t, rows, cols, index_count
        );
#ifdef DEBUG
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("index_select_rows_scalar launch failed: %s\n",
                   cudaGetErrorString(err));
        }
#endif
    }
}

// ============================================================================
// C API entry point (FFI compatible)
// dtype_code: 0 = f16, 1 = f32
// ============================================================================

extern "C" void run_index_select(
    const void* x,
    const uint32_t* indices,
    void* dst,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count,
    int multi_processor_count,
    uint32_t dtype_code
) {
    switch (dtype_code) {
    case 0: // __half
        launch_index_select<__half>(
            x, indices, dst, rows, cols, index_count, multi_processor_count
        );
        break;
    case 1: // float
        launch_index_select<float>(
            x, indices, dst, rows, cols, index_count, multi_processor_count
        );
        break;
    default:
        // Unsupported dtype; Rust side should have already rejected it.
        break;
    }
}