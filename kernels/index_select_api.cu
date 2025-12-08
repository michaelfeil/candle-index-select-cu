#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>
#include <type_traits>

// ---------------------------
// Scalar row-wise kernel
// ---------------------------
// x:   [rows, cols]
// idx: [index_count]
// out: [index_count, cols]
template <typename T>
__global__ void index_select_rows_scalar(
    const T* __restrict__ x,
    const uint32_t* __restrict__ idx,
    T* __restrict__ out,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    // Each block processes multiple output rows (indices) in a grid-stride loop.
    uint32_t j = blockIdx.x;
    uint32_t j_stride = gridDim.x;

    for (; j < index_count; j += j_stride) {
        uint32_t r = idx[j];
        // Optional bounds check (debug):
        // if (r >= rows) return;

        const T* __restrict__ src_row = x   + (size_t)r * cols;
        T*       __restrict__ dst_row = out + (size_t)j * cols;

        // Threads in a block walk columns contiguously -> coalesced
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
            dst_row[c] = src_row[c];
        }
    }
}

// ---------------------------
// Vectorized row-wise kernel
// ---------------------------
// VecType helper
template <typename T, int Vec> struct VecType {};
template <> struct VecType<float, 4> { using Type = float4; };

template <typename T, int Vec>
__global__ void index_select_rows_vec(
    const T* __restrict__ x,
    const uint32_t* __restrict__ idx,
    T* __restrict__ out,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    using VecT = typename VecType<T, Vec>::Type;

    const uint32_t cols_vec = cols / Vec;
    uint32_t j = blockIdx.x;
    uint32_t j_stride = gridDim.x;

    const VecT* __restrict__ x_vec   = reinterpret_cast<const VecT*>(x);
    VecT*       __restrict__ out_vec = reinterpret_cast<VecT*>(out);

    for (; j < index_count; j += j_stride) {
        uint32_t r = idx[j];
        // Optional bounds check:
        // if (r >= rows) return;

        const VecT* __restrict__ src_row =
            x_vec   + (size_t)r * cols_vec;
        VecT*       __restrict__ dst_row =
            out_vec + (size_t)j * cols_vec;

        for (uint32_t c_vec = threadIdx.x; c_vec < cols_vec; c_vec += blockDim.x) {
            dst_row[c_vec] = src_row[c_vec];
        }
    }
}

// ---------------------------
// Launch config helper
// ---------------------------
static inline void compute_launch_config_rows(
    uint32_t index_count,
    uint32_t cols,
    int multi_processor_count,
    dim3& grid,
    dim3& block
) {
    // Heuristic: many rows -> many blocks, but cap at mp * factor
    const int threads = 256;
    // 4–8x SMs is usually good for memory-bound kernels
    int max_blocks = multi_processor_count > 0 ? multi_processor_count * 8 : 1024;

    // If index_count is small, don't launch more blocks than rows.
    int blocks = (int)index_count;
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    grid  = dim3(blocks, 1, 1);
    block = dim3(threads, 1, 1);
}

// ---------------------------
// Type-specific launcher
// ---------------------------
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

    dim3 grid, block;
    compute_launch_config_rows(index_count, cols, multi_processor_count, grid, block);

    const T* x_t   = static_cast<const T*>(x);
    T*       dst_t = static_cast<T*>(dst);

    // Vectorized path for float, cols multiple of 4, and 16-byte alignment.
    if constexpr (std::is_same<T, float>::value) {
        const uintptr_t x_addr   = reinterpret_cast<uintptr_t>(x_t);
        const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_t);
        const bool aligned =
            ((x_addr  % alignof(float4)) == 0) &&
            ((dst_addr % alignof(float4)) == 0);

        if (aligned && (cols % 4 == 0)) {
            index_select_rows_vec<float, 4><<<grid, block>>>(
                x_t, idx, dst_t, rows, cols, index_count
            );
#ifdef DEBUG
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("index_select_rows_vec<float,4> launch failed: %s\n",
                       cudaGetErrorString(err));
            }
#endif
            return;
        }
    }

    // Scalar fallback (any T)
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

// ---------------------------
// C entry point
// ---------------------------
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
    // dtype_code: 0 = f16, 1 = f32
    switch (dtype_code) {
    case 0:
        // __half: scalar path only for now; can add __half2 vectorization later.
        launch_index_select<__half>(
            x, indices, dst, rows, cols, index_count, multi_processor_count);
        break;
    case 1:
        launch_index_select<float>(
            x, indices, dst, rows, cols, index_count, multi_processor_count);
        break;
    default:
        // Unsupported dtype; Rust side should have error'd out already.
        break;
    }
}