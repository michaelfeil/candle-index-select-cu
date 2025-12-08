#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>
#include <type_traits>

//
// Scalar kernel: works for all shapes, any T.
// layout: x [rows, cols], out [index_count, cols], indices [index_count]
//
template <typename T>
__global__ void index_select_kernel_scalar(
    const T* __restrict__ x,          // [rows, cols]
    const uint32_t* __restrict__ idx, // [index_count]
    T* __restrict__ out,              // [index_count, cols]
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    const uint64_t total = static_cast<uint64_t>(index_count) * cols;

    for (uint64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<uint64_t>(blockDim.x) * gridDim.x) {

        // decode flat index -> (j, c)
        uint32_t j = static_cast<uint32_t>(linear / cols);   // index position
        uint32_t c = static_cast<uint32_t>(linear % cols);   // column

        uint32_t r = idx[j];

        // NOTE: we assume indices are valid. If you want Torch-like error checking,
        // you can enable this in a debug build and report via a status buffer:
        // if (r >= rows) return;
        uint64_t src_off = static_cast<uint64_t>(r) * cols + c;
        out[linear] = x[src_off];
    }
}

//
// Vector type helper: maps (T, Vec) -> underlying vectorized type
//
template <typename T, int Vec> struct VecType;

template <> struct VecType<float, 2> { using Type = float2; };
template <> struct VecType<float, 4> { using Type = float4; };

// For __half you *can* add half2 / custom vector structs if you want,
// but we keep scalar for half for now.
//

//
// Vectorized kernel: treats the last dim as groups of Vec elements.
// Only used when cols is divisible by Vec and pointers are properly aligned.
//
template <typename T, int Vec>
__global__ void index_select_kernel_vec(
    const T* __restrict__ x,
    const uint32_t* __restrict__ idx,
    T* __restrict__ out,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    using VecT = typename VecType<T, Vec>::Type;

    const uint32_t cols_vec = cols / Vec;
    const uint64_t total_vec = static_cast<uint64_t>(index_count) * cols_vec;

    const VecT* __restrict__ x_vec   = reinterpret_cast<const VecT*>(x);
    VecT* __restrict__ out_vec       = reinterpret_cast<VecT*>(out);

    for (uint64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_vec;
         linear += static_cast<uint64_t>(blockDim.x) * gridDim.x) {

        uint32_t j = static_cast<uint32_t>(linear / cols_vec);
        uint32_t c_vec = static_cast<uint32_t>(linear % cols_vec);

        uint32_t r = idx[j];
        // if (r >= rows) return;  // optional debug check

        uint64_t src_off_vec = static_cast<uint64_t>(r) * cols_vec + c_vec;
        VecT v = x_vec[src_off_vec];
        out_vec[linear] = v;
    }
}

//
// Small helper to pick a grid/block size.
// This is very similar in spirit to what PyTorch does for many elementwise/gather ops:
// - 1D kernel over total elements
// - block size 256
// - grid clamped to multi_processor_count * factor
//
static inline void compute_launch_config(
    uint64_t total_elems,
    int multi_processor_count,
    dim3& grid,
    dim3& block
) {
    if (total_elems == 0) {
        grid = dim3(1, 1, 1);
        block = dim3(1, 1, 1);
        return;
    }

    int blk = 256;
    // Factor of SMs to launch; 8 is a reasonable default for modern GPUs (incl. H100)
    int max_blocks = multi_processor_count > 0 ? multi_processor_count * 8 : 1024;

    uint64_t grid_raw = (total_elems + blk - 1) / blk;
    if (grid_raw > static_cast<uint64_t>(max_blocks)) {
        grid_raw = static_cast<uint64_t>(max_blocks);
    }
    if (grid_raw < 1) grid_raw = 1;

    grid = dim3(static_cast<unsigned int>(grid_raw), 1, 1);
    block = dim3(blk, 1, 1);
}

//
// Type-specific launcher.
// This is the main entry from the C interface.
// We choose between vectorized (float4) and scalar paths here.
//
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

    const uint64_t total = static_cast<uint64_t>(index_count) * cols;
    dim3 grid, block;
    compute_launch_config(total, multi_processor_count, grid, block);

    const T* x_t   = static_cast<const T*>(x);
    T* dst_t       = static_cast<T*>(dst);

    // --- Vectorized path for float (float4) ---
    if constexpr (std::is_same<T, float>::value) {
        // Only use vec path when the last dimension is a multiple of 4
        // and both pointers are 16-byte aligned.
        const uintptr_t x_addr   = reinterpret_cast<uintptr_t>(x_t);
        const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_t);
        const bool aligned =
            ((x_addr % alignof(float4)) == 0) &&
            ((dst_addr % alignof(float4)) == 0);

        if (aligned && (cols % 4 == 0)) {
            const uint32_t cols_vec = cols / 4;
            const uint64_t total_vec =
                static_cast<uint64_t>(index_count) * cols_vec;

            dim3 grid_vec, block_vec;
            compute_launch_config(total_vec, multi_processor_count, grid_vec, block_vec);

            index_select_kernel_vec<float, 4><<<grid_vec, block_vec>>>(
                x_t, idx, dst_t, rows, cols, index_count
            );

#ifdef DEBUG
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("index_select_kernel_vec<float,4> launch failed: %s\n",
                       cudaGetErrorString(err));
            }
#endif
            return;
        }
        // Else: fall through to scalar kernel below.
    }

    // --- Scalar fallback path (any T, any shape) ---

    index_select_kernel_scalar<T><<<grid, block>>>(
        x_t,
        idx,
        dst_t,
        rows,
        cols,
        index_count
    );

#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("index_select_kernel_scalar launch failed: %s\n",
               cudaGetErrorString(err));
    }
#endif
}

//
// C API: matches what you already had in your Rust FFI.
//
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
        // For __half we use scalar kernel for now. You can add half2 / custom vec types
        // using the same pattern as float if you really want to squeeze more out.
        launch_index_select<__half>(
            x, indices, dst, rows, cols, index_count, multi_processor_count);
        break;
    case 1:
        launch_index_select<float>(
            x, indices, dst, rows, cols, index_count, multi_processor_count);
        break;
    default:
        // Unsupported dtype; Rust side will have already errored.
        break;
    }
}