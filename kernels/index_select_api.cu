#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>

template <typename T>
__global__ void index_select_kernel(
    const T* __restrict__ x,      // [rows, cols]
    const uint32_t* __restrict__ idx, // [index_count]
    T* __restrict__ out,          // [index_count, cols]
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count
) {
    const uint64_t total = static_cast<uint64_t>(index_count) * cols;
    for (uint64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        uint32_t j = static_cast<uint32_t>(linear / cols);   // index position
        uint32_t c = static_cast<uint32_t>(linear % cols);   // column
        uint32_t r = idx[j];
        // Optional: bounds check (can be `#ifdef DEBUG`).
        // if (r >= rows) continue;
        uint64_t src_off = static_cast<uint64_t>(r) * cols + c;
        out[linear] = x[src_off];
    }
}

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

    int block = 256;
    int max_blocks = multi_processor_count > 0 ? multi_processor_count * 8 : 1024;
    int grid = static_cast<int>((total + block - 1) / block);
    if (grid > max_blocks) grid = max_blocks;
    if (grid < 1) grid = 1;

    index_select_kernel<T><<<grid, block>>>(
        static_cast<const T*>(x),
        idx,
        static_cast<T*>(dst),
        rows,
        cols,
        index_count
    );

#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("index_select_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
#endif
}

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
        launch_index_select<__half>(
            x, indices, dst, rows, cols, index_count, multi_processor_count);
        break;
    case 1:
        launch_index_select<float>(
            x, indices, dst, rows, cols, index_count, multi_processor_count);
        break;
    default:
        // Unsupported dtype; nothing to do (Rust side will have already errored)
        break;
    }
}