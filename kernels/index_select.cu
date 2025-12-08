#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>

template <typename T>
__global__ void index_select_kernel(
    const T* __restrict__ input,
    const uint32_t* __restrict__ indices,
    T* __restrict__ output,
    size_t rows,
    size_t cols,
    size_t index_count)
{
    const size_t out_numel = (size_t)index_count * (size_t)cols;
    const size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

    for (size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < out_numel;
         linear += stride) {
        const size_t row_out = linear / cols;   // which index in indices
        const size_t col = linear % cols;

        const uint32_t row_in = indices[row_out];
        // Caller is expected to validate indices, but guard anyway.
        if (row_in >= rows) {
            // You can choose to write zeros or just skip.
            // Here we just write zero to be safe.
            output[linear] = (T)0;
        } else {
            const size_t src_offset = (size_t)row_in * cols + col;
            output[linear] = input[src_offset];
        }
    }
}

static inline int compute_grid(size_t numel, int block_size, int mp_count) {
    const size_t blocks_for_numel =
        (numel + (size_t)block_size - 1) / (size_t)block_size;
    const int max_blocks = mp_count * 4; // simple heuristic
    int grid = (int)blocks_for_numel;
    if (grid > max_blocks) grid = max_blocks;
    if (grid < 1) grid = 1;
    return grid;
}

extern "C" void candle_index_select_f32_u32(
    const void* input,
    const uint32_t* indices,
    void* output,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count,
    int32_t multi_processor_count)
{
    const float* in = static_cast<const float*>(input);
    float* out = static_cast<float*>(output);

    const size_t out_numel = (size_t)index_count * (size_t)cols;
    const int block = 256;
    const int grid = compute_grid(out_numel, block, multi_processor_count);

    index_select_kernel<float><<<grid, block>>>(
        in, indices, out, (size_t)rows, (size_t)cols, (size_t)index_count);

    // Let Candle handle errors via cudaGetLastError if desired; we do a best-effort check.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Optional: printf or assert in debug builds
    }
}

extern "C" void candle_index_select_f16_u32(
    const void* input,
    const uint32_t* indices,
    void* output,
    uint32_t rows,
    uint32_t cols,
    uint32_t index_count,
    int32_t multi_processor_count)
{
    const __half* in = static_cast<const __half*>(input);
    __half* out = static_cast<__half*>(output);

    const size_t out_numel = (size_t)index_count * (size_t)cols;
    const int block = 256;
    const int grid = compute_grid(out_numel, block, multi_processor_count);

    index_select_kernel<__half><<<grid, block>>>(
        in, indices, out, (size_t)rows, (size_t)cols, (size_t)index_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Optional: printf or assert in debug builds
    }
}