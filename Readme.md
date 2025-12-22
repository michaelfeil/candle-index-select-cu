# candle-index-select-cu

Fast CUDA `index_select` for [Candle](https://github.com/huggingface/candle). Inspired by https://github.com/huggingface/candle-layer-norm project.

This crate provides a specialized CUDA kernel for `index_select` along
dimension 0 on multi-dimensional tensors, with:

- **Optimized memory access patterns** using vectorized loads (float4 for F32, half2 for F16)
- **F32 and F16 data types** with dtype-specific optimizations
- **CUDA-only implementation** with automatic fallback to candle's builtin
- **Dimension 0 specialization** for row-based indexing operations
- **Contiguous memory layout optimization** for better cache utilization
- **Significant speedups** on large tensors (up to 4-22× vs candle, 4× vs PyTorch 2.1) 
- **Tested** against candle index_select for correctness.
- faster only if dim=0 and tensor dim=0 cols is divisible by 2 for fp16 (half2) and 4 for fp32 (float4) memory ops.

### Benchmark Results (H100 80GB)

#### vs candle 0.9.1 native kernels

| Input Shape            | Out Rows | DType | candle 0.9.1 | candle-index-select-cu | Speedup |
|------------------------|----------|-------|--------------|------------------------|---------|
| [100, 128]             | 200      | F32   | 16.436 µs    | 12.645 µs              | **1.30×** |
| [100, 128]             | 200      | F16   | 16.676 µs    | 12.505 µs              | **1.33×** |
| [16000, 1024]          | 12000    | F32   | 340.410 µs   | 46.250 µs              | **7.36×** |
| [16000, 1024]          | 12000    | F16   | 106.553 µs   | 28.435 µs              | **3.75×** |
| [16000, 1024]          | 70000    | F32   | 1.980 ms     | 204.540 µs             | **9.68×** |
| [16000, 1024]          | 70000    | F16   | 1.073 ms     | 106.728 µs             | **10.06×** |
| [100000, 2048]         | 500000   | F32   | 45.636 ms    | 2.828 ms               | **16.14×** |
| [100000, 2048]         | 500000   | F16   | 33.665 ms    | 1.469 ms               | **22.92×** |
| [10, 100, 128]         | 200      | F32   | 33.675 µs    | 16.217 µs              | **2.08×** |
| [10, 100, 128]         | 200      | F16   | 33.953 µs    | 17.115 µs              | **1.98×** |
| [2000, 64, 256]        | 10000    | F32   | 3.737 ms     | 432.668 µs             | **8.64×** |
| [2000, 64, 256]        | 10000    | F16   | 2.880 ms     | 211.468 µs             | **13.62×** |

*Benchmarks run on NVIDIA H100 80GB HBM3.*

#### vs PyTorch 2.1.0a0+b5021ba (CUDA 12.1)

| Input Shape            | Out Rows | DType | PyTorch      | candle-index-select-cu | Speedup |
|------------------------|----------|-------|--------------|------------------------|---------|
| [100, 128]             | 200      | F32   | 15.256 µs    | 12.645 µs              | **1.21×** |
| [100, 128]             | 200      | F16   | 14.928 µs    | 12.505 µs              | **1.19×** |
| [16000, 1024]          | 70000    | F32   | 515.830 µs   | 204.540 µs             | **2.52×** |
| [16000, 1024]          | 70000    | F16   | 489.939 µs   | 106.728 µs             | **4.59×** |
| [100000, 2048]         | 500000   | F32   | 7.958 ms     | 2.828 ms               | **2.81×** |
| [100000, 2048]         | 500000   | F16   | 7.127 ms     | 1.469 ms               | **4.85×** |
| [10, 100, 128]         | 200      | F32   | 25.007 µs    | 16.217 µs              | **1.54×** |
| [10, 100, 128]         | 200      | F16   | 24.808 µs    | 17.115 µs              | **1.45×** |
| [2000, 64, 256]        | 10000    | F32   | 1.035 ms     | 432.668 µs             | **2.39×** |
| [2000, 64, 256]        | 10000    | F16   | 1.005 ms     | 211.468 µs             | **4.75×** |

*Benchmarks depend on Pytorch version.*

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
candle-index-select-cu = "0.0.1"
candle-core = { version = "0.9", features = ["cuda"] }
```

Then use it in your code:

```rust
use candle_index_select_cu::index_select;
use candle_core::{Device, Tensor, DType};

let device = Device::new_cuda(0)?;
let x = Tensor::randn(0f32, 1.0, (1000, 512), &device)?;
let indices = Tensor::from_vec(vec![0u32, 5, 10, 15], 4, &device)?;

// Fast path for 2D + dim 0 + f32/f16 + u32 indices
// Falls back to candle's builtin for other cases
let result = index_select(&x, &indices, 0)?;
```

### Recommended Integration with Feature Flags

For better compatibility to CPU and other backends, consider wrapping the usage behind a CUDA feature flag:

```toml
[dependencies]
candle-index-select-cu = { version = "0.0.1", optional = true }
candle-core = "0.9"

[features]
default = []
cuda = ["candle-core/cuda", "candle-index-select-cu"]
```

```rust
#[cfg(feature = "cuda")]
use candle_index_select_cu;

#[inline]
pub fn index_select(tensor: &Tensor, ids: &Tensor, dim: usize) -> Result<Tensor> {
    #[cfg(not(feature = "cuda"))]
    {
        tensor.index_select(ids, dim)
    }
    #[cfg(feature = "cuda")]
    {
        candle_index_select_cu::index_select(tensor, ids, dim)
    }
}
```

## Feature Flags

This crate supports both older and newer versions of candle/cudarc:

| Feature | Candle Version | cudarc Version | Default |
|---------|---------------|----------------|---------|
| `cuda-12` | 0.9+ | 0.16+ | ✅ |
| `cuda-11` | pre-0.9 (0.6, 0.7, 0.8) | pre-0.16 | |

### For candle 0.9+ (default)

```toml
[dependencies]
candle-index-select-cu = { git = "https://github.com/michaelfeil/candle-index-select-cu" }
# best to globally pin candle -> via patches
```

### For older candle versions (pre-0.9)

```toml
[dependencies]
candle-index-select-cu = { git = "https://github.com/michaelfeil/candle-index-select-cu", default-features = false, features = ["cuda-11"] }
```

## Fast Path Conditions

The CUDA kernel is used when all of these conditions are met:

- Device: CUDA
- Input tensor: rank >= 2 (2D, 3D, 4D, etc.), fully contiguous
- Indices tensor: rank 1, contiguous, DType::U32
- Dimension: 0
- Input dtype: F16 or F32

For all other cases, the function falls back to candle's builtin `index_select`.

## Running Tests

```bash
cargo test
```

## Running Benchmarks

```bash
cargo bench
```

## Profiling

### NSys
```
cargo build --release --bin profile_index_select && nsys profile -t cuda,osrt --stats=true -o nsys_index_select   cargo/release/profile_index_select
```

### NCU
```
cargo build --release --bin profile_index_select && CUDA_LAUNCH_BLOCKING=1 ncu --set full     --devices 0     --target-processes all     --kernel-name-base demangled     -k "regex:.*index_select.*"     --launch-skip 1500     --launch-count 1     -o index_select_profile  cargo/release/profile_index_select
```
