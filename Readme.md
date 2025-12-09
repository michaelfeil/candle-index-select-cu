# candle-index-select-cu

Fast CUDA `index_select` for [Candle](https://github.com/huggingface/candle). Inspired by https://github.com/huggingface/candle-layer-norm project.

This crate provides a specialized CUDA kernel for `index_select` along
dimension 0 on 2D tensors `[rows, cols]`, with:

- 32-bit indices (`u32`)
- `f32` and `f16` data
- CUDA only
- Benchmarks and tests against the builtin implementation
- Makes index_select faster by allowing better copy kernels for contiguous layouts

### Benchmark Results (H100 80GB)

#### vs candle 0.9.1 native kernels

| Input Shape            | Out Rows | DType | candle 0.9.1 | candle-index-select-cu | Speedup |
|------------------------|----------|-------|--------------|------------------------|---------|
| [100, 128]             | 200      | F32   | 19.819 µs    | 15.315 µs              | **1.29×** |
| [100, 128]             | 200      | F16   | 19.244 µs    | 15.109 µs              | **1.27×** |
| [16000, 1024]          | 70000    | F32   | 1.734 ms     | 200.595 µs             | **8.64×** |
| [16000, 1024]          | 70000    | F16   | 1.088 ms     | 104.525 µs             | **10.41×** |
| [100000, 2048]         | 500000   | F32   | 26.643 ms    | 2.811 ms               | **9.48×** |
| [100000, 2048]         | 500000   | F16   | 31.391 ms    | 1.436 ms               | **21.87×** |
| [10, 100, 128]         | 200      | F32   | 35.645 µs    | 22.840 µs              | **1.56×** |
| [10, 100, 128]         | 200      | F16   | 34.558 µs    | 19.735 µs              | **1.75×** |
| [2000, 64, 256]        | 10000    | F32   | 5.003 ms     | 431.560 µs             | **11.59×** |
| [2000, 64, 256]        | 10000    | F16   | 2.415 ms     | 223.718 µs             | **10.80×** |

#### vs PyTorch 2.1 (CUDA 12.1)

| Input Shape            | Out Rows | DType | PyTorch      | candle-index-select-cu | Speedup |
|------------------------|----------|-------|--------------|------------------------|---------|
| [100, 128]             | 200      | F32   | 16.078 µs    | 15.315 µs              | **1.05×** |
| [100, 128]             | 200      | F16   | 15.603 µs    | 15.109 µs              | **1.03×** |
| [16000, 1024]          | 70000    | F32   | 519.141 µs   | 200.595 µs             | **2.59×** |
| [16000, 1024]          | 70000    | F16   | 492.888 µs   | 104.525 µs             | **4.71×** |
| [100000, 2048]         | 500000   | F32   | 7.981 ms     | 2.811 ms               | **2.84×** |
| [100000, 2048]         | 500000   | F16   | 7.191 ms     | 1.436 ms               | **5.01×** |
| [10, 100, 128]         | 200      | F32   | 25.257 µs    | 22.840 µs              | **1.11×** |
| [10, 100, 128]         | 200      | F16   | 25.297 µs    | 19.735 µs              | **1.28×** |
| [2000, 64, 256]        | 10000    | F32   | 1.034 ms     | 431.560 µs             | **2.40×** |
| [2000, 64, 256]        | 10000    | F16   | 1.004 ms     | 223.718 µs             | **4.49×** |

*Benchmarks run on NVIDIA H100 80GB HBM3.*

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
candle-index-select-cu = { git = "https://github.com/michaelfeil/candle-index-select-cu" }
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
```

### For older candle versions (pre-0.9)

```toml
[dependencies]
candle-index-select-cu = { git = "https://github.com/michaelfeil/candle-index-select-cu", default-features = false, features = ["cuda-11"] }
```

## Fast Path Conditions

The CUDA kernel is used when all of these conditions are met:

- Device: CUDA
- Input tensor: rank 2 (2D), last dim contiguous
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
