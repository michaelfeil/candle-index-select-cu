# candle-index-select-cu

Fast CUDA `index_select` for [Candle](https://github.com/huggingface/candle). Inspired by https://github.com/huggingface/candle-layer-norm project.

This crate provides a specialized CUDA kernel for `index_select` along
dimension 0 on 2D tensors `[rows, cols]`, with:

- 32-bit indices (`u32`)
- `f32` and `f16` data
- CUDA only
- Benchmarks and tests against the builtin implementation
- Makes index_select faster by allowing better copy kernels for contiguous layouts

### Benchmark vs candle 0.9.1 kernels

| Input Shape            | Out Rows | DType | candle 0.9.1 Time | candle-index-select-cu | Speedup |
|------------------------|----------|-------|-------------|-------------|---------|
| [100, 128]             | 200      | F32   | 16.459 µs   | 12.104 µs   | **1.36x** |
| [100, 128]             | 200      | F16   | 17.900 µs   | 12.262 µs   | **1.46x** |
| [16000, 1024]          | 70000    | F32   | 1.644 ms    | 202.462 µs  | **8.12×** |
| [16000, 1024]          | 70000    | F16   | 1.079 ms    | 123.780 µs  | **8.72x** |
| [100000, 2048]         | 500000   | F32   | 23.974 ms   | 2.838 ms    | **8.43×** |
| [100000, 2048]         | 500000   | F16   | 15.920 ms   | 1.716 ms    | **9.28×** |
| [10, 100, 128]         | 200      | F32   | 32.470 µs   | 16.036 µs   | **2.02×** |
| [10, 100, 128]         | 200      | F16   | 32.240 µs   | 16.390 µs   | **1.97×** |
| [2000, 64, 256]        | 10000    | F32   | 3.785 ms    | 453.247 µs  | **8.35×** |
| [2000, 64, 256]        | 10000    | F16   | 2.399 ms    | 282.735 µs  | **8.48x×** |

all measure on h100.

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
