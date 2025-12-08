# candle-index-select-cu

Fast CUDA `index_select` for [Candle](https://github.com/huggingface/candle). Inspired by https://github.com/huggingface/candle-layer-norm project.

This crate provides a specialized CUDA kernel for `index_select` along
dimension 0 on 2D tensors `[rows, cols]`, with:

- 32-bit indices (`u32`)
- `f32` and `f16` data
- CUDA only
- Benchmarks and tests against the builtin implementation
- Makes index_select faster by allowing better copy kernels for contiguous layouts

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
