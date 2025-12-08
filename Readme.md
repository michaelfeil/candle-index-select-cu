# candle-index-select-cu

Fast CUDA `index_select` for [Candle](https://github.com/huggingface/candle). Inspired by https://github.com/huggingface/candle-layer-norm project.

This crate provides a specialized CUDA kernel for `index_select` along
dimension 0 on 2D tensors `[rows, cols]`, with:

- 32-bit indices (`u32`)
- `f32` and `f16` data
- cuda only.
- Benchmarks and tests against the builtin implementation

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
candle-index-select-cu = { path = "./candle-index-select-cu" }
candle-core = { version = "0.9", features = ["cuda"] }

