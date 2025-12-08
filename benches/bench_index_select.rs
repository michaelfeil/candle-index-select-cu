use candle::{DType, Device, Result, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn setup_tensors(
    rows: usize,
    cols: usize,
    out_rows: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dev = if Device::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    let x_f32 = Tensor::randn(0.0f32, 1.0, (rows, cols), &dev)?;
    let x_f16 = x_f32.to_dtype(DType::F16)?;

    let idx_data: Vec<u32> = (0..out_rows).map(|i| (i as u32) % (rows as u32)).collect();
    let indices = Tensor::from_vec(idx_data, out_rows, &dev)?;

    Ok((x_f32, x_f16, indices.clone(), indices))
}

fn bench_index_select(c: &mut Criterion) {
    let rows = 16_000;
    let cols = 1024;
    let out_rows = 70_000;

    let (x_f32, x_f16, idx_f32, idx_f16) = match setup_tensors(rows, cols, out_rows) {
        Ok(t) => t,
        Err(e) => {
            println!("Failed to setup tensors, skipping benchmark: {:?}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("index_select");

    group.bench_function("native_f32", |b| {
        b.iter(|| black_box(x_f32.index_select(&idx_f32, 0).unwrap()))
    });

    group.bench_function("custom_f32", |b| {
        b.iter(|| black_box(candle_index_select_cu::index_select(&x_f32, &idx_f32, 0).unwrap()))
    });

    group.bench_function("native_f16", |b| {
        b.iter(|| black_box(x_f16.index_select(&idx_f16, 0).unwrap()))
    });

    group.bench_function("custom_f16", |b| {
        b.iter(|| black_box(candle_index_select_cu::index_select(&x_f16, &idx_f16, 0).unwrap()))
    });

    group.finish();
}

criterion_group!(benches, bench_index_select);
criterion_main!(benches);
