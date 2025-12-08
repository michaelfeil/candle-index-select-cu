use candle::{DType, Device, Result, Shape, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn setup_tensors<S: Into<Shape>>(
    shape: S,
    out_rows: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dev = Device::new_cuda(1)?;

    let shape = shape.into();
    let x_f32 = Tensor::randn(0.0f32, 1.0, shape.clone(), &dev)?;
    let x_f16 = x_f32.to_dtype(DType::F16)?;

    let idx_data: Vec<u32> = (0..out_rows)
        .map(|i| (i as u32) % (shape.dims()[0] as u32))
        .collect();
    let indices = Tensor::from_vec(idx_data, out_rows, &dev)?;

    Ok((x_f32, x_f16, indices.clone(), indices))
}

fn assert_equal(a: &Tensor, b: &Tensor, eps: f64) -> Result<()> {
    let diff = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?;
    let diff = (diff.sqr()?.sum_all()? / diff.elem_count() as f64)?;
    let diff = diff.to_scalar::<f32>()? as f64;
    if diff > eps {
        candle::bail!("tensors are not equal, diff: {}", diff);
    }
    Ok(())
}

fn run_benchmark<S: Into<Shape> + Clone>(
    c: &mut Criterion,
    group_name: &str,
    shape: S,
    out_rows: usize,
) {
    let (x_f32, x_f16, idx_f32, idx_f16) = match setup_tensors(shape, out_rows) {
        Ok(t) => t,
        Err(e) => {
            println!(
                "Failed to setup tensors for group {}, skipping benchmark: {:?}",
                group_name, e
            );
            return;
        }
    };

    let mut group = c.benchmark_group(group_name);
    group.sample_size(500);
    group.warm_up_time(std::time::Duration::from_millis(1500));

    group.bench_function("native_f32", |b| {
        b.iter(|| black_box(x_f32.index_select(&idx_f32, 0).unwrap()))
    });

    let native_result_f32 = x_f32.index_select(&idx_f32, 0).unwrap();
    let custom_result_f32 = candle_index_select_cu::index_select(&x_f32, &idx_f32, 0).unwrap();
    assert_equal(&native_result_f32, &custom_result_f32, 1e-6).unwrap();
    group.bench_function("custom_f32", |b| {
        b.iter(|| black_box(candle_index_select_cu::index_select(&x_f32, &idx_f32, 0).unwrap()))
    });

    group.bench_function("native_f16", |b| {
        b.iter(|| black_box(x_f16.index_select(&idx_f16, 0).unwrap()))
    });

    let native_result_f16 = x_f16.index_select(&idx_f16, 0).unwrap();
    let custom_result_f16 = candle_index_select_cu::index_select(&x_f16, &idx_f16, 0).unwrap();
    assert_equal(&native_result_f16, &custom_result_f16, 1e-2).unwrap();
    group.bench_function("custom_f16", |b| {
        b.iter(|| black_box(candle_index_select_cu::index_select(&x_f16, &idx_f16, 0).unwrap()))
    });

    group.finish();
}

fn bench_index_select(c: &mut Criterion) {
    run_benchmark(c, "index_select_short_2d", (100, 128), 200);
    run_benchmark(c, "index_select_long_2d", (16_000, 1024), 70_000);
    run_benchmark(c, "index_select_very_long_2d", (100_000, 2048), 500_000);
    run_benchmark(c, "index_select_3d", (10, 100, 128), 200);
}

criterion_group!(benches, bench_index_select);
criterion_main!(benches);
