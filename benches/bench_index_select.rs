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
    group.measurement_time(std::time::Duration::from_millis(5000));

    let _ = x_f32.index_select(&idx_f32, 0).unwrap(); // Warmup

    group.bench_function("native_f32", |b| {
        b.iter(|| {
            let result = black_box(x_f32.index_select(&idx_f32, 0).unwrap());
            // Wait for the GPU to finish by synchronizing the device
            result.device().synchronize().unwrap();
        })
    });

    let native_result_f32 = x_f32.index_select(&idx_f32, 0).unwrap();
    let custom_result_f32 = candle_index_select_cu::index_select(&x_f32, &idx_f32, 0).unwrap();
    assert_equal(&native_result_f32, &custom_result_f32, 1e-6).unwrap();
    group.bench_function("custom_f32", |b| {
        b.iter(|| {
            let result =
                black_box(candle_index_select_cu::index_select(&x_f32, &idx_f32, 0).unwrap());
            // Wait for the GPU to finish
            result.device().synchronize().unwrap();
        })
    });

    group.bench_function("native_f16", |b| {
        b.iter(|| {
            let result = black_box(x_f16.index_select(&idx_f16, 0).unwrap());
            // Wait for the GPU to finish
            result.device().synchronize().unwrap();
        })
    });

    let native_result_f16 = x_f16.index_select(&idx_f16, 0).unwrap();
    let custom_result_f16 = candle_index_select_cu::index_select(&x_f16, &idx_f16, 0).unwrap();
    assert_equal(&native_result_f16, &custom_result_f16, 1e-2).unwrap();
    group.bench_function("custom_f16", |b| {
        b.iter(|| {
            let result =
                black_box(candle_index_select_cu::index_select(&x_f16, &idx_f16, 0).unwrap());
            // Wait for the GPU to finish
            result.device().synchronize().unwrap();
        })
    });

    group.finish();

    // --- Manual Summary ---
    println!("\n--- Summary for {} ---", group_name);
    let shape_dims = x_f32.dims();
    println!("Shape: {:?}, Out Rows: {}", shape_dims, out_rows);

    // Helper to run a few iterations and get an average time
    let measure = |f: &mut dyn FnMut()| {
        let mut durations = Vec::new();
        // Warmup
        for _ in 0..5 {
            f();
        }
        // Measurement
        for _ in 0..100 {
            let start = std::time::Instant::now();
            f();
            durations.push(start.elapsed());
        }
        let avg_duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;
        avg_duration
    };

    let mut native_f32 = || {
        let t = x_f32.index_select(&idx_f32, 0).unwrap();
        t.device().synchronize().unwrap();
    };
    let mut custom_f32 = || {
        let t = candle_index_select_cu::index_select(&x_f32, &idx_f32, 0).unwrap();
        t.device().synchronize().unwrap();
    };
    let mut native_f16 = || {
        let t = x_f16.index_select(&idx_f16, 0).unwrap();
        t.device().synchronize().unwrap();
    };
    let mut custom_f16 = || {
        let t = candle_index_select_cu::index_select(&x_f16, &idx_f16, 0).unwrap();
        t.device().synchronize().unwrap();
    };

    let native_f32_dur = measure(&mut native_f32);
    let custom_f32_dur = measure(&mut custom_f32);
    let native_f16_dur = measure(&mut native_f16);
    let custom_f16_dur = measure(&mut custom_f16);

    let f32_speedup = native_f32_dur.as_secs_f64() / custom_f32_dur.as_secs_f64();
    println!(
        "F32: Native: {:>10.3?} | Custom: {:>10.3?} | Speedup: {:.2}x",
        native_f32_dur, custom_f32_dur, f32_speedup
    );

    let f16_speedup = native_f16_dur.as_secs_f64() / custom_f16_dur.as_secs_f64();
    println!(
        "F16: Native: {:>10.3?} | Custom: {:>10.3?} | Speedup: {:.2}x",
        native_f16_dur, custom_f16_dur, f16_speedup
    );
    println!("-----------------------------------\n");
}

fn bench_index_select(c: &mut Criterion) {
    run_benchmark(c, "index_select_short_2d", (100, 128), 200);
    run_benchmark(c, "index_select_mid_2d", (16_000, 1024), 12_000);
    run_benchmark(c, "index_select_long_2d", (16_000, 1024), 70_000);
    run_benchmark(c, "index_select_very_long_2d", (100_000, 2048), 500_000);
    run_benchmark(c, "index_select_3d", (10, 100, 128), 200);
    run_benchmark(c, "index_select_long_3d", (2000, 64, 256), 10_000);
}

criterion_group!(benches, bench_index_select);
criterion_main!(benches);
