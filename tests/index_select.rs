use candle_core as candle;
use candle::Tensor;

fn maybe_cuda_device() -> Option<candle::Device> {
    match candle::Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

fn allclose(a: &Tensor, b: &Tensor, tol: f32) -> candle::Result<bool> {
    let da = a.to_vec1::<f32>()?;
    let db = b.to_vec1::<f32>()?;
    if da.len() != db.len() {
        return Ok(false);
    }
    for (x, y) in da.iter().zip(db.iter()) {
        if (x - y).abs() > tol {
            return Ok(false);
        }
    }
    Ok(true)
}

#[test]
fn compare_with_builtin_f32() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()), // skip when no CUDA
    };

    let rows = 16_000;
    let cols = 1024;
    let out_rows = 70_000;

    let x = Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
    let idx_data: Vec<u32> = (0..out_rows)
        .map(|i| (i as u32) % (rows as u32))
        .collect();
    let indices = Tensor::from_vec(idx_data, out_rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(&fast.flatten_all()?, &baseline.flatten_all()?, 1e-4)?);

    Ok(())
}

#[test]
fn compare_with_builtin_f16() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let rows = 8_000;
    let cols = 1024;
    let out_rows = 32_000;

    let x_f32 = Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
    let x = x_f32.to_dtype(candle::DType::F16)?;

    let idx_data: Vec<u32> = (0..out_rows)
        .map(|i| (i as u32) % (rows as u32))
        .collect();
    let indices = Tensor::from_vec(idx_data, out_rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    // Looser tolerance due to f16
    assert!(allclose(
        &fast.to_dtype(candle::DType::F32)?.flatten_all()?,
        &baseline.to_dtype(candle::DType::F32)?.flatten_all()?,
        5e-3
    )?);

    Ok(())
}
// tOdo: add much more test (2D, 3D, contiguous, non-contiguous, dim 1, dim 2, etc.)
// IMPORTANT.