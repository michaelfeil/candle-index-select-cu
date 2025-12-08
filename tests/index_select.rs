use candle_index_select_cu as candle_index_select;
use candle_index_select::candle;

fn maybe_cuda_device() -> Option<candle::Device> {
    match candle::Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

fn allclose(a: &candle::Tensor, b: &candle::Tensor, tol: f32) -> candle::Result<bool> {
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

// =============================================================================
// 2D Tensor Tests
// =============================================================================

#[test]
fn test_2d_f32_contiguous() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let rows = 16_000;
    let cols = 1024;
    let out_rows = 70_000;

    let x = candle::Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
    let idx_data: Vec<u32> = (0..out_rows).map(|i| (i as u32) % (rows as u32)).collect();
    let indices = candle::Tensor::from_vec(idx_data, out_rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-4
    )?);

    Ok(())
}

#[test]
fn test_2d_f16_contiguous() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let rows = 8_000;
    let cols = 1024;
    let out_rows = 32_000;

    let x_f32 = candle::Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
    let x = x_f32.to_dtype(candle::DType::F16)?;

    let idx_data: Vec<u32> = (0..out_rows).map(|i| (i as u32) % (rows as u32)).collect();
    let indices = candle::Tensor::from_vec(idx_data, out_rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.to_dtype(candle::DType::F32)?.flatten_all()?,
        &baseline.to_dtype(candle::DType::F32)?.flatten_all()?,
        5e-3
    )?);

    Ok(())
}

#[test]
fn test_2d_small() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let x = candle::Tensor::randn(0.0f32, 1.0, (10, 8), &device)?;
    let indices = candle::Tensor::from_vec(vec![0u32, 2, 5, 9, 1], 5, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_2d_non_contiguous_slice() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    // Create tensor and take a slice (which may be non-contiguous depending on dim)
    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    let x_slice = x.narrow(0, 10, 50)?; // Take rows 10..60

    let indices = candle::Tensor::from_vec((0u32..30).collect::<Vec<_>>(), 30, &device)?;

    let fast = candle_index_select::index_select(&x_slice, &indices, 0)?;
    let baseline = x_slice.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

// =============================================================================
// 3D Tensor Tests (should fall back to candle's implementation)
// =============================================================================

#[test]
fn test_3d_contiguous_fallback() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    // 3D tensor - should use fallback since we only support 2D
    let x = candle::Tensor::randn(0.0f32, 1.0, (32, 64, 128), &device)?;
    let indices = candle::Tensor::from_vec((0u32..16).collect::<Vec<_>>(), 16, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_3d_dim1_fallback() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let x = candle::Tensor::randn(0.0f32, 1.0, (16, 32, 64), &device)?;
    let indices = candle::Tensor::from_vec((0u32..10).collect::<Vec<_>>(), 10, &device)?;

    // Index along dim 1 - should use fallback
    let fast = candle_index_select::index_select(&x, &indices, 1)?;
    let baseline = x.index_select(&indices, 1)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_3d_permuted_contiguous_fallback() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let x = candle::Tensor::randn(0.0f32, 1.0, (16, 32, 64), &device)?;
    let x_perm = x.permute((1, 0, 2))?.contiguous()?; // Permute then make contiguous

    let indices = candle::Tensor::from_vec((0u32..10).collect::<Vec<_>>(), 10, &device)?;

    let fast = candle_index_select::index_select(&x_perm, &indices, 0)?;
    let baseline = x_perm.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_duplicate_indices() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    // Many duplicates
    let indices = candle::Tensor::from_vec(vec![0u32, 0, 1, 1, 2, 2, 0, 5, 5, 5], 10, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_single_row_select() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let x = candle::Tensor::randn(0.0f32, 1.0, (1000, 256), &device)?;
    let indices = candle::Tensor::from_vec(vec![42u32], 1, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_all_rows_select() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let rows = 256;
    let x = candle::Tensor::randn(0.0f32, 1.0, (rows, 128), &device)?;
    let indices = candle::Tensor::from_vec((0..rows as u32).collect::<Vec<_>>(), rows, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}

#[test]
fn test_wide_columns() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    // Very wide tensor (embedding-like)
    let x = candle::Tensor::randn(0.0f32, 1.0, (1000, 4096), &device)?;
    let indices = candle::Tensor::from_vec((0u32..500).collect::<Vec<_>>(), 500, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-4
    )?);

    Ok(())
}

#[test]
fn test_narrow_columns() -> candle::Result<()> {
    let device = match maybe_cuda_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    // Very narrow tensor
    let x = candle::Tensor::randn(0.0f32, 1.0, (10000, 4), &device)?;
    let indices = candle::Tensor::from_vec((0u32..5000).collect::<Vec<_>>(), 5000, &device)?;

    let fast = candle_index_select::index_select(&x, &indices, 0)?;
    let baseline = x.index_select(&indices, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);

    Ok(())
}
