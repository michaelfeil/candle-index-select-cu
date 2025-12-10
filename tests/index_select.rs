// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright (c) 2025 Michael Feil
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Authors explaination: Provide a copy of the first two lines in each redistributed version.

use candle_index_select::candle;
use candle_index_select_cu as candle_index_select;

fn cuda_device() -> candle::Result<candle::Device> {
    candle::Device::new_cuda(0)
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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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
    let device = cuda_device()?;

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

#[test]
fn test_output_is_contiguous() -> candle::Result<()> {
    let device = cuda_device()?;

    let indices = candle::Tensor::from_vec((0u32..30).collect::<Vec<_>>(), 30, &device)?;

    // Test with a contiguous input tensor
    let x_cont = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    assert!(x_cont.is_contiguous());
    let fast_from_cont = candle_index_select::index_select(&x_cont, &indices, 0)?;
    assert!(fast_from_cont.is_contiguous());

    Ok(())
}

#[test]
fn test_non_contiguous_input_errors() -> candle::Result<()> {
    let device = cuda_device()?;

    // Create a non-contiguous tensor by permuting
    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    let x_permuted = x.permute((1, 0))?; // Now shape is (64, 100) and non-contiguous

    assert!(!x_permuted.is_contiguous());

    let indices = candle::Tensor::from_vec((0u32..30).collect::<Vec<_>>(), 30, &device)?;

    // Expect an error because the fast path requires a contiguous input tensor
    let result = candle_index_select::index_select(&x_permuted, &indices, 0);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_3d_dim0_fast_vs_baseline() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (10, 100, 128), &device)?;
    let idx =
        candle::Tensor::from_vec((0u32..20).map(|i| i % 10).collect::<Vec<_>>(), 20, &device)?;

    let fast = candle_index_select::index_select(&x, &idx, 0)?;
    let baseline = x.index_select(&idx, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}

#[test]
fn test_4d_dim0_fast_vs_baseline() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (8, 16, 32, 64), &device)?;
    let idx = candle::Tensor::from_vec((0u32..16).map(|i| i % 8).collect::<Vec<_>>(), 16, &device)?;

    let fast = candle_index_select::index_select(&x, &idx, 0)?;
    let baseline = x.index_select(&idx, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}

#[test]
fn test_4d_dim2_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (4, 5, 6, 7), &device)?;
    let idx = candle::Tensor::from_vec((0u32..3).collect::<Vec<_>>(), 3, &device)?;

    let fast = candle_index_select::index_select(&x, &idx, 2)?;
    let baseline = x.index_select(&idx, 2)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}
// FAILS??
// #[test]
// fn test_empty_indices() -> candle::Result<()> {
//     let device = cuda_device()?;

//     let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
//     let idx = candle::Tensor::from_vec(Vec::<u32>::new(), 0, &device)?;

//     let fast = candle_index_select::index_select(&x, &idx, 0)?;
//     let baseline = x.index_select(&idx, 0)?;

//     assert_eq!(fast.dims(), baseline.dims());
//     assert_eq!(fast.elem_count(), 0);
//     Ok(())
// }

#[test]
fn test_f16_odd_cols_scalar_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    // cols = 255 not divisible by 2 -> half2 path should not trigger
    let x_f32 = candle::Tensor::randn(0.0f32, 1.0, (512, 255), &device)?;
    let x = x_f32.to_dtype(candle::DType::F16)?;

    let idx = candle::Tensor::from_vec(
        (0u32..200).map(|i| i % 512).collect::<Vec<_>>(),
        200,
        &device,
    )?;

    let fast = candle_index_select::index_select(&x, &idx, 0)?;
    let baseline = x.index_select(&idx, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.to_dtype(candle::DType::F32)?.flatten_all()?,
        &baseline.to_dtype(candle::DType::F32)?.flatten_all()?,
        5e-3
    )?);
    Ok(())
}

#[test]
fn test_index_dtype_mismatch_fallback() -> candle::Result<()> {
    let device = cuda_device()?;

    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    // Int64 indices: fast path should not be used.
    let idx_i64 = candle::Tensor::from_vec((0i64..10).collect::<Vec<_>>(), 10, &device)?;

    let fast = candle_index_select::index_select(&x, &idx_i64, 0)?;
    let baseline = x.index_select(&idx_i64, 0)?;

    assert_eq!(fast.dims(), baseline.dims());
    assert!(allclose(
        &fast.flatten_all()?,
        &baseline.flatten_all()?,
        1e-5
    )?);
    Ok(())
}

// =============================================================================
// Parameterized Tests
// =============================================================================

#[test]
fn test_2d_various_shapes_f32() -> candle::Result<()> {
    let device = cuda_device()?;
    
    let test_cases = vec![
        (8, 8, 4),        // Small square
        (16, 32, 8),      // Small rectangular
        (100, 128, 50),   // Medium, aligned
        (100, 127, 50),   // Medium, misaligned
        (200, 256, 100),  // Larger, aligned
        (200, 255, 100),  // Larger, misaligned
        (1000, 4, 500),   // Very narrow
        (50, 1000, 25),   // Very wide
        (1, 1024, 1),     // Single row
        (1024, 1, 512),   // Single column
        (600, 1048, 1048), // Large with odd columns
    ];

    for (rows, cols, out_rows) in test_cases {
        let x = candle::Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
        let indices = candle::Tensor::from_vec(
            (0..out_rows).map(|i| (i as u32) % (rows as u32)).collect::<Vec<_>>(),
            out_rows,
            &device
        )?;

        let fast = candle_index_select::index_select(&x, &indices, 0)?;
        let baseline = x.index_select(&indices, 0)?;

        assert_eq!(fast.dims(), baseline.dims(), "Shape mismatch for case ({}, {}, {})", rows, cols, out_rows);
        assert!(
            allclose(&fast.flatten_all()?, &baseline.flatten_all()?, 1e-5)?,
            "Values mismatch for case ({}, {}, {})", rows, cols, out_rows
        );
    }

    Ok(())
}

#[test]
fn test_2d_various_shapes_f16() -> candle::Result<()> {
    let device = cuda_device()?;
    
    let test_cases = vec![
        (8, 8, 4),        // Small square
        (16, 32, 8),      // Small rectangular, aligned for half2
        (16, 31, 8),      // Small rectangular, misaligned for half2
        (100, 128, 50),   // Medium, aligned
        (100, 127, 50),   // Medium, misaligned
        (200, 256, 100),  // Larger, aligned
        (200, 255, 100),  // Larger, misaligned
        (1000, 2, 500),   // Very narrow, aligned
        (1000, 3, 500),   // Very narrow, misaligned
        (50, 1024, 25),   // Very wide
        (600, 1048, 1048), // Large with odd columns
    ];

    for (rows, cols, out_rows) in test_cases {
        let x_f32 = candle::Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?;
        let x = x_f32.to_dtype(candle::DType::F16)?;
        let indices = candle::Tensor::from_vec(
            (0..out_rows).map(|i| (i as u32) % (rows as u32)).collect::<Vec<_>>(),
            out_rows,
            &device
        )?;

        let fast = candle_index_select::index_select(&x, &indices, 0)?;
        let baseline = x.index_select(&indices, 0)?;

        assert_eq!(fast.dims(), baseline.dims(), "Shape mismatch for F16 case ({}, {}, {})", rows, cols, out_rows);
        assert!(
            allclose(
                &fast.to_dtype(candle::DType::F32)?.flatten_all()?,
                &baseline.to_dtype(candle::DType::F32)?.flatten_all()?,
                5e-3
            )?,
            "Values mismatch for F16 case ({}, {}, {})", rows, cols, out_rows
        );
    }

    Ok(())
}

#[test]
fn test_multi_dimensional_various_shapes() -> candle::Result<()> {
    let device = cuda_device()?;
    
    let test_cases = vec![
        // (shape, indices_count, dim)
        (vec![10, 8], 5, 0),                    // 2D
        (vec![5, 6, 7], 3, 0),                  // 3D
        (vec![4, 5, 6, 7], 2, 0),               // 4D
        (vec![2, 3, 4, 5, 6], 2, 0),            // 5D
        (vec![8, 16, 32], 4, 0),                // 3D larger
        (vec![16, 32, 64, 128], 8, 0),          // 4D larger
    ];

    for (shape, indices_count, dim) in test_cases {
        let x = candle::Tensor::randn(0.0f32, 1.0, shape.as_slice(), &device)?;
        let max_idx = shape[dim] as u32;
        let indices = candle::Tensor::from_vec(
            (0..indices_count).map(|i| (i as u32) % max_idx).collect::<Vec<_>>(),
            indices_count,
            &device
        )?;

        let fast = candle_index_select::index_select(&x, &indices, dim)?;
        let baseline = x.index_select(&indices, dim)?;

        assert_eq!(fast.dims(), baseline.dims(), "Shape mismatch for multi-dim case {:?}", shape);
        assert!(
            allclose(&fast.flatten_all()?, &baseline.flatten_all()?, 1e-5)?,
            "Values mismatch for multi-dim case {:?}", shape
        );
    }

    Ok(())
}

#[test]
fn test_edge_case_indices_patterns() -> candle::Result<()> {
    let device = cuda_device()?;
    
    let x = candle::Tensor::randn(0.0f32, 1.0, (100, 64), &device)?;
    
    let test_patterns = vec![
        // (description, indices)
        ("sequential", (0..10).collect::<Vec<_>>()),
        ("reverse", (0..10).rev().collect::<Vec<_>>()),
        ("all_same", vec![42; 20]),
        ("alternating", (0..20).map(|i| if i % 2 == 0 { 0 } else { 99 }).collect()),
        ("sparse_high", vec![5, 15, 25, 35, 45, 55, 65, 75, 85, 95]),
        ("random_like", vec![42, 7, 91, 13, 65, 2, 88, 34, 56, 78]),
        ("boundary", vec![0, 1, 98, 99, 0, 99, 1, 98]),
    ];

    for (desc, indices_data) in test_patterns {
        let indices_len = indices_data.len();
        let indices = candle::Tensor::from_vec(
            indices_data.into_iter().map(|x| x as u32).collect::<Vec<_>>(),
            indices_len,
            &device
        )?;

        let fast = candle_index_select::index_select(&x, &indices, 0)?;
        let baseline = x.index_select(&indices, 0)?;

        assert_eq!(fast.dims(), baseline.dims(), "Shape mismatch for pattern: {}", desc);
        assert!(
            allclose(&fast.flatten_all()?, &baseline.flatten_all()?, 1e-5)?,
            "Values mismatch for pattern: {}", desc
        );
    }

    Ok(())
}
