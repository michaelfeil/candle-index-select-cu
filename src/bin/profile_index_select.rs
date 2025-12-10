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

use candle::{DType, Device, Result, Tensor};
use std::os::raw::c_char;

// =============================
// NVTX FFI
// =============================
#[link(name = "nvToolsExt")]
extern "C" {
    fn nvtxRangePushA(msg: *const c_char) -> i32;
    fn nvtxRangePop() -> i32;
}

fn main() -> Result<()> {
    std::thread::sleep(std::time::Duration::from_secs(1)); // time to attach profiler
    // --- Config: this is your target case ---
    let rows = 16_000usize;
    let cols = 1_024usize;
    let out_rows = 70_000usize;
    let iters = 1000; // enough to get stable averages in Nsight

    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);

    // --- Setup tensors ---
    let x = Tensor::randn(0.0f32, 1.0, (rows, cols), &device)?
        .to_dtype(DType::F16)?; // mainly profiling f16; switch to F32 if you want

    let idx_data: Vec<u32> = (0..out_rows)
        .map(|i| (i as u32) % (rows as u32))
        .collect();
    let indices = Tensor::from_vec(idx_data, out_rows, &device)?
        .to_dtype(DType::U32)?;

    // --- Warmup (JIT, caches, etc.) ---
    for _ in 0..iters {
        let _ = candle_index_select::index_select(&x, &indices, 0)?;
    }
    device.synchronize()?;

    println!(
        "Profiling index_select: shape=({}, {}), out_rows={}, iters={}",
        rows, cols, out_rows, iters
    );

    // --- Profiling region with NVTX range per call ---
    for _ in 0..iters {
        unsafe {
            // This is exactly what you asked for:
            nvtxRangePushA(b"index_select_f16\0".as_ptr() as *const c_char);
        }

        let _ = candle_index_select::index_select(&x, &indices, 0)?;

        unsafe {
            nvtxRangePop();
        }
    }
    device.synchronize()?;

    Ok(())
}