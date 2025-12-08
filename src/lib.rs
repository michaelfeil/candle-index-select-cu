mod ffi;

pub use candle;
use candle::backend::BackendStorage;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use half::f16;

#[cfg(feature = "cuda-12")]
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
#[cfg(feature = "cuda-12")]
use candle::cuda_backend::cudarc::driver::DevicePtr;

#[cfg(feature = "cuda-11")]
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
#[cfg(feature = "cuda-11")]
use candle::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda-11")]
use candle::cuda_backend::WrapErr;

fn index_select_internal_type(dtype: DType) -> Result<u32> {
    let code = match dtype {
        DType::F16 => 0,
        DType::F32 => 1,
        dt => candle::bail!("candle-index-select only supports f16 and f32 (got {dt:?})"),
    };
    Ok(code)
}

/// Fast CUDA index_select along dim 0 for 2D inputs.
pub struct IndexSelect {
    pub dim: usize,
}

impl IndexSelect {
    #[cfg(feature = "cuda-12")]
    fn fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        ids: &CudaStorage,
        ids_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = x.device();

        let x_stride = x_l.stride();
        let ids_stride = ids_l.stride();
        let x_rank = x_stride.len();
        let ids_rank = ids_stride.len();

        if x_rank != 2 {
            candle::bail!("candle-index-select expects input tensors of rank 2. Found: {x_rank}");
        }
        if ids_rank != 1 {
            candle::bail!("candle-index-select expects index tensor of rank 1. Found: {ids_rank}");
        }
        if x_stride[x_rank - 1] != 1 {
            candle::bail!(
                "the last dim of x must be contiguous for candle-index-select ({x_stride:?})"
            );
        }
        if ids_stride[ids_rank - 1] != 1 {
            candle::bail!(
                "indices tensor must be contiguous for candle-index-select ({ids_stride:?})"
            );
        }
        if self.dim != 0 {
            candle::bail!(
                "candle-index-select only supports dim == 0 for now (got {})",
                self.dim
            );
        }

        let rows = x_l.dims()[0];
        let cols = x_l.dims()[1];
        let index_count = ids_l.dims()[0];

        let dtype_code = index_select_internal_type(x.dtype())?;

        let x = x.as_cuda_slice::<T>()?;
        let ids = ids.as_cuda_slice::<u32>()?;

        let x = x.slice(x_l.start_offset()..);
        let ids = ids.slice(ids_l.start_offset()..);

        let out_shape = Shape::from((index_count, cols));
        let out = unsafe { dev.alloc::<T>(out_shape.elem_count()) }?;

        // cudarc 0.16+ API: get stream from device, device_ptr takes stream arg
        let stream = dev.cuda_stream();
        let stream_ref = stream.as_ref();

        let multi_processors_count = stream
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .expect("failed to query CUDA multiprocessor count");

        {
            let (x_devptr, _x_sync) = x.device_ptr(stream_ref);
            let x_ptr = x_devptr as usize as *const core::ffi::c_void;

            let (ids_devptr, _ids_sync) = ids.device_ptr(stream_ref);
            let ids_ptr = ids_devptr as usize as *const u32;

            let (out_devptr, _out_sync) = out.device_ptr(stream_ref);
            let dst_ptr = out_devptr as usize as *mut core::ffi::c_void;

            unsafe {
                ffi::run_index_select(
                    x_ptr,
                    ids_ptr,
                    dst_ptr,
                    rows as u32,
                    cols as u32,
                    index_count as u32,
                    multi_processors_count,
                    dtype_code,
                );
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }

    #[cfg(feature = "cuda-11")]
    fn fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        ids: &CudaStorage,
        ids_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = x.device();

        let x_stride = x_l.stride();
        let ids_stride = ids_l.stride();
        let x_rank = x_stride.len();
        let ids_rank = ids_stride.len();

        if x_rank != 2 {
            candle::bail!("candle-index-select expects input tensors of rank 2. Found: {x_rank}");
        }
        if ids_rank != 1 {
            candle::bail!("candle-index-select expects index tensor of rank 1. Found: {ids_rank}");
        }
        if x_stride[x_rank - 1] != 1 {
            candle::bail!(
                "the last dim of x must be contiguous for candle-index-select ({x_stride:?})"
            );
        }
        if ids_stride[ids_rank - 1] != 1 {
            candle::bail!(
                "indices tensor must be contiguous for candle-index-select ({ids_stride:?})"
            );
        }
        if self.dim != 0 {
            candle::bail!(
                "candle-index-select only supports dim == 0 for now (got {})",
                self.dim
            );
        }

        let rows = x_l.dims()[0];
        let cols = x_l.dims()[1];
        let index_count = ids_l.dims()[0];

        let dtype_code = index_select_internal_type(x.dtype())?;

        let x = x.as_cuda_slice::<T>()?;
        let ids = ids.as_cuda_slice::<u32>()?;

        let x = x.slice(x_l.start_offset()..);
        let ids = ids.slice(ids_l.start_offset()..);

        let out_shape = Shape::from((index_count, cols));
        // Old cudarc API: alloc returns cudarc error, needs .w() wrapper
        let out = unsafe { dev.alloc::<T>(out_shape.elem_count()) }.w()?;

        // Old cudarc API: device_ptr() takes no args, returns &u64
        let x_ptr = *x.device_ptr() as *const core::ffi::c_void;
        let ids_ptr = *ids.device_ptr() as *const u32;
        let dst_ptr = *out.device_ptr() as *mut core::ffi::c_void;

        // Old candle API: attribute() on device directly
        let multi_processors_count = dev
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .expect("failed to query CUDA multiprocessor count");

        unsafe {
            ffi::run_index_select(
                x_ptr,
                ids_ptr,
                dst_ptr,
                rows as u32,
                cols as u32,
                index_count as u32,
                multi_processors_count,
                dtype_code,
            );
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }
}

impl candle::CustomOp2 for IndexSelect {
    fn name(&self) -> &'static str {
        "candle-index-select"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for candle-index-select (CUDA only)")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        ids: &CudaStorage,
        ids_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.fwd::<f16>(x, x_l, ids, ids_l),
            DType::F32 => self.fwd::<f32>(x, x_l, ids, ids_l),
            dt => candle::bail!("candle-index-select only supports f16 and f32 (got {dt:?})"),
        }
    }
}

/// Public API:
/// Fast path + fallback to `Tensor::index_select`.
///
/// * Fast path:
///   - Device: CUDA
///   - x: rank-2, last dim contiguous
///   - indices: rank-1, contiguous, DType::U32
///   - dim == 0
///   - dtype(x) ∈ {F16, F32}
///
/// * Fallback:
///   - everything else → `x.index_select(indices, dim)`
pub fn index_select(x: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
    use candle::{DType, Device};

    let devices_match = match (x.device(), indices.device()) {
        (Device::Cpu, Device::Cpu) => true,
        (Device::Cuda(a), Device::Cuda(b)) => a.id() == b.id(),
        _ => false,
    };
    if !devices_match {
        return x.index_select(indices, dim);
    }

    let device = x.device();

    if !matches!(device, Device::Cuda(_)) {
        return x.index_select(indices, dim);
    }

    if !matches!(x.dtype(), DType::F16 | DType::F32) {
        return x.index_select(indices, dim);
    }

    if indices.dtype() != DType::U32 {
        return x.index_select(indices, dim);
    }

    let x_shape = x.shape();
    let ids_shape = indices.shape();

    if x_shape.rank() != 2 || ids_shape.rank() != 1 {
        return x.index_select(indices, dim);
    }

    if dim != 0 {
        return x.index_select(indices, dim);
    }

    let (x_sto, x_l) = x.storage_and_layout();
    let (ids_sto, ids_l) = indices.storage_and_layout();

    match (&*x_sto, &*ids_sto) {
        (Storage::Cuda(_), Storage::Cuda(_)) => {
            let xs = x_l.stride();
            let is = ids_l.stride();
            if xs[xs.len() - 1] != 1 || is[is.len() - 1] != 1 {
                return x.index_select(indices, dim);
            }
        }
        _ => return x.index_select(indices, dim),
    }

    let op = IndexSelect { dim };
    x.apply_op2_no_bwd(indices, &op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
        let b = 10f32.powi(digits);
        let t = t.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let t = t
            .iter()
            .map(|row| row.iter().map(|v| (v * b).round() / b).collect())
            .collect();
        Ok(t)
    }

    #[test]
    fn test_index_select_matches_candle_f32() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rows = 128;
        let cols = 1024;

        let x = Tensor::randn(0., 1., (rows, cols), &device)?.to_dtype(DType::F32)?;
        let raw_idx: Vec<u32> = (0..rows as u32).flat_map(|i| [i, i]).collect();
        let index_count = raw_idx.len();
        let indices = Tensor::from_vec(raw_idx, index_count, &device)?.to_dtype(DType::U32)?;

        let y_fast = index_select(&x, &indices, 0)?;
        let y_ref = x.index_select(&indices, 0)?;

        assert_eq!(to_vec2_round(&y_fast, 3)?, to_vec2_round(&y_ref, 3)?);
        Ok(())
    }

    #[test]
    fn test_index_select_matches_candle_f16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rows = 128;
        let cols = 1024;

        let x = Tensor::randn(0., 1., (rows, cols), &device)?.to_dtype(DType::F16)?;
        let raw_idx: Vec<u32> = (0..rows as u32).collect();
        let indices = Tensor::from_vec(raw_idx, rows, &device)?.to_dtype(DType::U32)?;

        let y_fast = index_select(&x, &indices, 0)?;
        let y_ref = x
            .to_dtype(DType::F32)?
            .index_select(&indices, 0)?
            .to_dtype(DType::F16)?;

        assert_eq!(
            to_vec2_round(&y_fast.to_dtype(DType::F32)?, 3)?,
            to_vec2_round(&y_ref.to_dtype(DType::F32)?, 3)?
        );
        Ok(())
    }
}
