use core::ffi::{c_int, c_void};

extern "C" {
    /// Launch fast 2D index_select along dim 0.
    ///
    /// - `x`      : pointer to input [rows, cols]
    /// - `indices`: pointer to u32 indices [index_count]
    /// - `dst`    : pointer to output [index_count, cols]
    /// - `rows`   : rows of x
    /// - `cols`   : cols of x
    /// - `index_count`: length of indices
    /// - `multi_processor_count`: SM count (for grid sizing)
    /// - `dtype_code`: 0 = f16, 1 = f32
    pub(crate) fn run_index_select(
        x: *const c_void,
        indices: *const u32,
        dst: *mut c_void,
        rows: u32,
        cols: u32,
        index_count: u32,
        multi_processor_count: c_int,
        dtype_code: u32,
    );
}