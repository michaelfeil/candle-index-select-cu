#!/usr/bin/env python3
"""
SPDX-License-Identifier: Apache-2.0 OR MIT
Copyright (c) 2025 Michael Feil

Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
option. This file may not be copied, modified, or distributed
except according to those terms.

Authors explaination: Provide a copy of the first two lines in each redistributed version.


Benchmark PyTorch's index_select on GPU.
Run this alongside the Rust benchmark to compare performance.

Usage:
    python bench_pytorch.py
"""

import torch
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BenchmarkConfig:
    name: str
    shape: Tuple[int, ...]
    out_rows: int


def warmup(device: torch.device, iterations: int = 10):
    """Warmup GPU with some operations."""
    x = torch.randn(1000, 1000, device=device)
    for _ in range(iterations):
        _ = x @ x.T
    torch.cuda.synchronize()

@torch.no_grad()
def benchmark_index_select(
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    warmup_iters: int = 10,
    measure_iters: int = 100,
) -> Tuple[float, float]:
    """Benchmark index_select and return (cpu_time, gpu_time) in seconds."""
    
    # Setup tensors
    x = torch.randn(*config.shape, device=device, dtype=dtype)
    indices = torch.arange(config.out_rows, device=device, dtype=torch.long) % config.shape[0]
    
    # Warmup
    for _ in range(warmup_iters):
        _ = torch.index_select(x, 0, indices)
        torch.cuda.synchronize()
    
    # CPU timing (includes Python overhead)
    cpu_times = []
    for _ in range(measure_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.index_select(x, 0, indices)
        torch.cuda.synchronize()
        end = time.perf_counter()
        cpu_times.append(end - start)
    
    # GPU-only timing using CUDA events (no Python overhead)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    gpu_times = []
    for _ in range(measure_iters):
        start_event.record()
        _ = torch.index_select(x, 0, indices)
        end_event.record()
        torch.cuda.synchronize()
        gpu_times.append(start_event.elapsed_time(end_event) / 1000.0)  # ms -> s
    
    return sum(cpu_times) / len(cpu_times), sum(gpu_times) / len(gpu_times)


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.3f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.3f} ms"
    else:
        return f"{seconds:.3f} s"


def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print()
    
    # Warmup GPU
    warmup(device)
    
    # Benchmark configurations (matching Rust benchmarks)
    configs = [
        BenchmarkConfig("short_2d", (100, 128), 200),
        BenchmarkConfig("medium_2d", (16000, 1024), 12000),
        BenchmarkConfig("long_2d", (16_000, 1024), 70_000),
        BenchmarkConfig("very_long_2d", (100_000, 2048), 500_000),
        BenchmarkConfig("3d", (10, 100, 128), 200),
        BenchmarkConfig("long_3d", (2000, 64, 256), 10_000),
    ]
    
    dtypes = [
        ("F32", torch.float32),
        ("F16", torch.float16),
    ]
    
    print("=" * 80)
    print("PyTorch index_select Benchmark Results")
    print("=" * 80)
    print()
    
    # Markdown table header
    print("| Shape | Out Rows | DType | CPU Time | GPU Time | Overhead |")
    print("|-------|----------|-------|----------|----------|----------|")
    
    results = []
    
    for config in configs:
        for dtype_name, dtype in dtypes:
            try:
                cpu_time, gpu_time = benchmark_index_select(config, device, dtype)
                overhead_pct = ((cpu_time - gpu_time) / gpu_time) * 100
                results.append((config, dtype_name, cpu_time, gpu_time))
                print(f"| {str(config.shape):20} | {config.out_rows:8} | {dtype_name:5} | {format_time(cpu_time):>12} | {format_time(gpu_time):>12} | {overhead_pct:>6.1f}% |")
            except Exception as e:
                print(f"| {str(config.shape):20} | {config.out_rows:8} | {dtype_name:5} | ERROR: {e} |")
    
    print()
    print("=" * 80)
    print("Detailed Results")
    print("=" * 80)
    
    for config, dtype_name, cpu_time, gpu_time in results:
        overhead_pct = ((cpu_time - gpu_time) / gpu_time) * 100
        print(f"\n{config.name} ({dtype_name}):")
        print(f"  Shape: {config.shape}")
        print(f"  Output rows: {config.out_rows}")
        print(f"  CPU time (with Python): {format_time(cpu_time)}")
        print(f"  GPU time (CUDA events): {format_time(gpu_time)}")
        print(f"  Python overhead: {overhead_pct:.1f}%")
        
        # Calculate throughput
        input_elements = 1
        for d in config.shape:
            input_elements *= d
        output_elements = config.out_rows * config.shape[-1] if len(config.shape) == 2 else config.out_rows * config.shape[1] * config.shape[2]
        
        bytes_per_element = 4 if dtype_name == "F32" else 2
        read_bytes = config.out_rows * config.shape[-1] * bytes_per_element if len(config.shape) == 2 else config.out_rows * config.shape[1] * config.shape[2] * bytes_per_element
        write_bytes = read_bytes
        total_bytes = read_bytes + write_bytes
        
        bandwidth_gb_s = (total_bytes / gpu_time) / 1e9
        print(f"  Memory bandwidth: {bandwidth_gb_s:.2f} GB/s")


if __name__ == "__main__":
    main()
