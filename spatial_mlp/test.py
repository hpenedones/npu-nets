#!/usr/bin/env python3
"""
Benchmark: Spatial MLP pipeline on AMD XDNA 2 NPU vs CPU.

Architecture: 4-layer MLP (128→128 + ReLU each), 8 parallel pipelines.
Uses all 32 compute tiles (4 rows × 8 columns) with data flowing
between pipeline stages via on-chip ObjectFIFOs.
"""

import os
import sys
import time

IRON_DIR = "/home/hpenedones/source/IRON"
sys.path.insert(0, IRON_DIR)
os.chdir(IRON_DIR)

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, "/home/hpenedones/source/npu-spatial-nets")
from spatial_mlp.op import AIESpatialMLP
from spatial_mlp.design import to_tiled, from_tiled


def numpy_reference(input_data, weights_list):
    """Compute the MLP forward pass in NumPy (f32 accumulation)."""
    x = input_data.astype(np.float32)
    for W in weights_list:
        x = np.maximum(x @ W.astype(np.float32), 0)
    return x.astype(bfloat16)


def benchmark(H=128, B=16, num_layers=4, num_pipelines=8,
              warmup=5, timed_iters=50):
    from iron.common.aie_context import AIEContext

    total_samples = num_pipelines * B  # 128

    print(f"\n{'='*70}")
    print(f"Spatial MLP Benchmark")
    print(f"  Network: {num_layers}× Linear({H}→{H}) + ReLU")
    print(f"  Tiles:   {num_layers} rows × {num_pipelines} cols = "
          f"{num_layers * num_pipelines} compute tiles")
    print(f"  Batch:   {B} samples/pipeline × {num_pipelines} pipelines = "
          f"{total_samples} samples/invocation")
    print(f"  Params:  {num_layers * H * H:,} ({num_layers * H * H * 2 / 1024:.0f} KB)")
    total_flops = total_samples * num_layers * 2 * H * H
    print(f"  FLOPs:   {total_flops:,} per invocation")
    print(f"{'='*70}")

    # ── Generate test data ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    weights_list = [(rng.standard_normal((H, H)) / np.sqrt(H)).astype(bfloat16)
                    for _ in range(num_layers)]
    X = rng.standard_normal((total_samples, H)).astype(bfloat16)
    Y_ref = numpy_reference(X, weights_list)

    X_tiled = np.concatenate([to_tiled(X[i*B:(i+1)*B])
                              for i in range(num_pipelines)])
    W_tiled = np.concatenate([to_tiled(W) for W in weights_list])

    # ── NPU ─────────────────────────────────────────────────────────────
    ctx = AIEContext()
    op = AIESpatialMLP(H=H, B=B, num_layers=num_layers,
                       num_pipelines=num_pipelines, num_batches=1,
                       context=ctx)
    print("\nCompiling for NPU...")
    ctx.compile_all()
    ctx.prepare_runtime()

    op.write_buffer("input", X_tiled)
    op.write_buffer("weights", W_tiled)
    op.write_buffer("output", np.zeros(total_samples * H, dtype=bfloat16))

    print(f"Warmup ({warmup} iters)...")
    for _ in range(warmup):
        op.run_runlist()

    print(f"Timed ({timed_iters} iters)...")
    npu_times = []
    for _ in range(timed_iters):
        t = op.run_runlist()
        npu_times.append(t)

    npu_avg = np.mean(npu_times)
    npu_min = np.min(npu_times)
    npu_std = np.std(npu_times)

    # Read output for correctness check
    Y_npu_flat = op.read_buffer("output", (total_samples * H,), copy=True)
    Y_npu = np.concatenate([from_tiled(Y_npu_flat[i*B*H:(i+1)*B*H], B, H)
                            for i in range(num_pipelines)])
    close = np.isclose(Y_ref.astype(np.float32), Y_npu.astype(np.float32),
                       rtol=0.15, atol=0.01)

    npu_gflops = total_flops / npu_avg / 1e9
    npu_sps = total_samples / npu_avg

    print(f"\n--- NPU Results ---")
    print(f"  Correctness: {close.sum()}/{close.size} "
          f"({100*close.mean():.1f}%)")
    print(f"  Latency:     {npu_avg*1e6:.1f} ± {npu_std*1e6:.1f} µs "
          f"(min {npu_min*1e6:.1f} µs)")
    print(f"  Throughput:  {npu_gflops:.1f} GFLOPS")
    print(f"  Inference:   {npu_sps:,.0f} samples/sec")

    # ── CPU (PyTorch bf16) ──────────────────────────────────────────────
    import torch
    x_cpu = torch.from_numpy(X.astype(np.float32)).to(torch.bfloat16)
    w_cpu = [torch.from_numpy(W.astype(np.float32)).to(torch.bfloat16)
             for W in weights_list]

    for _ in range(20):
        y = x_cpu
        for W in w_cpu:
            y = torch.relu(y @ W)

    cpu_iters = 1000
    cpu_start = time.perf_counter()
    for _ in range(cpu_iters):
        y = x_cpu
        for W in w_cpu:
            y = torch.relu(y @ W)
    cpu_elapsed = (time.perf_counter() - cpu_start) / cpu_iters

    cpu_gflops = total_flops / cpu_elapsed / 1e9
    cpu_sps = total_samples / cpu_elapsed

    print(f"\n--- CPU Results (PyTorch bf16, {os.cpu_count()} cores) ---")
    print(f"  Latency:     {cpu_elapsed*1e6:.1f} µs")
    print(f"  Throughput:  {cpu_gflops:.1f} GFLOPS")
    print(f"  Inference:   {cpu_sps:,.0f} samples/sec")

    # ── Summary ─────────────────────────────────────────────────────────
    speedup = npu_sps / cpu_sps
    print(f"\n{'='*70}")
    print(f"  NPU:     {npu_sps:>12,.0f} samples/sec  "
          f"({npu_gflops:6.1f} GFLOPS)  {npu_avg*1e6:6.1f} µs")
    print(f"  CPU:     {cpu_sps:>12,.0f} samples/sec  "
          f"({cpu_gflops:6.1f} GFLOPS)  {cpu_elapsed*1e6:6.1f} µs")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"{'='*70}")

    # ── Analysis ────────────────────────────────────────────────────────
    compute_us = total_flops / 25e12 * 1e6  # at 25 TFLOPS peak
    overhead_us = npu_avg * 1e6 - compute_us
    print(f"\n  Analysis:")
    print(f"    Theoretical compute: {compute_us:.2f} µs (at 25 TFLOPS peak)")
    print(f"    Measured latency:    {npu_avg*1e6:.1f} µs")
    print(f"    Overhead:            {overhead_us:.1f} µs "
          f"({overhead_us/npu_avg/1e6*100:.0f}% of total)")
    print(f"    Compute/overhead:    {compute_us/overhead_us*100:.1f}%")
    print(f"\n  The {num_layers}×({H}×{H}) MLP has only {total_flops:,} FLOPs")
    print(f"  per invocation. The ~{overhead_us:.0f} µs XRT/DMA overhead")
    print(f"  dominates. NPU advantage grows with larger compute workloads.")


if __name__ == "__main__":
    benchmark()
