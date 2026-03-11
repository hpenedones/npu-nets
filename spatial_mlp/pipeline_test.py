#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test and benchmark the 4-stage pipeline MLP on the NPU.

Each column has 4 tiles with different weights.  Data flows through the
pipeline: h = ReLU(RMSNorm(ReLU(RMSNorm(ReLU(RMSNorm(ReLU(RMSNorm(h)@W0))@W1))@W2))@W3).

Each stage fuses RMSNorm + matmul + ReLU, preventing activation explosion
and matching the trained char-LM architecture.

Usage::

    source /opt/xilinx/xrt/setup.sh
    python -m spatial_mlp.pipeline_test [--cols 8] [--B 48]
"""

import os
import sys
import time
import argparse
from pathlib import Path

IRON_DIR = os.environ.get("IRON_DIR", str(Path.home() / "source" / "IRON"))
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)


def _setup_environment():
    sys.path.insert(0, IRON_DIR)
    os.chdir(IRON_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)


_setup_environment()

import numpy as np
from ml_dtypes import bfloat16
from spatial_mlp import to_tiled, from_tiled
from spatial_mlp.pipeline_op import AIEPipelineMLP, STAGES_PER_COL

H = 128


def reference_pipeline(X, weights, scale, num_stages):
    """NumPy reference: apply 4 stages of ReLU(RMSNorm(x, scale) @ W_i)."""
    x = X.astype(np.float32)
    scale_f32 = scale.astype(np.float32)
    for i in range(num_stages):
        # RMSNorm
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        x = x / rms * scale_f32
        # Matmul + ReLU
        W = weights[i].astype(np.float32)
        x = np.maximum(x @ W, 0)
    return x.astype(bfloat16)


def generate_test_data(H, B, num_cols, seed=42):
    """Create random weights, scale, and input for benchmarking."""
    rng = np.random.default_rng(seed)
    total_stages = num_cols * STAGES_PER_COL
    total_samples = num_cols * B

    # Scale weights to keep activations stable through 4 stages
    weights = []
    for _ in range(total_stages):
        W = (rng.standard_normal((H, H)) * 0.8 / np.sqrt(H)).astype(bfloat16)
        weights.append(W)

    # RMSNorm scale vector (shared across all stages, like pre_norm.scale)
    scale = np.ones(H, dtype=bfloat16)

    X = rng.standard_normal((total_samples, H)).astype(bfloat16)
    return weights, scale, X


def tile_weights_for_npu(weights, scale, H, num_cols):
    """Tile and concatenate weights+scale in host buffer layout.

    Layout: [col0_W0+s, col0_W1+s, col0_W2+s, col0_W3+s, col1_W0+s, ...].
    Each entry is [W_tiled(H×H), scale(H)] contiguous.

    For the pipeline test, all columns use the SAME 4 weights (so we can
    compare to a single reference pipeline).
    """
    parts = []
    for col in range(num_cols):
        for stage in range(STAGES_PER_COL):
            W = weights[stage]
            parts.append(np.concatenate([to_tiled(W), scale]))
    return np.concatenate(parts)


def tile_activations_for_npu(X, B, num_cols):
    """Tile activations: [col0_batch, col1_batch, ...] each B×H."""
    parts = []
    for col in range(num_cols):
        batch = X[col * B : (col + 1) * B]
        parts.append(to_tiled(batch))
    return np.concatenate(parts)


def untile_activations_from_npu(flat, B, H, num_cols):
    """Untile activations back to (num_cols × B, H)."""
    parts = []
    for col in range(num_cols):
        start = col * B * H
        end = start + B * H
        parts.append(from_tiled(flat[start:end], B, H))
    return np.concatenate(parts, axis=0)


def benchmark(H=128, B=48, num_cols=8, warmup=3, timed_iters=10):
    """Run the pipeline MLP on NPU and compare to CPU reference."""
    from iron.common.aie_context import AIEContext

    total_samples = num_cols * B
    stages = STAGES_PER_COL  # 4 stages per column
    flops_per_stage = total_samples * 2 * H * H
    total_flops = stages * flops_per_stage

    print(f"\n{'='*70}")
    print(f"Pipeline MLP Benchmark")
    print(f"  Pipeline:   {stages} stages per column "
          f"(ReLU(x @ W) × {stages})")
    print(f"  Columns:    {num_cols}")
    print(f"  Tiles:      {num_cols * stages} "
          f"({num_cols} cols × {stages} rows)")
    print(f"  Batch:      {B}/col × {num_cols} cols = "
          f"{total_samples} samples")
    print(f"  Weights:    {stages} × {H}×{H} per column "
          f"= {stages * H * H * 2 / 1024:.0f} KB/col")
    print(f"  FLOPs:      {total_flops/1e9:.4f} GFLOP per invocation")
    print(f"{'='*70}")

    # Generate test data
    weights, scale, X = generate_test_data(H, B, num_cols)

    # CPU reference (apply stages 0-3 to each column's batch)
    print("\nComputing CPU reference...")
    Y_ref_parts = []
    for col in range(num_cols):
        batch = X[col * B : (col + 1) * B]
        Y_ref_parts.append(reference_pipeline(batch, weights[:4], scale, stages))
    Y_ref = np.concatenate(Y_ref_parts, axis=0)

    # Tile data for NPU
    X_tiled = tile_activations_for_npu(X, B, num_cols)
    W_tiled = tile_weights_for_npu(weights, scale, H, num_cols)
    zero_output = np.zeros(total_samples * H, dtype=bfloat16)

    # Compile and run on NPU
    ctx = AIEContext()
    op = AIEPipelineMLP(H=H, B=B, num_cols=num_cols, context=ctx)
    print("Compiling for NPU...")
    ctx.compile_all()
    ctx.prepare_runtime()

    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        op.write_buffer("input", X_tiled)
        op.write_buffer("weights", W_tiled)
        op.write_buffer("output", zero_output.copy())
        op.run_runlist()

    # Timed runs
    print(f"Timed ({timed_iters} runs)...")
    npu_times = []
    for _ in range(timed_iters):
        op.write_buffer("input", X_tiled)
        op.write_buffer("weights", W_tiled)
        op.write_buffer("output", zero_output.copy())
        elapsed = op.run_runlist()
        npu_times.append(elapsed)

    Y_flat = op.read_buffer("output", (total_samples * H,), copy=True)
    Y_npu = untile_activations_from_npu(Y_flat, B, H, num_cols)

    # Correctness
    ref_f32 = Y_ref.astype(np.float32)
    npu_f32 = Y_npu.astype(np.float32)
    nonzero = np.abs(ref_f32) > 1e-6
    if nonzero.sum() > 0:
        median_rel_err = float(np.median(
            np.abs(ref_f32[nonzero] - npu_f32[nonzero])
            / np.abs(ref_f32[nonzero])))
    else:
        median_rel_err = float('nan')

    close = np.isclose(ref_f32, npu_f32, rtol=0.3, atol=0.01)
    pct_close = 100 * close.mean()

    # CPU benchmark
    import torch
    x_t = torch.from_numpy(X.astype(np.float32)).to(torch.bfloat16)
    ws_t = [torch.from_numpy(w.astype(np.float32)).to(torch.bfloat16)
            for w in weights[:4]]

    # Warmup
    for _ in range(5):
        h = x_t.clone()
        for col in range(num_cols):
            batch = h[col * B : (col + 1) * B]
            for w in ws_t:
                batch = torch.relu(batch @ w)

    cpu_runs = max(10, 100)
    t0 = time.perf_counter()
    for _ in range(cpu_runs):
        h = x_t.clone()
        for w in ws_t:
            h = torch.relu(h @ w)
    cpu_latency = (time.perf_counter() - t0) / cpu_runs

    npu_avg = np.mean(npu_times)
    npu_std = np.std(npu_times)

    npu_gflops = total_flops / npu_avg / 1e9
    cpu_gflops = total_flops / cpu_latency / 1e9

    print(f"\n--- Results ---")
    print(f"  Correctness:     {close.sum()}/{close.size} ({pct_close:.1f}%)")
    print(f"  Median rel err:  {median_rel_err:.4f}")
    print(f"  NPU latency:     {npu_avg*1e3:.3f} ± {npu_std*1e3:.3f} ms")
    print(f"  NPU throughput:  {npu_gflops:.1f} GFLOPS")
    print(f"  CPU latency:     {cpu_latency*1e3:.3f} ms")
    print(f"  CPU throughput:  {cpu_gflops:.1f} GFLOPS")
    print(f"  Speedup:         {cpu_latency / npu_avg:.1f}×")
    print(f"  Samples/sec:     {total_samples / npu_avg:,.0f}")
    print(f"{'='*70}")

    # Print sample values for debugging
    print(f"\nSample values (first 5 elements of col 0):")
    print(f"  Reference: {Y_ref[0, :5]}")
    print(f"  NPU:       {Y_npu[0, :5]}")

    return pct_close > 50  # basic correctness check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test pipeline MLP on NPU.")
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--B", type=int, default=48)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=10)
    args = parser.parse_args()

    success = benchmark(
        H=args.H, B=args.B, num_cols=args.cols,
        warmup=args.warmup, timed_iters=args.timed,
    )
    sys.exit(0 if success else 1)
