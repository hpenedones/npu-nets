# SPDX-License-Identifier: Apache-2.0
"""
Spatial MLP: 4-layer pipelined MLP on the XDNA 2 NPU.

Architecture:
  - 4 pipeline stages (rows 2-5), each: matmul(H×H) + ReLU
  - 8 parallel pipelines (columns 0-7), same weights, different samples
  - Data flows top-to-bottom via ObjectFIFOs (depth=2 for double buffering)
  - Weights broadcast from host → memory tile → all tiles in each row

Uses IRON's existing mm.cc kernel (tiled data layout, 2×2 block unrolling)
plus a custom in-place ReLU kernel.

With BFP16 emulation enabled: r=s=t=8, so all tile block sizes are 8×8.
This means inter-stage data format (matmul output → next matmul input) is
consistent without any conversion.

Tile SRAM budget per tile (~64 KB):
  - Weights: H×H×2 bytes (bf16)     = 32 KB for H=128
  - Input FIFO:  depth=2 × B×H×2    =  8 KB for B=16, H=128
  - Output FIFO: depth=2 × B×H×2    =  8 KB for B=16, H=128
  - Total:                            = 48 KB ✓
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern


def spatial_mlp(
    H: int = 128,
    B: int = 16,
    num_layers: int = 4,
    num_pipelines: int = 8,
    num_batches: int = 1,
):
    """
    Generate MLIR for a spatial pipelined MLP.

    Uses IRON's mm.cc (tiled layout) for matmul and a custom in-place ReLU.
    Data must be converted to tiled format before DMA transfer.

    With BFP16 emulation: r=s=t=8.
    Constraints: B % 16 == 0, H % 16 == 0 (for 2×2 block unrolling).
    """
    assert 1 <= num_layers <= 4, "NPU2 has 4 compute rows (rows 2-5)"
    assert 1 <= num_pipelines <= 8, "NPU2 has 8 columns"
    assert B % 16 == 0, f"B={B} must be divisible by 16 (2*r with r=8)"
    assert H % 16 == 0, f"H={H} must be divisible by 16 (2*t with t=8)"

    dtype = bfloat16

    # Activation buffer: B×H elements in tiled format
    act_ty = np.ndarray[(B * H,), np.dtype[dtype]]

    # Weight buffer: H×H elements in tiled format
    weight_ty = np.ndarray[(H * H,), np.dtype[dtype]]

    # Host-side tensor types (for DMA)
    input_ty = np.ndarray[(num_pipelines * B * H,), np.dtype[dtype]]
    output_ty = np.ndarray[(num_pipelines * B * H,), np.dtype[dtype]]
    all_weights_ty = np.ndarray[(num_layers * H * H,), np.dtype[dtype]]

    # Kernels from IRON's mm.cc (compiled with DIM_M=B, DIM_K=H, DIM_N=H)
    # zero_bf16(c_out): zeros a DIM_M×DIM_N output buffer
    zero_kernel = Kernel(
        "zero_bf16",
        "mlp_kernels.a",
        [act_ty],
    )

    # matmul_bf16_bf16(a_in, b_in, c_out): C += A @ B in tiled format
    matmul_kernel = Kernel(
        "matmul_bf16_bf16",
        "mlp_kernels.a",
        [act_ty, weight_ty, act_ty],
    )

    # relu_inplace_bf16(buf, size): in-place max(x, 0)
    relu_kernel = Kernel(
        "relu_inplace_bf16",
        "mlp_kernels.a",
        [act_ty, np.int32],
    )

    of_depth = 2

    # ── Input FIFOs: host → first compute row ───────────────────────────
    of_inputs = [
        ObjectFifo(act_ty, name=f"in_{i}", depth=of_depth)
        for i in range(num_pipelines)
    ]

    # ── Inter-stage FIFOs: between compute rows ─────────────────────────
    of_inter = []
    for layer in range(num_layers - 1):
        of_inter.append([
            ObjectFifo(act_ty, name=f"inter_{layer}_{i}", depth=of_depth)
            for i in range(num_pipelines)
        ])

    # ── Output FIFOs: last compute row → host ───────────────────────────
    of_outputs = [
        ObjectFifo(act_ty, name=f"out_{i}", depth=of_depth)
        for i in range(num_pipelines)
    ]

    # ── Weight FIFOs: broadcast from host to all pipelines per layer ────
    of_weight_in = []
    of_weight_bcast = []
    for layer in range(num_layers):
        w_in = ObjectFifo(weight_ty, name=f"w_in_{layer}", depth=1)
        w_fwd = w_in.cons().forward(
            name=f"w_fwd_{layer}",
            depth=1,
            placement=Tile(col=layer, row=1),
        )
        of_weight_in.append(w_in)
        of_weight_bcast.append(w_fwd)

    # ── Worker function: zero → matmul → relu_inplace ───────────────────
    def mlp_stage(of_in, of_out, of_w, zero_fn, matmul_fn, relu_fn):
        w = of_w.acquire(1)
        for _ in range_(num_batches):
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            zero_fn(y)
            matmul_fn(x, w, y)
            relu_fn(y, B * H)
            of_in.release(1)
            of_out.release(1)
        of_w.release(1)

    # ── Create workers ──────────────────────────────────────────────────
    workers = []
    for layer in range(num_layers):
        for i in range(num_pipelines):
            if layer == 0:
                in_fifo = of_inputs[i].cons()
            else:
                in_fifo = of_inter[layer - 1][i].cons()

            if layer == num_layers - 1:
                out_fifo = of_outputs[i].prod()
            else:
                out_fifo = of_inter[layer][i].prod()

            workers.append(
                Worker(
                    mlp_stage,
                    fn_args=[
                        in_fifo,
                        out_fifo,
                        of_weight_bcast[layer].cons(),
                        zero_kernel,
                        matmul_kernel,
                        relu_kernel,
                    ],
                    placement=Tile(col=i, row=2 + layer),
                )
            )

    # ── DMA access patterns ─────────────────────────────────────────────
    chunk_size = B * H

    # ── Runtime sequence ────────────────────────────────────────────────
    rt = Runtime()
    with rt.sequence(input_ty, all_weights_ty, output_ty) as (inp, weights, out):

        rt.start(*workers)

        # Fill weights: one per layer, broadcast to all pipelines
        tg_w = rt.task_group()
        for layer in range(num_layers):
            weight_tap = TensorAccessPattern(
                (1, num_layers * H * H),
                layer * H * H,
                [1, 1, 1, H * H],
                [0, 0, 0, 1],
            )
            rt.fill(
                of_weight_in[layer].prod(),
                weights,
                weight_tap,
                task_group=tg_w,
            )
        rt.finish_task_group(tg_w)

        # Process data batches
        for _ in range(num_batches):
            tg = rt.task_group()

            for i in range(num_pipelines):
                input_tap = TensorAccessPattern(
                    (1, num_pipelines * B * H),
                    i * chunk_size,
                    [1, 1, 1, chunk_size],
                    [0, 0, 0, 1],
                )
                rt.fill(
                    of_inputs[i].prod(),
                    inp,
                    input_tap,
                    task_group=tg,
                )

            for i in range(num_pipelines):
                output_tap = TensorAccessPattern(
                    (1, num_pipelines * B * H),
                    i * chunk_size,
                    [1, 1, 1, chunk_size],
                    [0, 0, 0, 1],
                )
                rt.drain(
                    of_outputs[i].cons(),
                    out,
                    output_tap,
                    wait=True,
                    task_group=tg,
                )

            rt.finish_task_group(tg)

    dev = NPU2()
    program = Program(dev, rt)
    module = program.resolve_program(SequentialPlacer())
    return module


# ── Tiling format conversion utilities ──────────────────────────────────
# With BFP16 emulation: r=s=t=8. All matrices use 8×8 block tiles.
# Within each tile, elements are stored in row-major order.
# Tiles themselves are stored in row-major order (of block rows/cols).

TILE_SIZE = 8  # r = s = t = 8 for bf16 with BFP16 emulation


def to_tiled(mat, block_r=TILE_SIZE, block_c=TILE_SIZE):
    """Convert row-major matrix to tiled (blocked) layout."""
    M, K = mat.shape
    assert M % block_r == 0 and K % block_c == 0
    return (mat.reshape(M // block_r, block_r, K // block_c, block_c)
            .transpose(0, 2, 1, 3)
            .reshape(-1))


def from_tiled(flat, M, K, block_r=TILE_SIZE, block_c=TILE_SIZE):
    """Convert tiled (blocked) layout back to row-major matrix."""
    assert M % block_r == 0 and K % block_c == 0
    return (flat.reshape(M // block_r, K // block_c, block_r, block_c)
            .transpose(0, 2, 1, 3)
            .reshape(M, K))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=128, help="Hidden dimension")
    p.add_argument("--B", type=int, default=16, help="Batch size per pipeline")
    p.add_argument("--layers", type=int, default=4, help="Number of layers")
    p.add_argument("--pipelines", type=int, default=8, help="Number of pipelines")
    p.add_argument("--batches", type=int, default=1, help="Batches per invocation")
    p.add_argument("-o", "--output", type=str, default="build/spatial_mlp.mlir")
    args = p.parse_args()

    module = spatial_mlp(
        H=args.H, B=args.B, num_layers=args.layers,
        num_pipelines=args.pipelines, num_batches=args.batches,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(str(module))
    print(f"Written to {out_path}")
