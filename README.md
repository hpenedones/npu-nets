# TileFlow: Spatial Neural Networks on AMD XDNA 2 NPU

TileFlow is an experimental project for hardware-software co-design on the
AMD Ryzen AI NPU (XDNA 2 / Strix Point architecture). It uses the
[IRON/MLIR-AIE](https://github.com/amd/IRON) toolchain to program the NPU
at the individual tile level — explicitly mapping neural network layers to
physical compute tiles and wiring them together with hardware data streams.

The goal: design a neural network architecture that maps **exactly** to the
NPU's 2D tile array and demonstrate inference throughput approaching the
chip's theoretical **25 TFLOPS** (bfloat16) peak — orders of magnitude faster
than CPU execution of the same network. The network can be any architecture
with learnable parameters and non-linearities — we design the network to
match the hardware, not the other way around.

## The Hardware

The XDNA 2 NPU in the Ryzen AI 9 HX 370 is a **spatial dataflow computer**:

```
         Col 0    Col 1    Col 2    Col 3    Col 4    Col 5    Col 6    Col 7
        ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
Row 5   │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │
        │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
        ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 4   │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │
        │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
        ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 3   │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │
        │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
        ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 2   │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │
        │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │ Tile   │
        ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 1   │MemTile │MemTile │MemTile │MemTile │MemTile │MemTile │MemTile │MemTile │
        │ 512 KB │ 512 KB │ 512 KB │ 512 KB │ 512 KB │ 512 KB │ 512 KB │ 512 KB │
        ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 0   │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │
        │  (DMA) │  (DMA) │  (DMA) │  (DMA) │  (DMA) │  (DMA) │  (DMA) │  (DMA) │
        └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

| Property | Value |
|---|---|
| Compute tiles | 32 (8 columns × 4 rows, rows 2–5) |
| Memory tiles | 8 (row 1, 512 KB each, 4 MB total) |
| Shim tiles | 8 (row 0, DMA interface to host DDR) |
| Per-tile SRAM | ~64 KB data memory |
| Per-tile compute | bf16 MMUL unit (VLIW+SIMD) |
| Clock | ~1.5 GHz |
| Peak throughput | **25 TFLOPS** (bfloat16) |
| Interconnect | ObjectFIFOs — depth-2 double-buffered tile-to-tile streams |
| Power | ~6 W |

## Phase 1 Results: Peak Throughput Benchmark

Single large GEMM benchmark (bfloat16) using IRON's AIEGEMM operator:

| Configuration | NPU Latency | NPU TFLOPS | CPU TFLOPS | Speedup |
|---|---|---|---|---|
| 2048³, 1 column | 48.1 ms | 0.36 | — | — |
| 2048³, 2 columns | 26.4 ms | 0.65 | — | — |
| 2048³, 8 columns | 7.2 ms | **2.38** | 1.83 | 1.3× |
| 4096³, 8 columns | 55.1 ms | **2.49** | 1.83 | 1.4× |

**Peak NPU: 2.49 TFLOPS** (10% of theoretical 25 TFLOPS). The modest speedup
over CPU for a single large GEMM is because the operation is **memory-bandwidth
limited** — data must stream from DDR through memory tiles into compute tiles.

**Key insight**: The NPU wins when data **stays on-chip** between operations.
A spatial pipeline avoids DDR round-trips — this is where massive speedup
should come from.

## Phase 2 Results: Spatial Pipeline MLP

### Architecture: 4-Stage Pipelined MLP × 8 Parallel Pipelines

A 4-layer MLP mapped to the 4×8 compute tile grid. Each tile runs one fused
matmul+ReLU layer; data flows through ObjectFIFOs and never returns to DDR:

```
Column 0        Column 1        ...  Column 7
(pipeline 0)    (pipeline 1)         (pipeline 7)
┌───────────┐   ┌───────────┐        ┌───────────┐
│ Row 2     │   │ Row 2     │        │ Row 2     │
│ MatMul₁   │   │ MatMul₁   │   ...  │ MatMul₁   │  Stage 1
│ + ReLU    │   │ + ReLU    │        │ + ReLU    │
├───────────┤   ├───────────┤        ├───────────┤
│ Row 3     │   │ Row 3     │        │ Row 3     │
│ MatMul₂   │   │ MatMul₂   │   ...  │ MatMul₂   │  Stage 2
│ + ReLU    │   │ + ReLU    │        │ + ReLU    │
├───────────┤   ├───────────┤        ├───────────┤
│ Row 4     │   │ Row 4     │        │ Row 4     │
│ MatMul₃   │   │ MatMul₃   │   ...  │ MatMul₃   │  Stage 3
│ + ReLU    │   │ + ReLU    │        │ + ReLU    │
├───────────┤   ├───────────┤        ├───────────┤
│ Row 5     │   │ Row 5     │        │ Row 5     │
│ MatMul₄   │   │ MatMul₄   │   ...  │ MatMul₄   │  Stage 4
│ (output)  │   │ (output)  │        │ (output)  │
└───────────┘   └───────────┘        └───────────┘
    ↑               ↑                     ↑
  input₀          input₁              input₇
```

- **8 columns** = 8 independent pipelines (same weights, different samples)
- **4 rows** = 4 pipeline stages (one MLP layer each)
- **Hidden dim** = 128 (weights 32 KB, fits in 64 KB tile SRAM)
- **Batch** = 16 per pipeline (4 KB I/O buffers, double-buffered)
- **Total parameters**: 4 × 128² = 65,536 (learnable, with ReLU non-linearities)

### Benchmark Results

| Metric | NPU (32 tiles) | CPU (24 cores, PyTorch bf16) |
|---|---|---|
| Latency | 127 µs | 111 µs |
| Throughput | 133 GFLOPS | 152 GFLOPS |
| Inference rate | 1.01M samples/sec | 1.16M samples/sec |
| Correctness | 79.4% (bf16 rounding across 4 layers) | — |

**Speedup: 0.87×** — the CPU is faster for this workload.

### Why the NPU Doesn't Win (Yet)

The theoretical compute for 128 samples through 4 layers of 128×128 matmuls is
16.8M FLOPs — which the NPU can execute in **0.67 µs** at 25 TFLOPS peak.
But the measured latency is **127 µs**, meaning **99% of the time is XRT/DMA
overhead** (kernel launch, instruction dispatch, DMA setup, synchronization).

```
Theoretical compute:  0.67 µs  ( 1% of total)
Driver/DMA overhead: ~126 µs   (99% of total)
────────────────────────────────────────────
Measured latency:     127 µs
```

The 64 KB tile SRAM limits us to H=128, B=16 — too small to overcome the
per-invocation overhead. For the NPU to show advantage, compute must dominate
overhead. The Phase 1 GEMM benchmark confirms this: a 4096³ matmul (137B FLOPs)
achieves 2.49 TFLOPS because compute (55 ms) >> overhead (~0.1 ms).

### What's Next

To demonstrate meaningful speedup, we need to increase compute per invocation:
1. **INT8 kernels**: 50 TOPS peak (2× bf16), weights half the size → H=256 could fit
2. **Memory tile staging**: Use 512 KB memory tiles to double-buffer larger weight
   matrices, allowing H>128 while keeping tile SRAM within budget
3. **Multi-batch streaming**: Process hundreds of batches per invocation to
   amortize the ~126 µs overhead across more useful compute
4. **Larger pipeline**: Chain more operations (e.g., attention + MLP) to
   increase on-chip compute before touching DDR

## Toolchain

| Component | Role |
|---|---|
| [IRON](https://github.com/amd/IRON) | Python API for tile layout + dataflow |
| [MLIR-AIE](https://github.com/Xilinx/mlir-aie) | MLIR dialect → hardware compilation |
| [Peano/LLVM-AIE](https://github.com/Xilinx/llvm-aie) | C++ compiler for per-tile kernels |
| [XRT](https://github.com/amd/xdna-driver) | Runtime for loading/executing on NPU |

## Project Phases

- [x] **Phase 0 — Toolchain Setup**: IRON installed, AXPY/GEMM/RELU tests all pass.
- [x] **Phase 1 — Peak Throughput**: GEMM benchmark on all 8 columns.
  Peak: 2.49 TFLOPS bf16 (10% of theoretical).
- [x] **Phase 2 — Spatial Pipeline MLP**: 4-layer pipelined MLP on 4×8 grid.
  All 32 tiles active, correct results, but overhead-dominated at H=128.
- [ ] **Phase 3 — Scale Up**: INT8 kernels, memory tile staging, or multi-batch
  streaming to increase compute-to-overhead ratio and achieve meaningful speedup.
- [ ] **Phase 4 — Training & Applications**: Backprop on NPU, pick real ML task.

## Hardware Requirements

- **Processor**: AMD Ryzen AI 9 HX 370 (or any XDNA 2 / Strix Point APU)
- **OS**: Linux, kernel 6.11+ with `amdxdna` driver
- **NPU device**: `/dev/accel/accel0` must be accessible
- **Runtime**: XRT (built from [xdna-driver](https://github.com/amd/xdna-driver))

## References

- [IRON repo](https://github.com/amd/IRON) — close-to-metal NPU programming
- [MLIR-AIE programming guide](https://github.com/Xilinx/mlir-aie/tree/main/programming_guide)
- [NPU training (arXiv)](https://arxiv.org/html/2504.03083v1) — backprop on AIE tiles
- [Linux kernel NPU docs](https://docs.kernel.org/accel/amdxdna/amdnpu.html)
- [IRON tutorial (IPDPS 2025)](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf)
