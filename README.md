# NPU-Native Neural Networks

Neural networks co-designed with the AMD XDNA 2 NPU — a 32-layer residual MLP,
with one layer per tile across 32 NPU tiles, programmed with
[IRON](https://github.com/amd/IRON).

## resmlp: 32-Layer Residual MLP on MNIST

Each NPU tile holds one weight matrix and computes `y = relu(x @ W) + x`.
Data flows through all 32 tiles in a serpentine path. The model trains on CPU
in PyTorch and runs inference on the NPU.

```
Input (784) → Linear → H=160
  → [relu(x @ W_i) + x] × 32 tiles    ← NPU snake pipeline
  → Linear → 10 classes
```

### Results

| Metric | Value |
|--------|-------|
| Parameters | 946K |
| MNIST test accuracy | 97.2% |
| NPU throughput | 24K images/sec |
| NPU latency | 0.33 ms / batch of 8 |

### Quick Start

```bash
# Train on CPU (~45 seconds)
python -m resmlp.train

# Train on NPU (forward+backward+SGD on 32 tiles)
python -m resmlp.train_npu --epochs 10

# Run MNIST inference on NPU
python -m resmlp.infer resmlp/checkpoints/resmlp_epoch009.pt --bench

# Test NPU pipeline correctness
python -m tests.test_training --cols 1
python -m tests.test_inference --cols 1
```

## Project Structure

```
resmlp/
├── __init__.py          # Tiled layout utilities (to_tiled / from_tiled)
├── model.py             # PyTorch model: embed → 32 × ResidualLinear → head
├── train.py             # CPU-only MNIST training
├── train_npu.py         # NPU-accelerated MNIST training
├── infer.py             # NPU inference with trained weights
├── design.py            # IRON snake pipeline (inference: 32 tiles)
├── op.py                # IRON operator wrapper (inference)
├── training_design.py   # IRON training pipeline (fwd + bwd + SGD)
└── training_op.py       # IRON operator wrapper (training)

aie_kernels/
├── matmul_relu_skip.cc  # Fused fwd kernel: c = relu(a @ w) + a
├── residual_backward.cc # Fused bwd kernel: grad + in-place SGD update
└── copy_activation.cc   # SRAM block copy utility

tests/
├── test_inference.py    # Snake pipeline correctness + benchmark
├── test_backward.py     # Single-layer backward validation
├── test_checkpoint.py   # Forward checkpoint probe
└── test_training.py     # Full training pipeline validation (fwd+bwd+SGD)
```

## Requirements

- **NPU**: AMD Ryzen AI (XDNA 2 / Strix Point) — e.g. Ryzen AI 9 HX 370
- **OS**: Linux, kernel 6.11+ with `amdxdna` driver
- **Toolchain**: [IRON](https://github.com/amd/IRON) /
  [XRT](https://github.com/amd/xdna-driver)

## References

- [IRON](https://github.com/amd/IRON) — close-to-metal NPU programming
- [MLIR-AIE programming guide](https://github.com/Xilinx/mlir-aie/tree/main/programming_guide)
- [Development logbook](logbook.md) — full history and technical notes
