# NPU-Native Neural Networks

`main` is now the curated paper branch for one story: a residual MLP designed
for the AMD XDNA 2 NPU, deployed as a forward-only conveyor-belt pipeline,
and evaluated on the HIGGS dataset where both throughput and latency matter.

Older MNIST, CIFAR, convnet, and full backward-pass experiments live on the
`experimental` branch and remain documented in `logbook.md`.

## What this branch keeps

- HIGGS data preparation and normalization
- CPU/GPU training for the residual MLP
- MLflow + Optuna tuning for full-data HIGGS runs
- Forward-only streaming NPU inference for the residual body
- The whitepaper and hardware notes for the HIGGS paper path

## Headline results

| Result | Value |
| --- | --- |
| Best throughput point | `H=32, L=8`, CPU head, about **4.18M samples/s** wall-clock |
| Best full-data manual run | `H=32, L=32`, 20-epoch schedule, **76.98%** test acc., **0.8542** ROC AUC |
| Best validation-selected tuning result | `H=64, L=32`, **77.98%** test acc., **0.8653** ROC AUC, **0.8770** PR AUC |
| Target hardware | AMD Ryzen AI 9 HX 370 / XDNA 2 |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For AMD GPU training, install a ROCm-enabled PyTorch wheel instead of the
default CPU build, for example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

For the XDNA2 NPU path you also need the AMD runtime/toolchain stack:

```bash
python -m pip install -e /path/to/IRON
source /opt/xilinx/xrt/setup.sh
```

`requirements.txt` covers the pip-installable Python dependencies kept on
`main`. The hardware path still depends on a working XRT installation and an
editable `IRON` checkout.

## Quick start

### 1. Prepare the full HIGGS cache

```bash
python -m resmlp.prepare_higgs_cache \
  --data-dir data/higgs_full \
  --train-splits train \
  --test-splits test
```

This produces a split-aware `data/higgs_full/HIGGS.pt` cache from the public
`jxie/higgs` Hugging Face mirror.

### 2. Train a strong HIGGS model on GPU

```bash
python -m resmlp.train \
  --data-dir data/higgs_full \
  --device cuda \
  --epochs 50 \
  --save-dir build/higgs_h64_l32
```

The curated defaults are already pointed at the current strong HIGGS region:
`H=64`, `L=32`, AdamW, cosine decay, moderate label smoothing, and full-data
validation.

### 3. Launch an MLflow + Optuna sweep

```bash
python -m resmlp.tune_higgs_optuna \
  --data-dir data/higgs_full \
  --device cuda \
  --study-name higgs-full-exploit \
  --experiment-name higgs-full-optuna
```

MLflow logs go under `mlruns/` and the Optuna study state lives in
`build/higgs_optuna.db`.

### 4. Benchmark the conveyor-belt NPU path

```bash
python -m resmlp.streaming_infer build/higgs_h64_l32/resmlp_best.pt \
  --data-dir data/higgs_full \
  --batch-size 48 \
  --num-cols 8 \
  --stream-depth 32 \
  --bench-samples 50000000
```

Use a smaller `--num-cols` value for shallower checkpoints. For the strongest
throughput point (`H=32, L=8`) the main branch uses the residual body on NPU
and keeps the tiny classifier head on CPU.

## Repository layout

```text
resmlp/
├── model.py                # Residual MLP used for HIGGS training and inference
├── data_utils.py           # HIGGS-only data loading and split helpers
├── prepare_higgs_cache.py  # Public-data cache materialization
├── train.py                # CPU/GPU training entry point
├── tune_higgs_optuna.py    # MLflow + Optuna HIGGS sweeps
├── streaming_design.py     # IRON conveyor-belt MLIR generator
├── streaming_op.py         # XRT operator wrapper for the NPU body
└── streaming_infer.py      # HIGGS evaluation / throughput benchmark CLI

aie_kernels/
└── matmul_relu_skip.cc     # Forward residual kernel used by the conveyor belt

tests/
├── test_higgs_data.py          # Native-width HIGGS data regression tests
└── test_streaming_inference.py # Streaming residual operator correctness test

docs/
├── whitepaper.tex          # Paper-focused source
├── whitepaper.pdf          # Built whitepaper
└── xdna2_hardware.png      # Hardware figure used in the paper
```

## Why HIGGS?

CIFAR-10 and MNIST were useful bring-up tasks, but HIGGS is the branch's real
target workload: a dense tabular classification problem with a direct link to
high-energy-physics event filtering. That makes high-throughput inference much
easier to defend as a systems result, rather than just a toy benchmark.

## Historical material

If you need the earlier MNIST, CIFAR, convnet, or full backward-pass work,
switch to `experimental` or read `logbook.md`.
