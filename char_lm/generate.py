#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate text using the trained character LM — on CPU or NPU.

Usage::

    # CPU inference (baseline)
    python -m char_lm.generate --device cpu --prompt "To be"

    # NPU inference — 32-tile pipeline (exact match to trained model)
    python -m char_lm.generate --device npu --prompt "To be"

The NPU path uses all 32 compute tiles arranged as 8 columns × 4 rows.
Each NPU call processes one 4-layer block of matmul+ReLU.  Between calls
the CPU applies RMSNorm, adds the character embedding and block bias,
and does the residual connection — exactly matching the trained model.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from ml_dtypes import bfloat16

from char_lm.train import load_checkpoint
from char_lm.data import Vocabulary

DATA_DIR = Path(__file__).parent.parent / "data"

# ── NPU configuration ────────────────────────────────────────────────────
H = 128
B = 48
NUM_COLS = 8
STAGES_PER_COL = 4
TOTAL_SAMPLES = NUM_COLS * B  # 384 parallel sequences


def _generate_cpu(
    model, vocab: Vocabulary, prompt: str, num_chars: int, temperature: float
) -> tuple[str, float]:
    """Generate text using pure CPU inference. Returns (text, elapsed_s)."""
    model.eval()
    prompt_ids = torch.tensor([vocab.encode(prompt)])

    t0 = time.perf_counter()
    generated = model.generate(
        prompt_ids, num_chars=num_chars, temperature=temperature
    )
    elapsed = time.perf_counter() - t0

    return vocab.decode(generated), elapsed


def _ensure_iron_env():
    """Set up IRON environment: chdir to IRON root, source XRT."""
    iron_dir = os.environ.get(
        "IRON_DIR", str(Path.home() / "source" / "IRON")
    )
    os.chdir(iron_dir)

    xrt_setup = "/opt/xilinx/xrt/setup.sh"
    if "XILINX_XRT" not in os.environ and Path(xrt_setup).exists():
        import subprocess
        env = subprocess.check_output(
            f"source {xrt_setup} && env", shell=True, text=True,
            executable="/bin/bash"
        )
        for line in env.strip().split("\n"):
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ[k] = v


def _setup_npu():
    """Compile pipeline NPU operator (32 tiles: 8 cols × 4 rows)."""
    _ensure_iron_env()
    from iron.common.aie_context import AIEContext
    from spatial_mlp.pipeline_op import AIEPipelineMLP

    ctx = AIEContext()
    op = AIEPipelineMLP(H=H, B=B, num_cols=NUM_COLS, context=ctx)
    print("Compiling NPU design (cached if unchanged)...")
    ctx.compile_all()
    ctx.prepare_runtime()
    print("NPU ready (32 tiles, 8 cols × 4 rows).")
    return op


def _rms_norm_np(x: np.ndarray, scale: np.ndarray, eps: float = 1e-6):
    """RMSNorm in float32, returns bfloat16."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * scale).astype(bfloat16)


def _generate_npu(
    model, vocab: Vocabulary, prompt: str, num_chars: int,
    temperature: float, npu_op
) -> tuple[str, float, float]:
    """Generate text using NPU pipeline — exact match to trained model.

    For each character step:
        for each block g:
            CPU:  h_b = RMSNorm(h) + embed + bias_g
            NPU:  h_b = ReLU(..ReLU(h_b @ W[4g]) @ W[4g+1]..) [4 layers]
            CPU:  h = h + h_b   (residual)
        CPU:  h = RMSNorm(h)   (post-norm)
        CPU:  logits = h @ readout   (sampling)

    Returns (text, total_elapsed_s, npu_compute_s).
    """
    from spatial_mlp import to_tiled, from_tiled

    model.eval()
    num_blocks = model.num_blocks

    # Extract model parameters as numpy arrays
    embed_w = model.embed.weight.data.numpy().astype(bfloat16)
    readout_w = model.readout.weight.data.numpy().astype(np.float32)
    readout_b = model.readout.bias.data.numpy().astype(np.float32)
    pre_norm_scale = model.pre_norm.scale.data.numpy().astype(np.float32)
    post_norm_scale = model.post_norm.scale.data.numpy().astype(np.float32)
    block_biases = [b.data.numpy().astype(bfloat16) for b in model.block_biases]

    # Pre-tile all weight groups
    # Host layout: [col0_W0, col0_W1, col0_W2, col0_W3, col1_W0, ...]
    W_tiled_groups = []
    for g in range(num_blocks):
        parts = []
        for col in range(NUM_COLS):
            for stage in range(STAGES_PER_COL):
                layer_idx = g * STAGES_PER_COL + stage
                W_bf16 = model.weights[layer_idx].data.numpy().astype(bfloat16)
                parts.append(to_tiled(W_bf16))
        W_tiled_groups.append(np.concatenate(parts))

    prompt_ids = vocab.encode(prompt)
    hidden = np.zeros((TOTAL_SAMPLES, H), dtype=bfloat16)
    output_zeros = np.zeros(TOTAL_SAMPLES * H, dtype=bfloat16)

    generated = list(prompt_ids)
    npu_total = 0.0
    t0 = time.perf_counter()

    def _run_one_block(hidden, embed_vec, block_idx):
        """Run one block: CPU pre-processing → NPU matmul chain → CPU residual."""
        # CPU: h_b = RMSNorm(h) + embed + bias
        h_block = _rms_norm_np(hidden, pre_norm_scale)
        h_block = h_block + embed_vec[np.newaxis, :] + block_biases[block_idx][np.newaxis, :]

        # NPU: 4 layers of matmul+ReLU
        input_tiled = np.concatenate([
            to_tiled(h_block[c * B : (c + 1) * B])
            for c in range(NUM_COLS)
        ])
        npu_op.write_buffer("input", input_tiled)
        npu_op.write_buffer("weights", W_tiled_groups[block_idx])
        npu_op.write_buffer("output", output_zeros.copy())
        npu_time = npu_op.run_runlist()
        out_flat = npu_op.read_buffer(
            "output", (TOTAL_SAMPLES * H,), copy=True
        )
        h_block_out = np.concatenate([
            from_tiled(out_flat[c * B * H : (c + 1) * B * H], B, H)
            for c in range(NUM_COLS)
        ])

        # CPU: residual
        hidden = (hidden.astype(np.float32) + h_block_out.astype(np.float32)).astype(bfloat16)
        return hidden, npu_time

    def _run_all_blocks(hidden, char_idx):
        """Run all blocks for one character step."""
        npu_time = 0.0
        embed_vec = embed_w[char_idx]
        for g in range(num_blocks):
            hidden, dt = _run_one_block(hidden, embed_vec, g)
            npu_time += dt
        # Post-norm
        hidden = _rms_norm_np(hidden, post_norm_scale)
        return hidden, npu_time

    # Process prompt
    for char_idx in prompt_ids:
        hidden, dt = _run_all_blocks(hidden, char_idx)
        npu_total += dt

    # Generate new characters
    for step in range(num_chars):
        h_f32 = hidden[0].astype(np.float32)
        logits = (h_f32 @ readout_w.T + readout_b) / temperature
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)
        generated.append(next_idx)
        hidden, dt = _run_all_blocks(hidden, next_idx)
        npu_total += dt

    total_elapsed = time.perf_counter() - t0
    return vocab.decode(generated), total_elapsed, npu_total


def main():
    parser = argparse.ArgumentParser(description="Generate text with char LM")
    parser.add_argument("--device",
                        choices=["cpu", "npu"],
                        default="cpu",
                        help="Inference device (default: cpu)")
    parser.add_argument("--prompt", type=str, default="To be, or not to be",
                        help="Text prompt to start generation")
    parser.add_argument("--num-chars", type=int, default=200,
                        help="Number of characters to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--checkpoint", type=str,
                        default=str(DATA_DIR / "charlm_checkpoint.pt"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        print("Run `python -m char_lm.train` first.")
        sys.exit(1)

    model, vocab = load_checkpoint(ckpt_path)
    params = model.count_parameters()
    device_label = args.device.upper()

    print("=" * 60)
    print(f"TileFlow Character LM — Generate ({device_label})")
    print("=" * 60)
    print(f"Model: {model.num_layers} layers, "
          f"{model.num_blocks} blocks of {model.block_size}")
    print(f"Parameters: {params['total']:,}")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Generating {args.num_chars} characters...")
    print()

    if args.device == "cpu":
        text, elapsed = _generate_cpu(
            model, vocab, args.prompt, args.num_chars, args.temperature
        )
        chars_per_sec = args.num_chars / elapsed
        print(text)
        print()
        print(f"CPU: {elapsed:.3f}s, {chars_per_sec:.0f} chars/s")

    elif args.device == "npu":
        npu_op = _setup_npu()
        text, total_elapsed, npu_time = _generate_npu(
            model, vocab, args.prompt, args.num_chars, args.temperature, npu_op
        )
        total_steps = len(args.prompt) + args.num_chars
        chars_per_sec = args.num_chars / total_elapsed
        num_npu_calls = model.num_blocks * total_steps
        print(text)
        print()
        print(f"NPU (32 tiles): {total_elapsed:.3f}s total, "
              f"NPU compute: {npu_time:.3f}s "
              f"({npu_time/total_elapsed*100:.0f}%)")
        print(f"Per character: {total_elapsed/total_steps*1000:.2f}ms "
              f"({model.num_blocks} NPU calls × "
              f"{npu_time/num_npu_calls*1000:.3f}ms)")
        print(f"Throughput: {chars_per_sec:.0f} chars/s (1 seq), "
              f"{TOTAL_SAMPLES * chars_per_sec:.0f} chars/s "
              f"({TOTAL_SAMPLES} parallel seqs)")


if __name__ == "__main__":
    main()
