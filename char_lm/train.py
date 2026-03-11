#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train the recurrent character-level language model on tiny Shakespeare.

Usage::

    python -m char_lm.train [--depth 500] [--epochs 10] [--lr 1e-3]

The trained checkpoint is saved to ``data/charlm_checkpoint.pt``.
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from char_lm.data import load_shakespeare, Vocabulary
from char_lm.model import RecurrentCharLM

CHECKPOINT_DIR = Path(__file__).parent.parent / "data"


def _clamp_spectral_norm(model: RecurrentCharLM, max_norm: float = 1.0):
    """Project each W to have spectral norm ≤ max_norm.

    Skipped when spectral norm is already small (e.g. residual models
    with scaled init) to avoid SVD convergence issues.
    """
    with torch.no_grad():
        for W in model.weights:
            # Frobenius norm is a cheap upper bound on spectral norm;
            # skip the expensive SVD if Frobenius < max_norm.
            fro = torch.linalg.norm(W, ord='fro')
            if fro <= max_norm:
                continue
            try:
                sigma = torch.linalg.norm(W, ord=2)
                if sigma > max_norm:
                    W.mul_(max_norm / sigma)
            except torch._C._LinAlgError:
                pass  # ill-conditioned; skip this step


def train_epoch(
    model: RecurrentCharLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 50,
    clamp_interval: int = 20,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)  # (batch, seq_len, vocab)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if num_batches % clamp_interval == 0:
            _clamp_spectral_norm(model)

        total_loss += loss.item()
        num_batches += 1

        if num_batches % log_interval == 0:
            avg = total_loss / num_batches
            print(f"    batch {num_batches:4d} | loss {avg:.4f}", flush=True)

    return total_loss / num_batches


@torch.no_grad()
def eval_loss(model: RecurrentCharLM, loader: DataLoader, device: torch.device) -> float:
    """Compute average cross-entropy loss on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(
    model: RecurrentCharLM, vocab: Vocabulary, path: Path, metadata: dict
):
    """Save model weights (always on CPU), vocabulary, and training metadata."""
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "vocab_chars": vocab.chars,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "block_size": model.block_size,
            "bptt_blocks": model.bptt_blocks,
            "vocab_size": len(vocab.chars),
            **metadata,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[RecurrentCharLM, Vocabulary]:
    """Load model and vocabulary from a checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    vocab = Vocabulary(ckpt["vocab_chars"])
    model = RecurrentCharLM(
        vocab_size=ckpt.get("vocab_size", len(vocab.chars)),
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        block_size=ckpt.get("block_size", 4),
        bptt_blocks=ckpt.get("bptt_blocks", None),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, vocab


def main():
    parser = argparse.ArgumentParser(description="Train character LM")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of weight matrices (default: 32)")
    parser.add_argument("--block-size", type=int, default=4,
                        help="Layers per block / NPU pipeline stages (default: 4)")
    parser.add_argument("--bptt-blocks", type=int, default=None,
                        help="Blocks with gradients, None=all (default: all)")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Training sequence length (default: 64)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Training device: cpu, cuda, or auto (default: auto)")
    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("TileFlow Character Language Model — Training")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_ds, val_ds, vocab = load_shakespeare(seq_len=args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    print(f"Vocabulary: {vocab.size} characters")
    print(f"Training:   {len(train_ds)} sequences of length {args.seq_len}")
    print(f"Validation: {len(val_ds)} sequences")
    print()

    # Create model
    model = RecurrentCharLM(
        vocab_size=vocab.size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        block_size=args.block_size,
        bptt_blocks=args.bptt_blocks,
    )
    params = model.count_parameters()
    print(f"Model: {args.num_layers} layers, "
          f"{model.num_blocks} blocks of {args.block_size}, "
          f"hidden={args.hidden_size}")
    print(f"Parameters:")
    print(f"  Embedding:    {params['embedding']:,}")
    print(f"  Recurrent W:  {params['recurrent_W']:,}")
    print(f"  Block biases: {params['recurrent_b']:,}")
    print(f"  Readout:      {params['readout']:,}")
    print(f"  Total:        {params['total']:,}")
    print(f"  BPTT blocks:  {model.bptt_blocks}/{model.num_blocks}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)

    # Training loop
    best_val_loss = float("inf")
    ckpt_path = CHECKPOINT_DIR / "charlm_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_loss = eval_loss(model, val_loader, device)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, vocab, ckpt_path, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            marker = " ← saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

    # Generate a sample (always on CPU for compatibility)
    print("\n" + "=" * 60)
    print("Sample generation (CPU, temperature=0.8):")
    print("=" * 60)
    model_cpu, vocab = load_checkpoint(ckpt_path)
    prompt = "\nTo be, or not to be"
    prompt_ids = torch.tensor([vocab.encode(prompt)])
    generated = model_cpu.generate(prompt_ids, num_chars=200, temperature=0.8)
    print(vocab.decode(generated))
    print()
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
