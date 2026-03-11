#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Interactive text completion with a trained character LM.

Usage::

    python -m char_lm.complete                         # latest checkpoint
    python -m char_lm.complete --checkpoint data/wikipedia_transformer_checkpoint.pt
    python -m char_lm.complete --chars 500 --temperature 0.6
    python -m char_lm.complete --benchmark             # measure chars/s
"""

import argparse
import sys
import time
import torch
from pathlib import Path

from char_lm.data import Vocabulary
from char_lm.model import RecurrentCharLM
from char_lm.transformer_baseline import TransformerCharLM

DATA_DIR = Path(__file__).parent.parent / "data"


def load_any_checkpoint(path: Path):
    """Load either a recurrent or transformer checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    vocab = Vocabulary(ckpt["vocab_chars"])

    if "config" in ckpt:
        # Transformer checkpoint
        cfg = ckpt["config"]
        model = TransformerCharLM(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            max_seq_len=cfg["max_seq_len"],
        )
        kind = f"Transformer {cfg['n_layers']}L {cfg['d_model']}d"
    else:
        # Recurrent checkpoint
        model = RecurrentCharLM(
            vocab_size=ckpt.get("vocab_size", len(vocab.chars)),
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            block_size=ckpt.get("block_size", 4),
            bptt_blocks=ckpt.get("bptt_blocks", None),
        )
        kind = f"Recurrent {ckpt['num_layers']} layers"

    model.load_state_dict(ckpt["model_state"])
    return model, vocab, kind


def run_benchmark(model, vocab, kind, num_chars=500, num_warmup=50):
    """Measure inference throughput in characters/second."""
    device = next(model.parameters()).device
    # Use a fixed prompt (lowercase to avoid shift-encoding issues)
    prompt = "the "
    prompt_ids = torch.tensor([vocab.encode(prompt)], device=device)

    # Warmup
    print(f"Warming up ({num_warmup} chars)...")
    _ = model.generate(prompt_ids, num_chars=num_warmup, temperature=0.8)

    # Timed run
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Generating {num_chars} chars...")
    t0 = time.perf_counter()
    _ = model.generate(prompt_ids, num_chars=num_chars, temperature=0.8)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    chars_per_sec = num_chars / elapsed
    ms_per_char = elapsed / num_chars * 1000
    print(f"\n{'='*50}")
    print(f"Model:      {kind}")
    print(f"Device:     {device}")
    print(f"Generated:  {num_chars} characters")
    print(f"Time:       {elapsed:.3f}s")
    print(f"Throughput: {chars_per_sec:,.0f} chars/s")
    print(f"Latency:    {ms_per_char:.2f} ms/char")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Interactive char LM completion")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--chars", type=int, default=200,
                        help="Characters to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure inference throughput (chars/s)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda)")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Find the most recently modified checkpoint
        candidates = sorted(
            DATA_DIR.glob("*_checkpoint.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Also check legacy names
        for legacy_name in ["charlm_checkpoint.pt", "transformer_baseline.pt"]:
            legacy = DATA_DIR / legacy_name
            if legacy.exists():
                candidates.append(legacy)
        if not candidates:
            print("No checkpoint found in data/.")
            print("Run `python -m char_lm.train` or "
                  "`python -m char_lm.transformer_baseline` first.")
            sys.exit(1)
        ckpt_path = candidates[0]

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, vocab, kind = load_any_checkpoint(ckpt_path)
    model.eval()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {kind}, {params:,} params")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Device: {device}")
    print(f"Vocab: {vocab.size} characters")

    if args.benchmark:
        run_benchmark(model, vocab, kind, num_chars=args.chars)
        return

    print(f"Type a prompt and press Enter. Ctrl-C to quit.\n")

    try:
        while True:
            prompt = input(">>> ")
            if not prompt:
                continue
            # Filter prompt to known vocab chars
            safe = "".join(c for c in prompt if c in vocab.char_to_idx
                           or c.lower() in vocab.char_to_idx
                           or c.isupper())
            prompt_ids = torch.tensor([vocab.encode(safe)])
            generated = model.generate(
                prompt_ids, num_chars=args.chars, temperature=args.temperature
            )
            text = vocab.decode(generated)
            # Print only the generated part (after prompt)
            print(text[len(safe):])
            print()
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
