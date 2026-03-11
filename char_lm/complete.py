#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Interactive text completion with the trained character LM.

Usage::

    python -m char_lm.complete
    python -m char_lm.complete --chars 500 --temperature 0.6
"""

import argparse
import sys
import torch
from pathlib import Path

from char_lm.train import load_checkpoint

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Interactive char LM completion")
    parser.add_argument("--checkpoint", type=str,
                        default=str(DATA_DIR / "charlm_checkpoint.pt"))
    parser.add_argument("--chars", type=int, default=200,
                        help="Characters to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run `python -m char_lm.train` first.")
        sys.exit(1)

    model, vocab = load_checkpoint(ckpt_path)
    model.eval()
    print(f"Loaded model: {model.num_layers} layers, {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Type a prompt and press Enter. Ctrl-C to quit.\n")

    try:
        while True:
            prompt = input(">>> ")
            if not prompt:
                continue
            prompt_ids = torch.tensor([vocab.encode(prompt)])
            generated = model.generate(
                prompt_ids, num_chars=args.chars, temperature=args.temperature
            )
            text = vocab.decode(generated)
            # Print only the generated part (after prompt)
            print(text[len(prompt):])
            print()
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
