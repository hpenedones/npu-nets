#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dataset and vocabulary for character-level language modelling.

**Shift encoding:** Uppercase letters are represented as two characters —
a shift marker ``^`` followed by the lowercase letter.  This halves the
alphabet cost in the vocabulary (26 entries instead of 52) while still
letting the model learn capitalisation patterns.  All digits 0-9 are
included even if absent from the training corpus.

The vocabulary maps each unique character to an integer index.
SequenceDataset produces fixed-length windows of (input, target) pairs
where target[t] = input[t+1] (next-character prediction).
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

SHIFT_CHAR = "^"  # precedes a lowercase letter to mean "uppercase"


def shift_encode(text: str) -> str:
    """Encode uppercase letters as SHIFT_CHAR + lowercase.

    Example: ``"Hello"`` → ``"^hello"``
    """
    out = []
    for c in text:
        if c.isupper():
            out.append(SHIFT_CHAR)
            out.append(c.lower())
        else:
            out.append(c)
    return "".join(out)


def shift_decode(text: str) -> str:
    """Decode SHIFT_CHAR + lowercase back to uppercase.

    Example: ``"^hello"`` → ``"Hello"``
    """
    out = []
    shift = False
    for c in text:
        if c == SHIFT_CHAR:
            shift = True
        elif shift:
            out.append(c.upper())
            shift = False
        else:
            out.append(c)
    return "".join(out)


def _canonical_vocab(text: str) -> list[str]:
    """Build canonical vocabulary: shift-encoded text chars + digits 0-9.

    Always includes all 10 digits even if they don't appear in the text,
    so the model can handle numbers in arbitrary prompts.
    """
    encoded = shift_encode(text)
    chars = set(encoded)
    # Ensure all digits are present
    for d in "0123456789":
        chars.add(d)
    return sorted(chars)


class Vocabulary:
    """Bidirectional mapping between characters and integer indices."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.char_to_idx = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        """Build vocabulary from raw text (applies shift encoding)."""
        return cls(_canonical_vocab(text))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text()))

    def save(self, path: Path):
        path.write_text(json.dumps(self.chars))

    def encode(self, text: str) -> list[int]:
        """Encode raw text (with uppercase) to index sequence."""
        return [self.char_to_idx[c] for c in shift_encode(text)]

    def decode(self, indices) -> str:
        """Decode index sequence back to text (restoring uppercase)."""
        raw = "".join(self.chars[i] for i in indices)
        return shift_decode(raw)

    @property
    def size(self) -> int:
        return len(self.chars)


class SequenceDataset(Dataset):
    """Fixed-length windows over a shift-encoded character corpus.

    Each item is (input_ids, target_ids) where target_ids are shifted
    by one position: target[t] = input[t+1].
    """

    def __init__(self, encoded_text: str, vocab: Vocabulary, seq_len: int = 64):
        indices = [vocab.char_to_idx[c] for c in encoded_text]
        self.data = torch.tensor(indices, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def load_shakespeare(seq_len: int = 64, val_fraction: float = 0.1):
    """Load tiny Shakespeare and return train/val datasets + vocabulary.

    The text is shift-encoded (uppercase → ``^`` + lowercase) before
    building the vocabulary and splitting into sequences.
    """
    raw_text = (DATA_DIR / "shakespeare.txt").read_text()
    vocab = Vocabulary.from_text(raw_text)
    encoded = shift_encode(raw_text)

    split = int(len(encoded) * (1 - val_fraction))
    train_ds = SequenceDataset(encoded[:split], vocab, seq_len)
    val_ds = SequenceDataset(encoded[split:], vocab, seq_len)

    return train_ds, val_ds, vocab
