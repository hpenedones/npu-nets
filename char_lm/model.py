#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Recurrent character-level language model designed to map exactly to NPU.

Architecture (per character step):
    for each block g = 0..num_blocks-1:
        1. Injection:  h_b = h + embed(char) + bias_g                 (CPU)
        2. Block (4 stages, each on one NPU tile):
             h_b = ReLU(RMSNorm(h_b) @ W[4g+j])  for j = 0..3       (NPU)
        3. Residual:   h = h + h_b                                    (CPU)
    4. Post-norm:  h = RMSNorm(h)                                     (CPU)
    5. Readout:    logits = h @ W_out + b_out                         (CPU)

Each pipeline stage fuses RMSNorm + matmul + ReLU.  The norm prevents
activation explosion through the 4-layer chain and is the key difference
from the previous "pure matmul+ReLU" blocks.  CPU work between NPU calls
is limited to embed injection, bias addition, and residual connection.

With block_size=4 and num_layers=32: 8 blocks = 8 NPU calls per character.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class RecurrentCharLM(nn.Module):
    """Character-level language model with block-recurrent core.

    Parameters
    ----------
    vocab_size : int
        Number of distinct characters.
    hidden_size : int
        Dimension of the hidden state and weight matrices (multiple of 8).
    num_layers : int
        Number of distinct weight matrices (must be divisible by block_size).
    block_size : int
        Layers per block.  Each block is a pure matmul+ReLU chain that
        maps to one NPU pipeline call (default: 4 = STAGES_PER_COL).
    bptt_blocks : int
        How many of the final blocks carry gradients (truncated BPTT).
        Default: all blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 32,
        block_size: int = 4,
        bptt_blocks: int | None = None,
    ):
        super().__init__()
        assert num_layers % block_size == 0, (
            f"num_layers={num_layers} must be divisible by block_size={block_size}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_blocks = num_layers // block_size
        self.bptt_blocks = bptt_blocks if bptt_blocks is not None else self.num_blocks

        # Keep depth/bptt_depth for checkpoint compatibility
        self.depth = num_layers
        self.bptt_depth = self.bptt_blocks * block_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size, hidden_size))
            for _ in range(num_layers)
        ])
        self.block_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(self.num_blocks)
        ])
        self.pre_norm = RMSNorm(hidden_size)
        self.post_norm = RMSNorm(hidden_size)
        self.readout = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    @property
    def depth_per_layer(self) -> int:
        return 1

    def _init_weights(self):
        """Initialise weights for stable deep recurrence."""
        for W in self.weights:
            nn.init.orthogonal_(W)
            W.data.mul_(1.0 / (self.num_blocks ** 0.5))
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _apply_block(self, h_block: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Apply one block: per-layer RMSNorm + matmul + ReLU (NPU pipeline).

        Each of the 4 stages normalises its input before the matmul, preventing
        activation explosion and matching the fused norm+mm+relu NPU kernel.
        """
        start = block_idx * self.block_size
        for i in range(start, start + self.block_size):
            h_block = F.relu(self.pre_norm(h_block) @ self.weights[i])
        return h_block

    def _apply_recurrence(
        self, h: torch.Tensor, embed: torch.Tensor
    ) -> torch.Tensor:
        """Apply all blocks with truncated BPTT.

        Each block:
            h_b = h + embed + bias_g            (CPU: injection)
            h_b = norm+mm+relu × 4 layers       (NPU: fused pipeline)
            h = h + h_b                          (CPU: residual)

        Only the last bptt_blocks blocks carry gradients.
        """
        no_grad_blocks = self.num_blocks - self.bptt_blocks

        for g in range(self.num_blocks):
            if g < no_grad_blocks:
                with torch.no_grad():
                    h_block = h + embed + self.block_biases[g]
                    h_block = self._apply_block(h_block, g)
                    h = h + h_block
            else:
                if g == no_grad_blocks:
                    h = h.detach().requires_grad_(True)
                h_block = h + embed + self.block_biases[g]
                h_block = self._apply_block(h_block, g)
                h = h + h_block

        return h

    def forward(
        self, chars: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of characters, predict next char at each step.

        Parameters
        ----------
        chars : (batch, seq_len) long tensor of character indices.
        hidden : (batch, hidden_size) initial hidden state, or None for zeros.

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        hidden : (batch, hidden_size) — final hidden state (detached).
        """
        batch_size, seq_len = chars.shape
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.hidden_size, device=chars.device
            )

        embeds = self.embed(chars)  # (batch, seq_len, hidden_size)
        all_logits = []

        for t in range(seq_len):
            hidden = self._apply_recurrence(hidden, embeds[:, t])
            hidden = self.post_norm(hidden)
            all_logits.append(self.readout(hidden))

        logits = torch.stack(all_logits, dim=1)
        return logits, hidden.detach()

    @torch.no_grad()
    def generate(
        self,
        start_chars: torch.Tensor,
        num_chars: int = 200,
        temperature: float = 0.8,
        hidden: torch.Tensor | None = None,
    ) -> list[int]:
        """Autoregressively generate characters (CPU-only path)."""
        self.eval()
        device = next(self.parameters()).device

        _, hidden = self.forward(start_chars, hidden)

        generated = start_chars[0].tolist()
        current_char = start_chars[0, -1].unsqueeze(0).unsqueeze(0)

        for _ in range(num_chars):
            logits, hidden = self.forward(current_char, hidden)
            probs = F.softmax(logits[0, 0] / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)
            current_char = torch.tensor([[next_idx]], device=device)

        return generated

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts by component."""
        recurrent_w = sum(W.numel() for W in self.weights)
        recurrent_b = sum(b.numel() for b in self.block_biases)
        return {
            "embedding": self.embed.weight.numel(),
            "recurrent_W": recurrent_w,
            "recurrent_b": recurrent_b,
            "num_layers": self.num_layers,
            "readout": sum(p.numel() for p in self.readout.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
