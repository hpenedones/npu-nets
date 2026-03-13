"""
Train the 32-layer residual MLP on MNIST using the Ryzen AI NPU.

Architecture:
    CPU:  784 → Linear → 160          (embedding)
    NPU:  [y = relu(x @ W_i) + x] × 32  (forward + backward + SGD on NPU)
    CPU:  160 → Linear → 10           (classification head)

Strategy:
  For each batch of B=8 images:
    1. CPU: embed 784 → 160
    2. CPU: forward through 32 residual layers (cheap, needed to compute loss)
    3. CPU: classify 160 → 10, compute cross-entropy loss
    4. CPU: backprop through head → get gy (gradient at output of residual stack)
    5. NPU: one call to the fused training pipeline with (x, weights, gy)
       - NPU forward pass saves checkpoints
       - NPU backward pass computes gradients + updates W in-place via SGD
    6. CPU: update embed + head with Adam

Usage:
    python -m resmlp.train_npu                      # default 10 epochs
    python -m resmlp.train_npu --epochs 20
    python -m resmlp.train_npu --lr-npu 0.001       # NPU SGD learning rate
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ml_dtypes import bfloat16

from iron.common.aie_context import AIEContext
from resmlp import to_tiled, from_tiled
from resmlp.model import ResMLP
from resmlp.training_op import TrainingPipeline
from resmlp.training_design import ROWS_PER_COL


def get_dataloaders(batch_size, data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True,
                              transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2)
    return train_loader, test_loader


def cpu_forward_residual(x_np, weights_bf16):
    """Run the 32 residual layers on CPU in bfloat16-truncated arithmetic.
    
    Returns the final output and saves no state (the NPU will redo this
    with checkpointing during its own forward pass).
    """
    current = x_np.astype(np.float32)
    for w in weights_bf16:
        w_f32 = w.astype(np.float32)
        mm = (current @ w_f32.T).astype(bfloat16).astype(np.float32)
        relu_out = np.maximum(mm, 0)
        current = (current + relu_out).astype(bfloat16).astype(np.float32)
    return current


def main():
    parser = argparse.ArgumentParser(description="Train ResMLP on MNIST with NPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Adam LR for embed + head")
    parser.add_argument("--lr-npu", type=float, default=1e-3,
                        help="SGD LR used inside NPU kernels (baked into the call)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (B=8 to match NPU microbatch)")
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--weight-scale", type=float, default=0.02,
                        help="Initial weight scale for hidden layers")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="resmlp/checkpoints")
    args = parser.parse_args()

    B = 8
    H = args.hidden_dim
    num_cols = args.num_cols
    num_tiles = num_cols * ROWS_PER_COL

    assert args.num_layers == num_tiles
    assert args.batch_size == B, \
        f"Batch size must be {B} to match NPU microbatch dimension"

    # ── Model ────────────────────────────────────────────────────────────
    model = ResMLP(hidden_dim=H, num_layers=args.num_layers)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")
    else:
        # Scale hidden weights for bfloat16 stability
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.mul_(args.weight_scale / 0.1)

    total_params = sum(p.numel() for p in model.parameters())
    npu_params = sum(layer.weight.numel() for layer in model.layers)
    print(f"Model: {total_params:,} parameters")
    print(f"  NPU hidden: {npu_params:,}  ({num_tiles} × {H}×{H})")
    print(f"  CPU embed+head: {total_params - npu_params:,}")

    # CPU optimizer for embed + head only
    cpu_params = list(model.embed.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.Adam(cpu_params, lr=args.lr_head)
    criterion = nn.CrossEntropyLoss()

    # ── Compile NPU ──────────────────────────────────────────────────────
    print(f"\nCompiling NPU pipeline ({num_tiles} tiles)...", flush=True)
    t0 = time.time()
    ctx = AIEContext()
    npu_op = TrainingPipeline(H=H, B=B, num_cols=num_cols, context=ctx)
    ctx.compile_all()
    ctx.prepare_runtime()
    print(f"  Compiled in {time.time() - t0:.1f}s")

    # ── Pack NPU weights ─────────────────────────────────────────────────
    npu_weights = model.export_npu_weights()  # list of (H,H) bfloat16
    W_packed = np.concatenate([to_tiled(w) for w in npu_weights])

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(args.batch_size)
    zero_buf = np.zeros(B * H, dtype=bfloat16)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"Training for {args.epochs} epochs  |  lr_head={args.lr_head}  lr_npu={args.lr_npu}")
    print(f"{'═' * 70}\n")

    for epoch in range(args.epochs):
        ep_t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        npu_time = 0.0
        npu_calls = 0

        model.train()

        for images, labels in train_loader:
            optimizer.zero_grad()

            # ── Step 1: CPU embedding ──
            x_flat = images.view(B, -1).float()
            x_embedded = torch.nn.functional.linear(x_flat, model.embed.weight, model.embed.bias)

            # ── Step 2: CPU forward through residual layers (bfloat16 sim) ──
            x_np = x_embedded.detach().numpy().astype(bfloat16)
            y_hidden_np = cpu_forward_residual(x_np, npu_weights)

            # ── Step 3: CPU classification head (with autograd) ──
            y_hidden_t = torch.from_numpy(y_hidden_np.astype(np.float32))
            y_hidden_t.requires_grad_(True)
            logits = model.head(y_hidden_t)
            loss = criterion(logits, labels)

            # ── Step 4: Backprop through head → gy ──
            loss.backward()
            gy_np = y_hidden_t.grad.numpy().astype(bfloat16)

            # ── Step 5: NPU forward + backward + SGD ──
            npu_op.write_buffer("act_in", to_tiled(x_np))
            npu_op.write_buffer("weights_in", W_packed)
            npu_op.write_buffer("act_out", zero_buf.copy())
            npu_op.write_buffer("grad_in", to_tiled(gy_np))
            npu_op.write_buffer("grad_out", zero_buf.copy())

            t_npu = time.perf_counter()
            npu_op.run_runlist()
            npu_time += time.perf_counter() - t_npu
            npu_calls += 1

            # Read updated weights back
            flat_w = npu_op.read_buffer("weights_in", (num_tiles * H * H,), copy=True)
            W_packed = flat_w.copy()
            for i in range(num_tiles):
                w_tile = flat_w[i * H * H : (i + 1) * H * H]
                npu_weights[i] = from_tiled(w_tile, H, H)

            # ── Step 6: Backprop through embed using NPU-computed grad_out ──
            gx_flat = npu_op.read_buffer("grad_out", (B * H,), copy=True)
            gx_np = from_tiled(gx_flat, B, H).astype(np.float32)
            # Backprop through embed: loss.backward() already set head grads,
            # now we add embed grads by backpropagating grad_out through embed.
            x_embedded.backward(torch.from_numpy(gx_np))
            optimizer.step()

            # ── Stats ──
            running_loss += loss.item() * B
            preds = logits.detach().argmax(1)
            correct += (preds == labels).sum().item()
            total += B

        # ── Evaluate ──
        ep_time = time.time() - ep_t0
        train_loss = running_loss / total
        train_acc = correct / total

        # Test accuracy using bfloat16-truncated forward pass
        test_correct = 0
        test_total = 0
        test_loss_sum = 0.0
        with torch.no_grad():
            embed_w = model.embed.weight
            embed_b = model.embed.bias
            head_w = model.head.weight
            head_b = model.head.bias
            for images, labels in test_loader:
                cur_B = images.size(0)
                x_flat = images.view(cur_B, -1).float()
                x_emb = (x_flat @ embed_w.T + embed_b).numpy().astype(bfloat16)
                y_hid = cpu_forward_residual(x_emb, npu_weights)
                y_t = torch.from_numpy(y_hid.astype(np.float32))
                logits = y_t @ head_w.T + head_b
                loss = criterion(logits, labels)
                test_loss_sum += loss.item() * cur_B
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += cur_B

        test_loss = test_loss_sum / test_total
        test_acc = test_correct / test_total
        imgs_per_sec = total / ep_time
        npu_ms_per_batch = (npu_time / npu_calls * 1000) if npu_calls else 0

        print(f"  Epoch {epoch:2d}:  "
              f"train loss={train_loss:.4f} acc={train_acc:.4f}  |  "
              f"test loss={test_loss:.4f} acc={test_acc:.4f}  |  "
              f"{ep_time:.1f}s  {imgs_per_sec:.0f} img/s  "
              f"({npu_ms_per_batch:.1f} ms/npu-call, {npu_calls} calls)")

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            path = save_dir / f"resmlp_npu_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "test_acc": test_acc,
            }, path)
            print(f"    → saved {path}")

    print(f"\n{'═' * 70}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"{'═' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
