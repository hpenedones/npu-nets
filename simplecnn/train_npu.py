import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ml_dtypes import bfloat16
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from iron.common.aie_context import AIEContext

from simplecnn.config import BATCH_SIZE, IMG_ELEMS, LABELS_IO_ELEMS, TOTAL_WEIGHT_ELEMS
from simplecnn.model import TinyConvNet
from simplecnn.training_op import SimpleCNNTrainingPipeline


def get_dataloaders(batch_size, data_dir="data", num_workers=0, pin_memory=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def evaluate_model(model, loader, criterion, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            logits = model(images.float())
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    return total_loss / total, correct / total


def run_epoch(model, loader, npu_op, packed_weights, max_batches=None, window_batches=1):
    del model
    correct = 0
    total = 0
    npu_time = 0.0
    npu_calls = 0
    weights_initialized = False
    loader_iter = iter(loader)
    batch_idx = 0
    zero_images = np.zeros(IMG_ELEMS, dtype=bfloat16)

    while True:
        batch_group = []
        while len(batch_group) < window_batches:
            if max_batches is not None and batch_idx >= max_batches:
                break
            try:
                batch_group.append(next(loader_iter))
            except StopIteration:
                break
            batch_idx += 1

        if not batch_group:
            break

        image_tiles = []
        labels_list = []
        labels_io = np.full(window_batches * LABELS_IO_ELEMS, -1, dtype=np.int32)
        for group_idx, (images, labels) in enumerate(batch_group):
            images_np = images.numpy().astype(np.float32).astype(bfloat16, copy=False).reshape(-1)
            labels_np = labels.numpy().astype(np.int32, copy=False)
            image_tiles.append(images_np)
            labels_list.append(labels_np)
            offset = group_idx * LABELS_IO_ELEMS
            labels_io[offset : offset + BATCH_SIZE] = labels_np

        while len(image_tiles) < window_batches:
            image_tiles.append(zero_images)

        if not weights_initialized:
            npu_op.write_buffer("weights", packed_weights)
        npu_op.write_buffer("images", np.concatenate(image_tiles))
        npu_op.write_buffer("labels_io", labels_io)

        npu_time += npu_op.run_resident_window(
            sync_weights_to_device=not weights_initialized,
            sync_weights_from_device=False,
        )
        npu_calls += 1
        weights_initialized = True

        labels_out = npu_op.read_buffer(
            "labels_io",
            (window_batches * LABELS_IO_ELEMS,),
            copy=True,
            dtype=np.int32,
        )
        preds = labels_out.reshape(window_batches, LABELS_IO_ELEMS)[:, BATCH_SIZE:].reshape(-1)
        for group_idx, labels_np in enumerate(labels_list):
            pred_slice = preds[group_idx * BATCH_SIZE : (group_idx + 1) * BATCH_SIZE]
            correct += int((pred_slice == labels_np).sum())
            total += BATCH_SIZE

    if weights_initialized:
        npu_op.sync_resident_weights_from_device()
        packed_weights = npu_op.read_buffer("weights", (TOTAL_WEIGHT_ELEMS,), copy=True)
        if not np.isfinite(np.asarray(packed_weights, dtype=np.float32)).all():
            raise RuntimeError("weights became non-finite during resident simplecnn training")

    return {
        "train_loss": None,
        "train_acc": correct / total if total else None,
        "npu_time": npu_time,
        "npu_calls": npu_calls,
        "packed_weights": packed_weights,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the one-column simple convnet on MNIST")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr-npu", type=float, default=0.05)
    parser.add_argument("--conv-scale", type=float, default=1.0)
    parser.add_argument("--head-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--window-batches", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="simplecnn/checkpoints")
    args = parser.parse_args()

    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"Batch size must be {BATCH_SIZE}")

    model = TinyConvNet()
    model.scale_initial_weights(conv_scale=args.conv_scale, head_scale=args.head_scale)
    packed_weights = model.export_packed_weights()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print("  architecture: 3 stride-2 convs + GAP + linear (top-1 hinge head)")
    print(f"  lr_npu: {args.lr_npu}")
    print(f"  resident window_batches: {args.window_batches}")
    print(f"  conv/head scales: {args.conv_scale:g} / {args.head_scale:g}")

    print("\nCompiling NPU pipeline (1 column, 4 tiles)...", flush=True)
    t0 = time.time()
    ctx = AIEContext(use_runlist=False)
    npu_op = SimpleCNNTrainingPipeline(
        sgd_lr=args.lr_npu,
        window_batches=args.window_batches,
        context=ctx,
    )
    ctx.compile_all()
    ctx.prepare_runtime()
    print(f"  Compiled in {time.time() - t0:.1f}s")

    train_loader, test_loader = get_dataloaders(args.batch_size)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 70}")
    print(f"Training for {args.epochs} epochs  |  pipeline=simplecnn-full-npu")
    print(f"{'═' * 70}\n")

    for epoch in range(args.epochs):
        ep_t0 = time.time()
        stats = run_epoch(
            model,
            train_loader,
            npu_op,
            packed_weights,
            max_batches=args.max_train_batches,
            window_batches=args.window_batches,
        )
        packed_weights = stats["packed_weights"]
        model.load_packed_weights(packed_weights)

        test_loss, test_acc = evaluate_model(
            model, test_loader, criterion, max_batches=args.max_eval_batches
        )
        ep_time = time.time() - ep_t0
        train_images = (
            len(train_loader.dataset)
            if args.max_train_batches is None
            else min(len(train_loader.dataset), args.max_train_batches * BATCH_SIZE)
        )
        imgs_per_sec = train_images / ep_time
        npu_ms = stats["npu_time"] / stats["npu_calls"] * 1000 if stats["npu_calls"] else 0.0

        print(
            f"  Epoch {epoch:2d}:  "
            f"train loss=n/a acc={stats['train_acc']:.4f}  |  "
            f"test loss={test_loss:.4f} acc={test_acc:.4f}  |  "
            f"{ep_time:.1f}s  {imgs_per_sec:.0f} img/s  "
            f"({npu_ms:.1f} ms/npu-call, {stats['npu_calls']} calls)"
        )

    path = save_dir / f"simplecnn_full_npu_epoch{args.epochs - 1:03d}.pt"
    torch.save(
        {
            "epoch": args.epochs - 1,
            "model": model.state_dict(),
            "test_acc": test_acc,
            "test_loss": test_loss,
            "architecture": "conv3-gap-linear-hinge",
        },
        path,
    )
    print(f"\nSaved {path}")
    print(f"Final test accuracy: {test_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
