"""Evaluate and benchmark the forward-only HIGGS conveyor-belt path on the NPU."""

import argparse
import math
import sys
import time

import numpy as np
import torch
from ml_dtypes import bfloat16

from resmlp.xrt_env import ensure_xrt_python_path

ensure_xrt_python_path()

from iron.common.aie_context import AIEContext

from resmlp import from_tiled, to_tiled
from resmlp.data_utils import DEFAULT_SPLIT_SEED, DEFAULT_VAL_SIZE, get_eval_dataset
from resmlp.design import ROWS_PER_COL
from resmlp.model import ResMLP
from resmlp.streaming_op import StreamingResMLP


class HiggsStreamingInferenceService:
    """Host-facing streaming service for the HIGGS residual body on the NPU."""

    def __init__(
        self,
        checkpoint,
        *,
        hidden_dim=None,
        num_layers=None,
        batch_size=8,
        num_cols=None,
        stream_depth=32,
    ):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        dataset = ckpt.get("dataset", "higgs")
        if dataset.lower() != "higgs":
            raise ValueError(f"This curated branch only supports HIGGS checkpoints, got '{dataset}'")

        self.dataset = "higgs"
        self.B = batch_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else ckpt.get("hidden_dim", 64)
        self.num_layers = num_layers if num_layers is not None else ckpt.get("num_layers", 32)
        self.num_cols = num_cols if num_cols is not None else math.ceil(self.num_layers / ROWS_PER_COL)
        self.num_tiles = self.num_cols * ROWS_PER_COL
        self.stream_depth = stream_depth
        self.input_dim = ckpt.get("input_dim", 28)
        self.num_classes = ckpt.get("num_classes", 2)
        self.residual_bias = bool(ckpt.get("residual_bias", False))
        self.pipeline = ckpt.get("pipeline", "hybrid")
        self.epoch = ckpt["epoch"]
        self.eval_split = ckpt.get("eval_split", "test")
        self.val_size = ckpt.get("val_size", DEFAULT_VAL_SIZE)
        self.split_seed = ckpt.get("split_seed", DEFAULT_SPLIT_SEED)
        self.saved_eval_acc = ckpt.get(
            f"{self.eval_split}_acc",
            ckpt.get("val_acc", ckpt.get("test_acc")),
        )

        if self.B % 8 != 0:
            raise ValueError(f"batch_size must be divisible by 8, got {self.B}")
        if self.hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim must be divisible by 8, got {self.hidden_dim}")
        if self.num_layers != self.num_tiles:
            raise ValueError(
                f"Checkpoint uses {self.num_layers} residual layers, but {self.num_cols} columns provide "
                f"{self.num_tiles} tiles. Choose num_cols so they match exactly."
            )
        if self.residual_bias:
            raise ValueError("Streaming residual inference does not support residual-bias checkpoints.")
        if self.num_classes != 2:
            raise ValueError(
                f"This curated branch expects binary HIGGS checkpoints, got {self.num_classes} classes"
            )

        self.model = ResMLP(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            residual_bias=self.residual_bias,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.embed_weight = self.model.embed.weight.detach().float()
        self.embed_bias = self.model.embed.bias.detach().float()
        residual_weights = self.model.export_residual_weights()
        self.packed_weights_by_tile = np.stack(
            [np.asarray(to_tiled(weight), dtype=bfloat16) for weight in residual_weights]
        )
        self.cpu_head_weight = self.model.head.weight.detach().cpu().float().numpy()
        self.cpu_head_bias = self.model.head.bias.detach().cpu().float().numpy()
        self.zero_hidden = np.zeros((self.B, self.hidden_dim), dtype=bfloat16)

        ctx = AIEContext(use_runlist=False)
        self.npu_op = StreamingResMLP(
            self.packed_weights_by_tile,
            H=self.hidden_dim,
            B=self.B,
            num_cols=self.num_cols,
            stream_depth=self.stream_depth,
            context=ctx,
        )
        ctx.compile_all()
        ctx.prepare_runtime()

    def _embed_batch(self, features):
        flat = features.view(self.B, -1).float()
        hidden = (flat @ self.embed_weight.T + self.embed_bias).numpy()
        return hidden.astype(bfloat16)

    def _pad_and_pack_hidden(self, hidden_batches):
        valid = len(hidden_batches)
        hidden_tiles = [to_tiled(batch.astype(bfloat16, copy=False)) for batch in hidden_batches]
        while len(hidden_tiles) < self.stream_depth:
            hidden_tiles.append(to_tiled(self.zero_hidden))
        return np.concatenate(hidden_tiles), valid

    def process_hidden_chunk(self, hidden_batches):
        if not hidden_batches:
            return [], 0.0

        packed_input, valid = self._pad_and_pack_hidden(hidden_batches)
        self.npu_op.write_buffer("input", packed_input)
        elapsed = self.npu_op.run_stream()

        outputs = []
        hidden_flat = self.npu_op.read_buffer(
            "output",
            (self.stream_depth * self.B * self.hidden_dim,),
            copy=True,
        )
        for idx in range(valid):
            start = idx * self.B * self.hidden_dim
            stop = (idx + 1) * self.B * self.hidden_dim
            hidden = from_tiled(hidden_flat[start:stop], self.B, self.hidden_dim).astype(np.float32)
            outputs.append(hidden @ self.cpu_head_weight.T + self.cpu_head_bias)
        return outputs, elapsed

    def benchmark(self, repeated_features, num_samples, warmup_calls=4):
        if repeated_features.shape[0] != self.B:
            raise ValueError(
                f"Expected repeated_features batch dimension {self.B}, got {repeated_features.shape[0]}"
            )

        calls = math.ceil(num_samples / (self.B * self.stream_depth))
        processed_samples = calls * self.B * self.stream_depth
        repeated_features = repeated_features.contiguous()

        for _ in range(warmup_calls):
            hidden_batches = [self._embed_batch(repeated_features) for _ in range(self.stream_depth)]
            self.process_hidden_chunk(hidden_batches)

        cpu_embed_s = 0.0
        kernel_s = 0.0
        wall_t0 = time.perf_counter()
        for _ in range(calls):
            cpu_t0 = time.perf_counter()
            hidden_batches = [self._embed_batch(repeated_features) for _ in range(self.stream_depth)]
            cpu_embed_s += time.perf_counter() - cpu_t0
            _, elapsed = self.process_hidden_chunk(hidden_batches)
            kernel_s += elapsed
        wall_s = time.perf_counter() - wall_t0

        return {
            "num_samples_requested": num_samples,
            "num_samples_processed": processed_samples,
            "wall_s": wall_s,
            "cpu_embed_s": cpu_embed_s,
            "kernel_s": kernel_s,
            "wall_sample_s": processed_samples / wall_s,
            "kernel_sample_s": processed_samples / kernel_s,
        }


def success_exit_code(observed_acc, expected_acc, *, partial_run=False):
    if partial_run or expected_acc is None:
        return 0
    return 0 if observed_acc >= max(0.0, expected_acc - 0.10) else 1


def dataset_batch_iterator(dataset, batch_size):
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        actual_b = end - start
        features = torch.stack([dataset[i][0] for i in range(start, end)])
        labels = torch.tensor([dataset[i][1] for i in range(start, end)])
        if actual_b < batch_size:
            pad = torch.zeros(batch_size - actual_b, *features.shape[1:])
            features = torch.cat([features, pad])
        yield features, labels, actual_b


def main():
    parser = argparse.ArgumentParser(description="Streaming HIGGS inference on the NPU")
    parser.add_argument("checkpoint", help="Path to a trained .pt checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-cols", type=int, default=None)
    parser.add_argument("--stream-depth", type=int, default=32)
    parser.add_argument("--bench", action="store_true", help="Show timing for evaluation runs")
    parser.add_argument("--bench-samples", type=int, default=None)
    parser.add_argument("--eval-split", choices=["val", "test"], default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data/higgs_full")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    t0 = time.time()
    service = HiggsStreamingInferenceService(
        args.checkpoint,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        num_cols=args.num_cols,
        stream_depth=args.stream_depth,
    )
    print(
        f"  dataset={service.dataset}, pipeline={service.pipeline}, epoch={service.epoch}, "
        f"saved {service.eval_split} acc={service.saved_eval_acc}"
    )
    print(
        f"Compiled streaming NPU pipeline ({service.num_tiles} tiles, B={service.B}, H={service.hidden_dim}, "
        f"stream_depth={service.stream_depth}) in {time.time() - t0:.1f}s"
    )

    eval_split = args.eval_split or service.eval_split
    eval_ds = get_eval_dataset(
        service.dataset,
        split=eval_split,
        data_dir=args.data_dir,
        val_size=service.val_size,
        split_seed=service.split_seed,
    )

    if args.bench_samples is not None:
        repeated_features, _, _ = next(dataset_batch_iterator(eval_ds, service.B))
        stats = service.benchmark(repeated_features, args.bench_samples)
        print("\nRepeated-sample benchmark:")
        for key in (
            "num_samples_requested",
            "num_samples_processed",
            "wall_s",
            "cpu_embed_s",
            "kernel_s",
            "wall_sample_s",
            "kernel_sample_s",
        ):
            print(f"  {key}: {stats[key]}")
        return 0

    correct = 0
    total = 0
    npu_time = 0.0
    npu_calls = 0
    cpu_time = 0.0

    batch_iter = dataset_batch_iterator(eval_ds, service.B)
    consumed_batches = 0
    while True:
        batch_group = []
        try:
            while len(batch_group) < service.stream_depth:
                if args.max_batches is not None and consumed_batches >= args.max_batches:
                    break
                features, labels, actual_b = next(batch_iter)
                batch_group.append((features, labels, actual_b))
                consumed_batches += 1
        except StopIteration:
            pass

        if not batch_group:
            break

        hidden_batches = []
        batch_meta = []
        cpu_t0 = time.perf_counter()
        for features, labels, actual_b in batch_group:
            hidden_batches.append(service._embed_batch(features))
            batch_meta.append((labels, actual_b))
        cpu_time += time.perf_counter() - cpu_t0

        outputs, elapsed = service.process_hidden_chunk(hidden_batches)
        npu_time += elapsed
        npu_calls += 1

        for logits_np, (labels, actual_b) in zip(outputs, batch_meta):
            logits = torch.from_numpy(logits_np)
            preds = logits[:actual_b].argmax(1)
            correct += (preds == labels).sum().item()
            total += actual_b

    accuracy = correct / total if total else 0.0
    print(f"\n{'═' * 50}")
    print(f"Streaming NPU {eval_split} accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.bench and npu_calls:
        print("\nTiming:")
        print(f"  NPU total:    {npu_time * 1000:.1f} ms ({npu_time / npu_calls * 1000:.3f} ms/call)")
        print(f"  CPU total:    {cpu_time * 1000:.1f} ms")
        print(f"  NPU calls:    {npu_calls}")
        print(f"  Throughput:   {total / npu_time:.0f} samples/sec (NPU only)")

    return success_exit_code(
        accuracy,
        service.saved_eval_acc if eval_split == service.eval_split else None,
        partial_run=args.max_batches is not None,
    )


if __name__ == "__main__":
    sys.exit(main())
