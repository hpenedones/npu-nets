import argparse
import sys
import numpy as np
from ml_dtypes import bfloat16

from iron.common.aie_context import AIEContext
from resmlp import from_tiled, to_tiled
from resmlp.training_op import TrainingPipeline


def reference_forward_with_checkpoints(x, weights):
    checkpoints = []
    masks = []
    current_x = x.astype(np.float32)
    for i, w in enumerate(weights):
        # Cast inputs back to bfloat16 to simulate the memory load truncation
        layer_x = current_x.astype(bfloat16).astype(np.float32)
        layer_w = w.astype(np.float32)
        
        # Matrix multiply
        y_mm = layer_x @ layer_w.T
        # Truncate to bfloat16 (as stored by AIE before relu, or inside if skipped)
        y_mm = y_mm.astype(bfloat16).astype(np.float32)
        
        # ReLU and Mask
        mask = (y_mm > 0).astype(np.float32)
        y_relu = y_mm * mask
        y_relu = y_relu.astype(bfloat16).astype(np.float32)
        
        # Residual addition
        current_x = layer_x + y_relu
        current_x = current_x.astype(bfloat16).astype(np.float32)
        
        checkpoints.append(layer_x)
        masks.append(mask)
    return checkpoints, masks, current_x.astype(bfloat16).astype(np.float32)
    

def reference_backward_sgd_pipeline(gy, checkpoints, masks, weights, lr):
    current_gy = gy.astype(np.float32)
    new_weights = []
    
    # Iterate backwards through layers
    for i in reversed(range(len(weights))):
        w_f32 = weights[i].astype(np.float32)
        x_f32 = checkpoints[i].astype(np.float32)
        mask_f32 = masks[i].astype(np.float32)
        
        gz = (current_gy * mask_f32).astype(bfloat16).astype(np.float32)
        dw = (x_f32.T @ gz).astype(bfloat16).astype(np.float32)
        w_new = (w_f32 - lr * dw).astype(bfloat16).astype(np.float32)
        
        gx_mm = (gz @ w_f32.T).astype(bfloat16).astype(np.float32)
        gx = (gx_mm + current_gy).astype(bfloat16).astype(np.float32)
        
        new_weights.insert(0, w_new.astype(bfloat16))
        current_gy = gx

    return current_gy.astype(bfloat16), new_weights


def compare(name, ref, got, rtol, atol):
    ref_f32 = ref.astype(np.float32)
    got_f32 = got.astype(np.float32)
    close = np.isclose(ref_f32, got_f32, rtol=rtol, atol=atol)
    pct = close.mean() * 100.0
    max_diff = np.max(np.abs(ref_f32 - got_f32))
    status = "PASS" if pct > 95 else "FAIL"
    print(f"  [{status}] {name}: {pct:.1f}% close, max diff = {max_diff:.4f}")
    if pct <= 95:
        print(f"    ref [0,:5]: {ref_f32.reshape(ref.shape)[0, :5]}")
        print(f"    npu [0,:5]: {got_f32.reshape(got.shape)[0, :5]}")
    return pct > 95


def run_test(H=160, B=8, cols=4, scale=0.05, lr=0.01):
    rng = np.random.default_rng(42)
    
    # Forward Pass Inputs
    x = rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
    # We need w for each tile!
    raw_weights = [
        (rng.standard_normal((H, H)).astype(np.float32) * scale).astype(bfloat16)
        for _ in range(cols * 4)
    ]
    # NPU takes [w]
    npu_weights = []
    for w in raw_weights:
        npu_weights.append(w)

    # Reference Forward Pass
    checkpoints_ref, masks_ref, y_ref = reference_forward_with_checkpoints(x, raw_weights)
    
    # Backward Pass Inputs
    gy = rng.standard_normal((B, H)).astype(np.float32).astype(bfloat16)
    
    # Reference Backward Pass
    gx_ref, w_new_ref = reference_backward_sgd_pipeline(gy, checkpoints_ref, masks_ref, raw_weights, lr)

    ctx = AIEContext()
    op = TrainingPipeline(H=H, B=B, num_cols=cols, context=ctx)
    print(f"Compiling training pipeline ({cols*4} tiles, B={B}, H={H})...", flush=True)
    ctx.compile_all()
    ctx.prepare_runtime()

    op.write_buffer("act_in", to_tiled(x))
    op.write_buffer("weights_in", np.concatenate([to_tiled(w) for w in npu_weights]))
    op.write_buffer("grad_in", to_tiled(gy))
    
    op.write_buffer("act_out", np.zeros(B * H, dtype=bfloat16))
    op.write_buffer("grad_out", np.zeros(B * H, dtype=bfloat16))
    
    # Run the NPU program!
    print("\n\nSTARTING NPU RUN...\n", flush=True); op.run_runlist(); print("\n\nFINISHED NPU RUN\n", flush=True)

    y_npu = from_tiled(op.read_buffer("act_out", (B * H,), copy=True), B, H)
    gx_npu = from_tiled(op.read_buffer("grad_out", (B * H,), copy=True), B, H)
    
    # The weights buffer holds the updated weights.
    # Note: `copy=True` since read_buffer returns a view into NPU mem.
    flat_w_npu = op.read_buffer("weights_in", (cols * 4 * H * H,), copy=True)
    w_new_npu = []
    for i in range(cols * 4):
        # The weights are layout sequentially: w_0, w_1...
        w_tile = flat_w_npu[i * H * H : (i + 1) * H * H]
        w_new_npu.append(from_tiled(w_tile, H, H))

    print("\nTraining Pipeline Checks:")
    ok = True
    ok &= compare("forward output", y_ref, y_npu, rtol=0.20, atol=0.40)
    ok &= compare("backward grad_in", gx_ref, gx_npu, rtol=0.20, atol=0.40)
    
    for i, (ref, got) in enumerate(zip(w_new_ref, w_new_npu)):
        ok &= compare(f"weight update w_{i}", ref, got, rtol=0.10, atol=0.10)
        
    return ok


def main():
    parser = argparse.ArgumentParser(description="Test 32-tile training pipeline")
    parser.add_argument("--H", type=int, default=160)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8, help="Number of columns to use (1..8, where each col has 4 tiles)")
    parser.add_argument("--scale", type=float, default=0.002, help="Initialization scale for weights")
    args = parser.parse_args()

    print(f"═══ Full NPU Training Pipeline (cols={args.cols}, B={args.B}, H={args.H}) ═══\n")
    ok = run_test(H=args.H, B=args.B, cols=args.cols, scale=args.scale)
    print(f"\n{'═' * 50}")
    print("ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
