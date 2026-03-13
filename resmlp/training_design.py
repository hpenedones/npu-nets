"""
IRON design for the full 32-tile NPU training pipeline.

Forward Pass:
- x propagates tile-to-tile.
- Each tile produces a local (x, mask) checkpoint buffered in SRAM (via a local ObjectFifo holding Depth copies if needed. But for 1 microbatch, depth=1 is fine).
Wait, all tiles need to store their (x, mask) until the backward pass reaches them!
Since SRAM is limited, we must spill checkpoints to DDR OR only run 1 microbatch at a time so depth=1 is enough?
Ah, if we only run 1 microbatch, each tile processes 1 forward step, saves 1 checkpoint in its local SRAM, and waits for the backward pass!
Wait, if B=8, H=160, then x is 2.5KB. x and mask is 5KB. A tile has 64KB. We can easily store 5KB in SRAM!

Yes! 100% on-chip training:
1. Forward Pipeline (Tiles 0 -> 31):
   - Tile i receives x_in.
   - Saves x_in to local SRAM (ckpt_x).
   - Computes y, saves mask to local SRAM (ckpt_mask).
   - Sends y to Tile i+1.
2. Loss computation (Host or NPU, but for now we assume Host sends gradient `gy_in` back).
3. Backward Pipeline (Tiles 31 -> 0):
   - Tile i receives dy_in from Tile i+1 (or Host if i=31).
   - Reads ckpt_x, ckpt_mask from local SRAM.
   - Computes gradient dx = ...
   - Updates W in place.
   - Sends dx to Tile i-1.
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib.tap import TensorAccessPattern

ROWS_PER_COL = 4


def snake_tile_order(num_cols):
    tiles = []
    for col in range(num_cols):
        rows = range(2, 6) if col % 2 == 0 else range(5, 1, -1)
        for row in rows:
            tiles.append((col, row))
    return tiles


def training_pipeline(H=160, B=8, num_cols=8):
    assert 1 <= num_cols <= 8
    assert B % 8 == 0 and H % 8 == 0
    num_tiles = num_cols * ROWS_PER_COL
    tile_order = snake_tile_order(num_cols)

    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    # Each tile holds 1 weight matrix: w
    wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]
    col_wt_ty = np.ndarray[(ROWS_PER_COL * H * H,), np.dtype[bfloat16]]
    
    # Checkpoint holds both x and mask
    ckpt_ty = np.ndarray[(2 * B * H,), np.dtype[bfloat16]]

    # Kernels
    fwd_kernel = Kernel(
        "matmul_relu_skip_bf16",
        "resmlp_training_kernels.a",
        [act_ty, wt_ty, act_ty, ckpt_ty, np.int32],
    )
    copy_kernel = Kernel(
        "copy_activation_bf16",
        "resmlp_training_kernels.a",
        [act_ty, act_ty], 
    )
    bwd_kernel = Kernel(
        "residual_backward_and_update_bf16",
        "resmlp_training_kernels.a",
        [ckpt_ty, wt_ty, act_ty, act_ty],
    )

    # Weights
    wt_ddrs = []
    wt_endpoints = []
    for col in range(num_cols):
        wt_ddr = ObjectFifo(col_wt_ty, name=f"wt_col{col}", depth=1)
        splits = wt_ddr.cons().split(
            offsets=[H * H * r for r in range(ROWS_PER_COL)],
            obj_types=[wt_ty] * ROWS_PER_COL,
            names=[f"wt_{col}_{r}" for r in range(ROWS_PER_COL)],
            depths=[1] * ROWS_PER_COL,
            placement=Tile(col=col, row=1),
        )
        wt_ddrs.append(wt_ddr)
        for r in range(ROWS_PER_COL):
            wt_endpoints.append(splits[r])

    # Forward Activations
    act_in = ObjectFifo(act_ty, name="act_in", depth=1)
    act_out = ObjectFifo(act_ty, name="act_out", depth=1)
    act_inter = [ObjectFifo(act_ty, name=f"act_{i}", depth=1) for i in range(num_tiles - 1)]

    # Backward Gradients
    grad_in = ObjectFifo(act_ty, name="grad_in", depth=1)  # Gradient from Host
    grad_out = ObjectFifo(act_ty, name="grad_out", depth=1) # Gradient to Host
    grad_inter = [ObjectFifo(act_ty, name=f"grad_{i}", depth=1) for i in range(num_tiles - 1)]

    workers = []
    for idx, (col, row) in enumerate(tile_order):
        
        local_ckpt = ObjectFifo(ckpt_ty, name=f"ckpt_{idx}", depth=1)
        
        in_ep = act_in.cons() if idx == 0 else act_inter[idx - 1].cons()
        out_ep = act_out.prod() if idx == num_tiles - 1 else act_inter[idx].prod()
        
        wt_ep = wt_endpoints[idx]
        wt_cons = wt_ep.cons() if hasattr(wt_ep, 'cons') else wt_ep
        
        g_in_ep = grad_in.cons() if idx == num_tiles - 1 else grad_inter[idx].cons()
        g_out_ep = grad_out.prod() if idx == 0 else grad_inter[idx - 1].prod()

        def make_worker(i_ep, o_ep, w_ep, gi_ep, go_ep):
            def tile_worker(of_in, of_out, of_w, of_gin, of_gout, of_ckpt_prod, of_ckpt_cons, cp_kern, fwd_kern, bwd_kern):
                # FORWARD PASS
                x = of_in.acquire(1)
                y = of_out.acquire(1)
                w = of_w.acquire(1)
                ckpt = of_ckpt_prod.acquire(1)
                
                x_ckpt = ckpt[0:B*H]
                cp_kern(x, x_ckpt)
                fwd_kern(x, w, y, ckpt, B * H)
                
                of_in.release(1)
                of_out.release(1)
                of_ckpt_prod.release(1)
                
                # BACKWARD PASS
                gy = of_gin.acquire(1)
                gx = of_gout.acquire(1)
                ckpt_read = of_ckpt_cons.acquire(1)
                
                bwd_kern(ckpt_read, w, gy, gx)
                
                of_gin.release(1)
                of_gout.release(1)
                of_ckpt_cons.release(1)
                of_w.release(1)

            return tile_worker
            
        workers.append(Worker(
            make_worker(in_ep, out_ep, wt_cons, g_in_ep, g_out_ep),
            [in_ep, out_ep, wt_cons, g_in_ep, g_out_ep, local_ckpt.prod(), local_ckpt.cons(), copy_kernel, fwd_kernel, bwd_kernel],
            placement=Tile(col=col, row=row),
            stack_size=0x400,
        ))

    host_act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    host_wt_ty = np.ndarray[(num_tiles * H * H,), np.dtype[bfloat16]]
    
    col_wt_elems = ROWS_PER_COL * H * H

    rt = Runtime()
    with rt.sequence(host_act_ty, host_wt_ty, host_act_ty, host_act_ty, host_act_ty) as (x_inp, wts, y_out, g_inp, gx_out):
        rt.start(*workers)

        tg_fwd = rt.task_group()
        # Feed weights to all columns
        for c in range(num_cols):
            tap = TensorAccessPattern(
                (1, num_tiles * H * H),
                c * col_wt_elems,
                [1, ROWS_PER_COL, H, H],
                [0, H * H, H, 1],
            )
            rt.fill(wt_ddrs[c].prod(), wts, tap, task_group=tg_fwd)
            
        rt.fill(act_in.prod(), x_inp, task_group=tg_fwd)
        rt.drain(act_out.cons(), y_out, wait=True, task_group=tg_fwd)
        rt.finish_task_group(tg_fwd)

        tg_bwd = rt.task_group()
        rt.fill(grad_in.prod(), g_inp, task_group=tg_bwd)
        rt.drain(grad_out.cons(), gx_out, wait=True, task_group=tg_bwd)
        rt.finish_task_group(tg_bwd)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
