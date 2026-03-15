"""
IRON design for the full-NPU MNIST training pipeline.

Default tile layout at 8 columns (32 tiles = 1 embed + 30 residual + 1 head):
    Tile 0  (col 0, row 2):  Embed -- 784 -> H matmul, streamed over K chunks
    Tiles 1-30 (snake):      Residual -- relu(x @ W) + x
    Tile 31 (col 7, row 5):  Head -- H -> 16 matmul + softmax + CE loss

Reduced-shape variants reuse the same pattern with fewer columns, so the
residual depth becomes `num_cols * 4 - 2`.

Forward:
    Host -> x_raw[8x784] -> [Embed] -> act -> [Res 0..N-1] -> [Head] -> preds -> Host
    Host -> labels[8] -> [Head]

Backward:
    [Head] -> gy[8xH] -> [Res N-1..0] -> gy[8xH] -> [Embed]
    Host -> x_raw[8x784] -> [Embed]  (re-stream for embed grad)
    Host <- updated embed/head weights
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern

from resmlp.artifact_utils import full_training_kernel_archive_name

ROWS_PER_COL = 4
NUM_RESIDUAL = 30
N_CLS_PADDED = 16
EMBED_CHUNK_ROWS = 56


def residual_drainback_enabled(H, num_cols):
    return H <= 64 and num_cols <= 2


def snake_tile_order(num_cols):
    tiles = []
    for col in range(num_cols):
        rows = range(2, 6) if col % 2 == 0 else range(5, 1, -1)
        for row in rows:
            tiles.append((col, row))
    return tiles


def full_training_pipeline(
    H=32,
    B=8,
    K_EMBED=784,
    num_cols=8,
    window_batches=1,
    sgd_lr=0.005,
):
    assert 2 <= num_cols <= 8 and B % 8 == 0 and H % 8 == 0 and K_EMBED % 8 == 0
    assert K_EMBED % EMBED_CHUNK_ROWS == 0
    assert window_batches >= 1
    use_split_backward = H >= 160
    archive_name = full_training_kernel_archive_name(
        Path(__file__).resolve().parent.parent,
        B=B,
        H=H,
        embed_chunk_rows=EMBED_CHUNK_ROWS,
        n_cls_padded=N_CLS_PADDED,
        sgd_lr=sgd_lr,
    )
    num_tiles = num_cols * ROWS_PER_COL
    num_residual = num_tiles - 2
    assert num_residual >= 1
    tile_order = snake_tile_order(num_cols)
    num_embed_chunks = K_EMBED // EMBED_CHUNK_ROWS
    use_residual_drainback = residual_drainback_enabled(H, num_cols)

    # Types
    act_ty = np.ndarray[(B * H,), np.dtype[bfloat16]]
    embed_in_chunk_ty = np.ndarray[(B * EMBED_CHUNK_ROWS,), np.dtype[bfloat16]]
    embed_wt_chunk_ty = np.ndarray[(EMBED_CHUNK_ROWS * H,), np.dtype[bfloat16]]
    res_wt_ty = np.ndarray[(H * H,), np.dtype[bfloat16]]
    head_wt_ty = np.ndarray[(H * N_CLS_PADDED,), np.dtype[bfloat16]]
    ckpt_ty = np.ndarray[(2 * B * H,), np.dtype[bfloat16]]
    labels_ty = np.ndarray[(B,), np.dtype[np.int32]]
    d_logits_ty = np.ndarray[(B * N_CLS_PADDED,), np.dtype[bfloat16]]

    # Kernels
    embed_fwd_k = Kernel(
        "embed_forward_bf16",
        archive_name,
        [embed_in_chunk_ty, embed_wt_chunk_ty, act_ty, np.int32],
    )
    embed_bwd_k = Kernel(
        "embed_backward_bf16",
        archive_name,
        [embed_in_chunk_ty, embed_wt_chunk_ty, act_ty],
    )
    embed_wt_copy_k = Kernel(
        "copy_embed_weight_bf16",
        archive_name,
        [embed_wt_chunk_ty, embed_wt_chunk_ty],
    )
    res_fwd_k = Kernel(
        "matmul_relu_skip_bf16",
        archive_name,
        [act_ty, res_wt_ty, act_ty, ckpt_ty, np.int32],
    )
    res_copy_k = Kernel(
        "copy_activation_bf16",
        archive_name,
        [act_ty, act_ty],
    )
    if use_residual_drainback:
        res_wt_copy_k = Kernel(
            "copy_res_weight_bf16",
            archive_name,
            [res_wt_ty, res_wt_ty],
        )
    res_bwd_k = Kernel(
        "residual_backward_and_update_bf16",
        archive_name,
        [ckpt_ty, res_wt_ty, act_ty, act_ty],
    )
    res_bwd_gx_k = Kernel(
        "residual_grad_input_from_ckpt_bf16",
        archive_name,
        [ckpt_ty, res_wt_ty, act_ty, act_ty],
    )
    res_bwd_update_k = Kernel(
        "residual_sgd_update_from_gz_bf16",
        archive_name,
        [ckpt_ty, res_wt_ty],
    )
    head_fwd_k = Kernel(
        "head_forward_loss_bf16",
        archive_name,
        [act_ty, head_wt_ty, labels_ty, d_logits_ty, labels_ty],
    )
    head_bwd_k = Kernel(
        "head_backward_bf16",
        archive_name,
        [act_ty, head_wt_ty, d_logits_ty, act_ty],
    )
    head_wt_copy_k = Kernel(
        "copy_head_weight_bf16",
        archive_name,
        [head_wt_ty, head_wt_ty],
    )
    # Weight FIFOs
    embed_wt_fifo = ObjectFifo(embed_wt_chunk_ty, name="embed_wt", depth=1)
    embed_wt_out_fifo = ObjectFifo(embed_wt_chunk_ty, name="embed_wt_out", depth=1)
    head_wt_fifo = ObjectFifo(head_wt_ty, name="head_wt", depth=1)
    head_wt_out = ObjectFifo(head_wt_ty, name="head_wt_out", depth=1)

    res_per_col = [ROWS_PER_COL] * num_cols
    res_per_col[0] -= 1
    res_per_col[-1] -= 1
    assert sum(res_per_col) == num_residual
    res_wt_ddrs = []
    res_wt_endpoints = []
    res_wt_out_ddrs = []
    res_wt_out_endpoints = []
    for col_idx, n_res in enumerate(res_per_col):
        col_wt_ty = np.ndarray[(n_res * H * H,), np.dtype[bfloat16]]
        wt_ddr = ObjectFifo(col_wt_ty, name=f"res_wt_col{col_idx}", depth=1)
        splits = wt_ddr.cons().split(
            offsets=[H * H * r for r in range(n_res)],
            obj_types=[res_wt_ty] * n_res,
            names=[f"res_wt_{col_idx}_{r}" for r in range(n_res)],
            depths=[1] * n_res,
            placement=Tile(col=col_idx, row=1),
        )
        res_wt_ddrs.append(wt_ddr)
        for r in range(n_res):
            res_wt_endpoints.append(splits[r])
        if use_residual_drainback:
            wt_out_ddr = ObjectFifo(
                col_wt_ty,
                name=f"res_wt_out_col{col_idx}",
                depth=1,
            )
            joins = wt_out_ddr.prod().join(
                offsets=[H * H * r for r in range(n_res)],
                obj_types=[res_wt_ty] * n_res,
                names=[f"res_wt_out_{col_idx}_{r}" for r in range(n_res)],
                depths=[1] * n_res,
                placement=Tile(col=col_idx, row=1),
            )
            res_wt_out_ddrs.append(wt_out_ddr)
            for r in range(n_res):
                res_wt_out_endpoints.append(joins[r])

    # Activation / label / gradient FIFOs
    embed_in = ObjectFifo(embed_in_chunk_ty, name="embed_in", depth=1)
    act_fifos = [
        ObjectFifo(act_ty, name=f"act_{i}", depth=1)
        for i in range(num_residual + 1)
    ]
    labels_fifo = ObjectFifo(labels_ty, name="labels", depth=1)
    preds_fifo = ObjectFifo(labels_ty, name="preds", depth=1)
    done_fifo = ObjectFifo(np.ndarray[(1,), np.dtype[np.int32]], name="done", depth=1)
    grad_fifos = [
        ObjectFifo(act_ty, name=f"grad_{i}", depth=1)
        for i in range(num_residual + 1)
    ]

    workers = []

    # Embed worker
    embed_col, embed_row = tile_order[0]
    x_chunk_elems = B * EMBED_CHUNK_ROWS
    wt_chunk_elems = EMBED_CHUNK_ROWS * H
    embed_wt_ep = embed_wt_fifo.cons()
    embed_wt_cons = embed_wt_ep.cons() if hasattr(embed_wt_ep, "cons") else embed_wt_ep

    def make_embed_worker():
        def w(
            of_x_fwd,
            of_y_out,
            of_w,
            of_w_out,
            of_grad_in,
            of_done,
            fwd_k,
            bwd_k,
            copy_wt_k,
        ):
            window_loop = range(1)
            if window_batches > 1:
                window_loop = range_(window_batches)

            for _ in window_loop:
                y = of_y_out.acquire(1)
                for chunk_idx in range(num_embed_chunks):
                    x = of_x_fwd.acquire(1)
                    wt = of_w.acquire(1)
                    fwd_k(x, wt, y, 1 if chunk_idx == 0 else 0)
                    of_x_fwd.release(1)
                    of_w.release(1)
                of_y_out.release(1)

                dy = of_grad_in.acquire(1)
                for _ in range(num_embed_chunks):
                    x = of_x_fwd.acquire(1)
                    wt = of_w.acquire(1)
                    wt_out = of_w_out.acquire(1)
                    bwd_k(x, wt, dy)
                    copy_wt_k(wt, wt_out)
                    of_x_fwd.release(1)
                    of_w.release(1)
                    of_w_out.release(1)
                done = of_done.acquire(1)
                done[0] = 1
                of_grad_in.release(1)
                of_done.release(1)

        return w

    workers.append(
        Worker(
            make_embed_worker(),
            [
                embed_in.cons(),
                act_fifos[0].prod(),
                embed_wt_cons,
                embed_wt_out_fifo.prod(),
                grad_fifos[num_residual].cons(),
                done_fifo.prod(),
                embed_fwd_k,
                embed_bwd_k,
                embed_wt_copy_k,
            ],
            placement=Tile(col=embed_col, row=embed_row),
            stack_size=0x400,
        )
    )

    # Residual workers
    for res_idx in range(num_residual):
        col, row = tile_order[res_idx + 1]
        local_ckpt = ObjectFifo(ckpt_ty, name=f"ckpt_{res_idx}", depth=1)

        in_ep = act_fifos[res_idx].cons()
        out_ep = act_fifos[res_idx + 1].prod()
        wt_ep = res_wt_endpoints[res_idx]
        wt_c = wt_ep.cons() if hasattr(wt_ep, "cons") else wt_ep
        wt_out_ep = None
        if use_residual_drainback:
            wt_out_ep = res_wt_out_endpoints[res_idx]
            wt_out_ep = wt_out_ep.prod() if hasattr(wt_out_ep, "prod") else wt_out_ep
        g_in = grad_fifos[num_residual - 1 - res_idx].cons()
        g_out = grad_fifos[num_residual - res_idx].prod()

        if use_residual_drainback:
            def make_res():
                def w(
                    of_in,
                    of_out,
                    of_w,
                    of_w_out,
                    of_gin,
                    of_gout,
                    of_cp,
                    of_cr,
                    cp_k,
                    copy_w_k,
                    fwd_k,
                    bwd_k,
                    bwd_gx_k,
                    bwd_update_k,
                ):
                    ww = of_w.acquire(1)
                    try:
                        window_loop = range(1)
                        if window_batches > 1:
                            window_loop = range_(window_batches)

                        for _ in window_loop:
                            x = of_in.acquire(1)
                            y = of_out.acquire(1)
                            ck = of_cp.acquire(1)
                            cp_k(x, ck[0 : B * H])
                            fwd_k(x, ww, y, ck, B * H)
                            of_in.release(1)
                            of_out.release(1)
                            of_cp.release(1)

                            gy = of_gin.acquire(1)
                            gx = of_gout.acquire(1)
                            cr = of_cr.acquire(1)
                            if use_split_backward:
                                bwd_gx_k(cr, ww, gy, gx)
                                bwd_update_k(cr, ww)
                            else:
                                bwd_k(cr, ww, gy, gx)
                            of_gin.release(1)
                            of_gout.release(1)
                            of_cr.release(1)

                        wt_out = of_w_out.acquire(1)
                        copy_w_k(ww, wt_out)
                        of_w_out.release(1)
                    finally:
                        of_w.release(1)

                return w

            worker_args = [
                in_ep,
                out_ep,
                wt_c,
                wt_out_ep,
                g_in,
                g_out,
                local_ckpt.prod(),
                local_ckpt.cons(),
                res_copy_k,
                res_wt_copy_k,
                res_fwd_k,
                res_bwd_k,
                res_bwd_gx_k,
                res_bwd_update_k,
            ]
        else:
            def make_res():
                def w(
                    of_in,
                    of_out,
                    of_w,
                    of_gin,
                    of_gout,
                    of_cp,
                    of_cr,
                    cp_k,
                    fwd_k,
                    bwd_k,
                    bwd_gx_k,
                    bwd_update_k,
                ):
                    ww = of_w.acquire(1)
                    try:
                        window_loop = range(1)
                        if window_batches > 1:
                            window_loop = range_(window_batches)

                        for _ in window_loop:
                            x = of_in.acquire(1)
                            y = of_out.acquire(1)
                            ck = of_cp.acquire(1)
                            cp_k(x, ck[0 : B * H])
                            fwd_k(x, ww, y, ck, B * H)
                            of_in.release(1)
                            of_out.release(1)
                            of_cp.release(1)

                            gy = of_gin.acquire(1)
                            gx = of_gout.acquire(1)
                            cr = of_cr.acquire(1)
                            if use_split_backward:
                                bwd_gx_k(cr, ww, gy, gx)
                                bwd_update_k(cr, ww)
                            else:
                                bwd_k(cr, ww, gy, gx)
                            of_gin.release(1)
                            of_gout.release(1)
                            of_cr.release(1)
                    finally:
                        of_w.release(1)

                return w

            worker_args = [
                in_ep,
                out_ep,
                wt_c,
                g_in,
                g_out,
                local_ckpt.prod(),
                local_ckpt.cons(),
                res_copy_k,
                res_fwd_k,
                res_bwd_k,
                res_bwd_gx_k,
                res_bwd_update_k,
            ]

        workers.append(
            Worker(
                make_res(),
                worker_args,
                placement=Tile(col=col, row=row),
                stack_size=0x400,
            )
        )

    # Head worker
    head_col, head_row = tile_order[num_residual + 1]
    head_ckpt_fifo = ObjectFifo(act_ty, name="head_ckpt", depth=1)
    d_logits_fifo = ObjectFifo(d_logits_ty, name="d_logits", depth=1)
    head_wt_ep = head_wt_fifo.cons()
    head_wt_cons = head_wt_ep.cons() if hasattr(head_wt_ep, "cons") else head_wt_ep

    def make_head():
        def w(
            of_yin,
            of_wt,
            of_lab,
            of_wt_out,
            of_cp,
            of_cr,
            of_dlp,
            of_dlc,
            of_gout,
            of_preds,
            cp_k,
            fwd_k,
            bwd_k,
            copy_wt_k,
        ):
            wt = of_wt.acquire(1)
            try:
                window_loop = range(1)
                if window_batches > 1:
                    window_loop = range_(window_batches)

                for _ in window_loop:
                    y = of_yin.acquire(1)
                    lb = of_lab.acquire(1)
                    ck = of_cp.acquire(1)
                    dl = of_dlp.acquire(1)
                    pred = of_preds.acquire(1)

                    cp_k(y, ck)
                    fwd_k(y, wt, lb, dl, pred)
                    of_yin.release(1)
                    of_lab.release(1)
                    of_cp.release(1)
                    of_dlp.release(1)
                    of_preds.release(1)

                    cr = of_cr.acquire(1)
                    dr = of_dlc.acquire(1)
                    gy = of_gout.acquire(1)
                    bwd_k(cr, wt, dr, gy)
                    of_cr.release(1)
                    of_dlc.release(1)
                    of_gout.release(1)

                wt_out = of_wt_out.acquire(1)
                copy_wt_k(wt, wt_out)
                of_wt_out.release(1)
            finally:
                of_wt.release(1)

        return w

    workers.append(
        Worker(
            make_head(),
            [
                act_fifos[num_residual].cons(),
                head_wt_cons,
                labels_fifo.cons(),
                head_wt_out.prod(),
                head_ckpt_fifo.prod(),
                head_ckpt_fifo.cons(),
                d_logits_fifo.prod(),
                d_logits_fifo.cons(),
                grad_fifos[0].prod(),
                preds_fifo.prod(),
                res_copy_k,
                head_fwd_k,
                head_bwd_k,
                head_wt_copy_k,
            ],
            placement=Tile(col=head_col, row=head_row),
            stack_size=0x4000,
        )
    )

    # Runtime sequence
    host_embed_in_ty = np.ndarray[(window_batches * B * K_EMBED,), np.dtype[bfloat16]]
    host_embed_pair_ty = np.ndarray[(2 * K_EMBED * H,), np.dtype[bfloat16]]
    host_res_wt_ty = np.ndarray[(num_residual * H * H,), np.dtype[bfloat16]]
    host_head_wt_ty = np.ndarray[(H * N_CLS_PADDED,), np.dtype[bfloat16]]
    host_labels_ty = np.ndarray[(window_batches * 2 * B + 1,), np.dtype[np.int32]]

    rt = Runtime()
    with rt.sequence(
        host_embed_in_ty,
        host_embed_pair_ty,
        host_res_wt_ty,
        host_head_wt_ty,
        host_labels_ty,
    ) as (
        x_raw,
        wt_embed_pair,
        wt_res,
        wt_head,
        labels,
    ):
        rt.start(*workers)

        embed_in_taps = [
            TensorAccessPattern(
                (1, window_batches * B * K_EMBED),
                batch_idx * B * K_EMBED,
                [num_embed_chunks, x_chunk_elems],
                [x_chunk_elems, 1],
            )
            for batch_idx in range(window_batches)
        ]
        embed_wt_taps = [
            TensorAccessPattern(
                (1, 2 * K_EMBED * H),
                embed_base,
                [num_embed_chunks, EMBED_CHUNK_ROWS, H],
                [wt_chunk_elems, H, 1],
            )
            for embed_base in (0, K_EMBED * H)
        ]
        res_wt_out_taps = []
        if use_residual_drainback:
            res_wt_offset = 0
            for n_res in res_per_col:
                res_wt_out_taps.append(
                    TensorAccessPattern(
                        (1, num_residual * H * H),
                        res_wt_offset,
                        [n_res, H, H],
                        [H * H, H, 1],
                    )
                )
                res_wt_offset += n_res * H * H

        for batch_idx in range(window_batches):
            embed_src_tap = embed_wt_taps[batch_idx % 2]
            embed_dst_tap = embed_wt_taps[(batch_idx + 1) % 2]

            tg_fwd = rt.task_group()
            rt.fill(
                embed_in.prod(),
                x_raw,
                tap=embed_in_taps[batch_idx],
                task_group=tg_fwd,
            )
            rt.fill(
                embed_wt_fifo.prod(),
                wt_embed_pair,
                tap=embed_src_tap,
                task_group=tg_fwd,
            )
            if batch_idx == 0:
                rt.fill(head_wt_fifo.prod(), wt_head, task_group=tg_fwd)

                res_wt_offset = 0
                for n_res, wt_ddr in zip(res_per_col, res_wt_ddrs):
                    col_elems = n_res * H * H
                    tap = TensorAccessPattern(
                        (1, num_residual * H * H),
                        res_wt_offset,
                        [1, col_elems],
                        [0, 1],
                    )
                    rt.fill(wt_ddr.prod(), wt_res, tap, task_group=tg_fwd)
                    res_wt_offset += col_elems

            labels_in_tap = TensorAccessPattern(
                (1, window_batches * 2 * B + 1),
                batch_idx * 2 * B,
                [1, B],
                [0, 1],
            )
            preds_out_tap = TensorAccessPattern(
                (1, window_batches * 2 * B + 1),
                batch_idx * 2 * B + B,
                [1, B],
                [0, 1],
            )
            rt.fill(labels_fifo.prod(), labels, tap=labels_in_tap, task_group=tg_fwd)
            rt.drain(
                preds_fifo.cons(),
                labels,
                tap=preds_out_tap,
                wait=True,
                task_group=tg_fwd,
            )
            rt.finish_task_group(tg_fwd)

            tg_bwd = rt.task_group()
            rt.fill(
                embed_in.prod(),
                x_raw,
                tap=embed_in_taps[batch_idx],
                task_group=tg_bwd,
            )
            rt.fill(
                embed_wt_fifo.prod(),
                wt_embed_pair,
                tap=embed_src_tap,
                task_group=tg_bwd,
            )
            rt.drain(
                embed_wt_out_fifo.cons(),
                wt_embed_pair,
                tap=embed_dst_tap,
                task_group=tg_bwd,
            )
            done_tap = TensorAccessPattern(
                (1, window_batches * 2 * B + 1),
                window_batches * 2 * B,
                [1, 1],
                [0, 1],
            )
            rt.drain(done_fifo.cons(), labels, tap=done_tap, wait=True, task_group=tg_bwd)
            if batch_idx == window_batches - 1:
                if use_residual_drainback:
                    for wt_out_ddr, wt_out_tap in zip(res_wt_out_ddrs, res_wt_out_taps):
                        rt.drain(
                            wt_out_ddr.cons(),
                            wt_res,
                            tap=wt_out_tap,
                            task_group=tg_bwd,
                        )
                rt.drain(head_wt_out.cons(), wt_head, wait=True, task_group=tg_bwd)
            rt.finish_task_group(tg_bwd)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
