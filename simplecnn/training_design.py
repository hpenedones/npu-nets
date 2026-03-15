import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern

from simplecnn.config import (
    ACT1_ELEMS,
    ACT2_ELEMS,
    ACT3_ELEMS,
    BATCH_SIZE,
    C1,
    C2,
    C3,
    CONV1_CKPT_ELEMS,
    CONV1_W_ELEMS,
    CONV2_CKPT_ELEMS,
    CONV2_W_ELEMS,
    CONV3_CKPT_ELEMS,
    CONV3_OUT_H,
    CONV3_OUT_W,
    CONV3_W_ELEMS,
    HEAD_W_ELEMS,
    IMG_ELEMS,
    LABELS_IO_ELEMS,
    N_CLASSES,
    POOLED_ELEMS,
    TOTAL_WEIGHT_ELEMS,
    WEIGHT_OFFSETS,
)
def simplecnn_training_pipeline(archive_name, sgd_lr=0.0005):
    del sgd_lr

    img_ty = np.ndarray[(IMG_ELEMS,), np.dtype[bfloat16]]
    act1_ty = np.ndarray[(ACT1_ELEMS,), np.dtype[bfloat16]]
    act2_ty = np.ndarray[(ACT2_ELEMS,), np.dtype[bfloat16]]
    act3_ty = np.ndarray[(ACT3_ELEMS,), np.dtype[bfloat16]]
    pooled_ty = np.ndarray[(POOLED_ELEMS,), np.dtype[bfloat16]]

    conv1_w_ty = np.ndarray[(CONV1_W_ELEMS,), np.dtype[bfloat16]]
    conv2_w_ty = np.ndarray[(CONV2_W_ELEMS,), np.dtype[bfloat16]]
    conv3_w_ty = np.ndarray[(CONV3_W_ELEMS,), np.dtype[bfloat16]]
    head_w_ty = np.ndarray[(HEAD_W_ELEMS,), np.dtype[bfloat16]]
    host_weights_ty = np.ndarray[(TOTAL_WEIGHT_ELEMS,), np.dtype[bfloat16]]

    ckpt1_ty = np.ndarray[(CONV1_CKPT_ELEMS,), np.dtype[bfloat16]]
    ckpt2_ty = np.ndarray[(CONV2_CKPT_ELEMS,), np.dtype[bfloat16]]
    ckpt3_ty = np.ndarray[(CONV3_CKPT_ELEMS,), np.dtype[bfloat16]]

    labels_ty = np.ndarray[(BATCH_SIZE,), np.dtype[np.int32]]
    labels_io_ty = np.ndarray[(LABELS_IO_ELEMS,), np.dtype[np.int32]]
    d_logits_ty = np.ndarray[(BATCH_SIZE * N_CLASSES,), np.dtype[bfloat16]]

    conv1_fwd = Kernel(
        "conv1_forward_relu_bf16",
        archive_name,
        [img_ty, conv1_w_ty, act1_ty, ckpt1_ty],
    )
    conv1_bwd = Kernel(
        "conv1_backward_update_bf16",
        archive_name,
        [ckpt1_ty, conv1_w_ty, act1_ty],
    )
    conv2_fwd = Kernel(
        "conv2_forward_relu_bf16",
        archive_name,
        [act1_ty, conv2_w_ty, act2_ty, ckpt2_ty],
    )
    conv2_bwd = Kernel(
        "conv2_backward_update_bf16",
        archive_name,
        [ckpt2_ty, conv2_w_ty, act2_ty, act1_ty],
    )
    conv3_fwd = Kernel(
        "conv3_forward_relu_bf16",
        archive_name,
        [act2_ty, conv3_w_ty, act3_ty, ckpt3_ty],
    )
    conv3_bwd = Kernel(
        "conv3_backward_update_bf16",
        archive_name,
        [ckpt3_ty, conv3_w_ty, act3_ty, act2_ty],
    )
    gap_fwd = Kernel(
        "gap_forward_bf16",
        archive_name,
        [act3_ty, pooled_ty],
    )
    gap_bwd = Kernel(
        "gap_backward_bf16",
        archive_name,
        [pooled_ty, act3_ty],
    )
    head_fwd = Kernel(
        "simple_head_forward_loss_bf16",
        archive_name,
        [pooled_ty, head_w_ty, labels_ty, d_logits_ty, labels_ty],
    )
    head_bwd = Kernel(
        "simple_head_backward_bf16",
        archive_name,
        [pooled_ty, head_w_ty, d_logits_ty, pooled_ty],
    )
    copy_conv1 = Kernel(
        "copy_conv1_weight_bf16",
        archive_name,
        [conv1_w_ty, conv1_w_ty],
    )
    copy_conv2 = Kernel(
        "copy_conv2_weight_bf16",
        archive_name,
        [conv2_w_ty, conv2_w_ty],
    )
    copy_conv3 = Kernel(
        "copy_conv3_weight_bf16",
        archive_name,
        [conv3_w_ty, conv3_w_ty],
    )
    copy_head = Kernel(
        "copy_head_weight_bf16",
        archive_name,
        [head_w_ty, head_w_ty],
    )

    img_fifo = ObjectFifo(img_ty, name="images", depth=1)
    act1_fifo = ObjectFifo(act1_ty, name="act1", depth=1)
    act2_fifo = ObjectFifo(act2_ty, name="act2", depth=1)
    pooled_fifo = ObjectFifo(pooled_ty, name="pooled", depth=1)

    d_pooled_from_head_fifo = ObjectFifo(pooled_ty, name="d_pooled_from_head", depth=1)
    grad2_fifo = ObjectFifo(act2_ty, name="grad2", depth=1)
    grad1_fifo = ObjectFifo(act1_ty, name="grad1", depth=1)

    labels_fifo = ObjectFifo(labels_ty, name="labels", depth=1)
    preds_fifo = ObjectFifo(labels_ty, name="preds", depth=1)

    conv1_w_fifo = ObjectFifo(conv1_w_ty, name="conv1_w", depth=1)
    conv2_w_fifo = ObjectFifo(conv2_w_ty, name="conv2_w", depth=1)
    conv3_w_fifo = ObjectFifo(conv3_w_ty, name="conv3_w", depth=1)
    head_w_fifo = ObjectFifo(head_w_ty, name="head_w", depth=1)

    conv1_w_out_fifo = ObjectFifo(conv1_w_ty, name="conv1_w_out", depth=1)
    conv2_w_out_fifo = ObjectFifo(conv2_w_ty, name="conv2_w_out", depth=1)
    conv3_w_out_fifo = ObjectFifo(conv3_w_ty, name="conv3_w_out", depth=1)
    head_w_out_fifo = ObjectFifo(head_w_ty, name="head_w_out", depth=1)

    ckpt1_fifo = ObjectFifo(ckpt1_ty, name="ckpt1", depth=1)
    ckpt2_fifo = ObjectFifo(ckpt2_ty, name="ckpt2", depth=1)
    ckpt3_fifo = ObjectFifo(ckpt3_ty, name="ckpt3", depth=1)
    d_logits_fifo = ObjectFifo(d_logits_ty, name="d_logits", depth=1)
    act3_local_fifo = ObjectFifo(act3_ty, name="act3_local", depth=1)
    grad3_local_fifo = ObjectFifo(act3_ty, name="grad3_local", depth=1)

    def conv1_worker(
        of_img,
        of_act1,
        of_w,
        of_w_out,
        of_ckpt_prod,
        of_ckpt_cons,
        of_grad1,
        fwd_k,
        bwd_k,
        copy_k,
    ):
        w = of_w.acquire(1)
        try:
            x = of_img.acquire(1)
            y = of_act1.acquire(1)
            ckpt = of_ckpt_prod.acquire(1)
            fwd_k(x, w, y, ckpt)
            of_img.release(1)
            of_act1.release(1)
            of_ckpt_prod.release(1)

            gy = of_grad1.acquire(1)
            ckpt_read = of_ckpt_cons.acquire(1)
            bwd_k(ckpt_read, w, gy)
            of_grad1.release(1)
            of_ckpt_cons.release(1)

            w_out = of_w_out.acquire(1)
            copy_k(w, w_out)
            of_w_out.release(1)
        finally:
            of_w.release(1)

    def conv_worker(
        of_in,
        of_out,
        of_w,
        of_w_out,
        of_ckpt_prod,
        of_ckpt_cons,
        of_grad_in,
        of_grad_out,
        fwd_k,
        bwd_k,
        copy_k,
    ):
        w = of_w.acquire(1)
        try:
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            ckpt = of_ckpt_prod.acquire(1)
            fwd_k(x, w, y, ckpt)
            of_in.release(1)
            of_out.release(1)
            of_ckpt_prod.release(1)

            gy = of_grad_in.acquire(1)
            gx = of_grad_out.acquire(1)
            ckpt_read = of_ckpt_cons.acquire(1)
            bwd_k(ckpt_read, w, gy, gx)
            of_grad_in.release(1)
            of_grad_out.release(1)
            of_ckpt_cons.release(1)

            w_out = of_w_out.acquire(1)
            copy_k(w, w_out)
            of_w_out.release(1)
        finally:
            of_w.release(1)

    def conv3_worker(
        of_in,
        of_pooled_out,
        of_w,
        of_w_out,
        of_ckpt_prod,
        of_ckpt_cons,
        of_d_pooled_in,
        of_grad_out,
        of_act3_prod,
        of_act3_cons,
        of_grad3_prod,
        of_grad3_cons,
        fwd_k,
        bwd_k,
        gap_fwd_k,
        gap_bwd_k,
        copy_k,
    ):
        w = of_w.acquire(1)
        try:
            x = of_in.acquire(1)
            act3 = of_act3_prod.acquire(1)
            ckpt = of_ckpt_prod.acquire(1)
            fwd_k(x, w, act3, ckpt)
            of_in.release(1)
            of_act3_prod.release(1)
            of_ckpt_prod.release(1)

            act3_read = of_act3_cons.acquire(1)
            pooled = of_pooled_out.acquire(1)
            gap_fwd_k(act3_read, pooled)
            of_act3_cons.release(1)
            of_pooled_out.release(1)

            d_pooled = of_d_pooled_in.acquire(1)
            grad3 = of_grad3_prod.acquire(1)
            gap_bwd_k(d_pooled, grad3)
            of_d_pooled_in.release(1)
            of_grad3_prod.release(1)

            grad3_read = of_grad3_cons.acquire(1)
            grad2 = of_grad_out.acquire(1)
            ckpt_read = of_ckpt_cons.acquire(1)
            bwd_k(ckpt_read, w, grad3_read, grad2)
            of_grad3_cons.release(1)
            of_grad_out.release(1)
            of_ckpt_cons.release(1)

            w_out = of_w_out.acquire(1)
            copy_k(w, w_out)
            of_w_out.release(1)
        finally:
            of_w.release(1)

    def head_worker(
        of_pooled_in,
        of_head_w,
        of_head_w_out,
        of_labels,
        of_preds,
        of_d_pooled_out,
        of_dlogits_prod,
        of_dlogits_cons,
        head_fwd_k,
        head_bwd_k,
        copy_k,
    ):
        w = of_head_w.acquire(1)
        try:
            pooled_in = of_pooled_in.acquire(1)
            labels = of_labels.acquire(1)
            d_logits = of_dlogits_prod.acquire(1)
            preds = of_preds.acquire(1)
            head_fwd_k(pooled_in, w, labels, d_logits, preds)
            of_labels.release(1)
            of_dlogits_prod.release(1)
            of_preds.release(1)

            d_logits_read = of_dlogits_cons.acquire(1)
            d_pooled_out = of_d_pooled_out.acquire(1)
            head_bwd_k(pooled_in, w, d_logits_read, d_pooled_out)
            of_dlogits_cons.release(1)
            of_d_pooled_out.release(1)
            of_pooled_in.release(1)

            w_out = of_head_w_out.acquire(1)
            copy_k(w, w_out)
            of_head_w_out.release(1)
        finally:
            of_head_w.release(1)

    workers = [
        Worker(
            conv1_worker,
            [
                img_fifo.cons(),
                act1_fifo.prod(),
                conv1_w_fifo.cons(),
                conv1_w_out_fifo.prod(),
                ckpt1_fifo.prod(),
                ckpt1_fifo.cons(),
                grad1_fifo.cons(),
                conv1_fwd,
                conv1_bwd,
                copy_conv1,
            ],
            placement=Tile(col=0, row=2),
            stack_size=0x1000,
        ),
        Worker(
            conv_worker,
            [
                act1_fifo.cons(),
                act2_fifo.prod(),
                conv2_w_fifo.cons(),
                conv2_w_out_fifo.prod(),
                ckpt2_fifo.prod(),
                ckpt2_fifo.cons(),
                grad2_fifo.cons(),
                grad1_fifo.prod(),
                conv2_fwd,
                conv2_bwd,
                copy_conv2,
            ],
            placement=Tile(col=0, row=3),
            stack_size=0x1000,
        ),
        Worker(
            conv3_worker,
            [
                act2_fifo.cons(),
                pooled_fifo.prod(),
                conv3_w_fifo.cons(),
                conv3_w_out_fifo.prod(),
                ckpt3_fifo.prod(),
                ckpt3_fifo.cons(),
                d_pooled_from_head_fifo.cons(),
                grad2_fifo.prod(),
                act3_local_fifo.prod(),
                act3_local_fifo.cons(),
                grad3_local_fifo.prod(),
                grad3_local_fifo.cons(),
                conv3_fwd,
                conv3_bwd,
                gap_fwd,
                gap_bwd,
                copy_conv3,
            ],
            placement=Tile(col=0, row=4),
            stack_size=0x1000,
        ),
        Worker(
            head_worker,
            [
                pooled_fifo.cons(),
                head_w_fifo.cons(),
                head_w_out_fifo.prod(),
                labels_fifo.cons(),
                preds_fifo.prod(),
                d_pooled_from_head_fifo.prod(),
                d_logits_fifo.prod(),
                d_logits_fifo.cons(),
                head_fwd,
                head_bwd,
                copy_head,
            ],
            placement=Tile(col=0, row=5),
            stack_size=0x3000,
        ),
    ]

    def weight_tap(offset: int, elems: int) -> TensorAccessPattern:
        return TensorAccessPattern(
            (1, TOTAL_WEIGHT_ELEMS),
            offset,
            [1, elems],
            [0, 1],
        )

    labels_in_tap = TensorAccessPattern(
        (1, LABELS_IO_ELEMS),
        0,
        [1, BATCH_SIZE],
        [0, 1],
    )
    preds_out_tap = TensorAccessPattern(
        (1, LABELS_IO_ELEMS),
        BATCH_SIZE,
        [1, BATCH_SIZE],
        [0, 1],
    )

    rt = Runtime()
    with rt.sequence(img_ty, host_weights_ty, host_weights_ty, labels_io_ty) as (
        images,
        weights_in,
        weights_out,
        labels_io,
    ):
        rt.start(*workers)

        tg = rt.task_group()
        rt.fill(img_fifo.prod(), images, task_group=tg)
        rt.fill(conv1_w_fifo.prod(), weights_in, tap=weight_tap(WEIGHT_OFFSETS.conv1, CONV1_W_ELEMS), task_group=tg)
        rt.fill(conv2_w_fifo.prod(), weights_in, tap=weight_tap(WEIGHT_OFFSETS.conv2, CONV2_W_ELEMS), task_group=tg)
        rt.fill(conv3_w_fifo.prod(), weights_in, tap=weight_tap(WEIGHT_OFFSETS.conv3, CONV3_W_ELEMS), task_group=tg)
        rt.fill(head_w_fifo.prod(), weights_in, tap=weight_tap(WEIGHT_OFFSETS.head, HEAD_W_ELEMS), task_group=tg)
        rt.fill(labels_fifo.prod(), labels_io, tap=labels_in_tap, task_group=tg)

        rt.drain(preds_fifo.cons(), labels_io, tap=preds_out_tap, task_group=tg)
        rt.drain(
            conv1_w_out_fifo.cons(),
            weights_out,
            tap=weight_tap(WEIGHT_OFFSETS.conv1, CONV1_W_ELEMS),
            wait=True,
            task_group=tg,
        )
        rt.drain(
            conv2_w_out_fifo.cons(),
            weights_out,
            tap=weight_tap(WEIGHT_OFFSETS.conv2, CONV2_W_ELEMS),
            wait=True,
            task_group=tg,
        )
        rt.drain(
            conv3_w_out_fifo.cons(),
            weights_out,
            tap=weight_tap(WEIGHT_OFFSETS.conv3, CONV3_W_ELEMS),
            wait=True,
            task_group=tg,
        )
        rt.drain(
            head_w_out_fifo.cons(),
            weights_out,
            tap=weight_tap(WEIGHT_OFFSETS.head, HEAD_W_ELEMS),
            wait=True,
            task_group=tg,
        )
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
