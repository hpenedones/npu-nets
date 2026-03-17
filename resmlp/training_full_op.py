"""Operator wrapper for the full 100% NPU training pipeline."""

from pathlib import Path
import time

import numpy as np
import torch
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)
from iron.common.aie_device_manager import pyxrt

from resmlp.artifact_utils import (
    full_training_kernel_archive_name,
    full_training_kernel_tag,
    sgd_lr_token,
    source_fingerprint,
)
from resmlp.training_full_design import (
    EMBED_CHUNK_ROWS,
    N_CLS_PADDED,
    ROWS_PER_COL,
)


class FullTrainingPipeline(AIEOperatorBase):
    """32-tile pipeline: embed(784→H) + 30×residual(H→H) + head(H→10) + loss + SGD."""

    def __init__(
        self,
        H=32,
        B=8,
        K_EMBED=784,
        num_cols=8,
        window_batches=1,
        sgd_lr=0.005,
        context=None,
    ):
        self.H = H
        self.B = B
        self.K_EMBED = K_EMBED
        self.num_cols = num_cols
        self.num_residual = num_cols * ROWS_PER_COL - 2
        self.window_batches = window_batches
        self.sgd_lr = sgd_lr
        self._insts_synced = False
        super().__init__(context=context)

    def get_artifacts(self, prefix="full_training_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        H, B, K = self.H, self.B, self.K_EMBED
        assert K % EMBED_CHUNK_ROWS == 0
        lr_tag = sgd_lr_token(self.sgd_lr)
        kernel_tag = full_training_kernel_tag(
            project_dir,
            B=B,
            H=H,
            embed_chunk_rows=EMBED_CHUNK_ROWS,
            n_cls_padded=N_CLS_PADDED,
            sgd_lr=self.sgd_lr,
        )
        design_tag = source_fingerprint(
            operator_dir / "training_full_design.py",
            operator_dir / "training_full_op.py",
        )
        archive_name = full_training_kernel_archive_name(
            project_dir,
            B=B,
            H=H,
            embed_chunk_rows=EMBED_CHUNK_ROWS,
            n_cls_padded=N_CLS_PADDED,
            sgd_lr=self.sgd_lr,
        )
        kernels_dir = project_dir / "aie_kernels"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{B}x{H}_c{self.num_cols}_w{self.window_batches}_{lr_tag}_{design_tag}.mlir",
            import_path=operator_dir / "training_full_design.py",
            callback_fn="full_training_pipeline",
            callback_kwargs={
                "H": H,
                "B": B,
                "K_EMBED": K,
                "num_cols": self.num_cols,
                "window_batches": self.window_batches,
                "sgd_lr": self.sgd_lr,
            },
            requires_context=False,
        )

        res_kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            f"-DSGD_LR={self.sgd_lr:.8g}f",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        embed_kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K_EMBED={EMBED_CHUNK_ROWS}",
            f"-DDIM_H={H}",
            f"-DSGD_LR={self.sgd_lr:.8g}f",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        head_kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_H={H}",
            f"-DDIM_N_CLS={N_CLS_PADDED}",
            f"-DNUM_CLASSES=10",
            f"-DSGD_LR={self.sgd_lr:.8g}f",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]
        copy_embed_flags = [
            f"-DDIM_M={EMBED_CHUNK_ROWS}",
            f"-DDIM_K={H}",
            "-DCOPY_KERNEL_NAME=copy_embed_weight_bf16",
        ]
        copy_head_flags = [
            f"-DDIM_M={H}",
            f"-DDIM_K={N_CLS_PADDED}",
            "-DCOPY_KERNEL_NAME=copy_head_weight_bf16",
        ]
        copy_res_weight_flags = [
            f"-DDIM_M={H}",
            f"-DDIM_K={H}",
            "-DCOPY_KERNEL_NAME=copy_res_weight_bf16",
        ]
        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{B}x{H}_c{self.num_cols}_w{self.window_batches}_{lr_tag}_{design_tag}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"full_matmul_relu_skip_{kernel_tag}.o",
                            extra_flags=res_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "matmul_relu_skip.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"full_residual_backward_{kernel_tag}.o",
                            extra_flags=res_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "residual_backward.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"full_copy_activation_{kernel_tag}.o",
                            extra_flags=[f"-DDIM_M={B}", f"-DDIM_K={H}"],
                            depends=[SourceArtifact.new(kernels_dir / "copy_activation.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"copy_embed_weight_{kernel_tag}.o",
                            extra_flags=copy_embed_flags,
                            depends=[SourceArtifact.new(kernels_dir / "copy_activation.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"copy_head_weight_{kernel_tag}.o",
                            extra_flags=copy_head_flags,
                            depends=[SourceArtifact.new(kernels_dir / "copy_activation.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"copy_res_weight_{kernel_tag}.o",
                            extra_flags=copy_res_weight_flags,
                            depends=[SourceArtifact.new(kernels_dir / "copy_activation.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"embed_forward_{kernel_tag}.o",
                            extra_flags=embed_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "embed_forward.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"embed_backward_{kernel_tag}.o",
                            extra_flags=embed_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "embed_backward.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"head_forward_loss_{kernel_tag}.o",
                            extra_flags=head_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "head_forward_loss.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"head_backward_{kernel_tag}.o",
                            extra_flags=head_kernel_flags,
                            depends=[SourceArtifact.new(kernels_dir / "head_backward.cc")],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{B}x{H}_c{self.num_cols}_w{self.window_batches}_{lr_tag}_{design_tag}.bin",
            depends=[mlir_artifact],
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    def set_up_runtime(self):
        H, B, K = self.H, self.B, self.K_EMBED

        self.add_kernel(
            "full_training",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("x_raw", self.window_batches * B * K)
        self.add_buffer("embed_wt", 2 * K * H)
        self.add_buffer("res_wt", self.num_residual * H * H)
        self.add_buffer("head_wt", H * N_CLS_PADDED)
        self.add_buffer("labels", self.window_batches * 2 * B + 1, dtype="int32")

        self.add_to_runlist(
            "full_training",
            "x_raw",
            "embed_wt",
            "res_wt",
            "head_wt",
            "labels",
        )

    @property
    def _embed_logical_elems(self):
        return self.K_EMBED * self.H

    @property
    def _embed_logical_bytes(self):
        return self._embed_logical_elems * np.dtype(bfloat16).itemsize

    @property
    def _embed_final_offset_bytes(self):
        return self._embed_logical_bytes if self.window_batches % 2 == 1 else 0

    def read_buffer(self, buffer_name, shape, copy=False, dtype=bfloat16):
        if buffer_name != "embed_wt":
            return super().read_buffer(buffer_name, shape, copy=copy, dtype=dtype)

        requested_elems = int(np.prod(shape))
        if requested_elems != self._embed_logical_elems:
            return super().read_buffer(buffer_name, shape, copy=copy, dtype=dtype)

        mv = self.get_bo(buffer_name).map()
        arr = np.frombuffer(
            mv,
            dtype=dtype,
            count=requested_elems,
            offset=self._embed_final_offset_bytes,
        ).reshape(shape)
        return arr.copy() if copy else arr

    def write_buffer(self, buffer_name, array):
        if buffer_name == "labels":
            if isinstance(array, torch.Tensor):
                src = array.detach().cpu().numpy()
            else:
                src = np.asarray(array)
            src_bytes = src.ravel().view(np.uint8)
            bo = self.get_bo(buffer_name)
            mv = bo.map()
            dst_bytes = np.frombuffer(mv, dtype=np.uint8, count=bo.size())
            dst_bytes[:] = 0
            np.copyto(dst_bytes[: src_bytes.size], src_bytes, casting="no")
            return

        if buffer_name != "embed_wt":
            return super().write_buffer(buffer_name, array)

        if isinstance(array, torch.Tensor):
            src = array.detach().cpu().numpy()
        else:
            src = np.asarray(array)
        src_bytes = src.ravel().view(np.uint8)

        bo = self.get_bo(buffer_name)
        mv = bo.map()
        dst_bytes = np.frombuffer(mv, dtype=np.uint8, count=bo.size())

        if src_bytes.size == self._embed_logical_bytes:
            dst_bytes[:] = 0
            np.copyto(dst_bytes[: src_bytes.size], src_bytes, casting="no")
            np.copyto(
                dst_bytes[
                    self._embed_logical_bytes : self._embed_logical_bytes
                    + src_bytes.size
                ],
                src_bytes,
                casting="no",
            )
            return

        if src_bytes.size != 2 * self._embed_logical_bytes:
            raise ValueError(
                "embed_wt write expects either one logical weight tensor or the full "
                "packed ping-pong buffer"
            )

        np.copyto(dst_bytes[: src_bytes.size], src_bytes, casting="no")

    def _full_training_run_args(self):
        return (
            self.buffer_bos["x_raw"],
            self.buffer_bos["embed_wt"],
            self.buffer_bos["res_wt"],
            self.buffer_bos["head_wt"],
            self.buffer_bos["labels"],
        )

    def _sync_buffers(self, names, direction):
        for name in names:
            self.buffer_bos[name].sync(direction)

    def run_resident_window(self, *, sync_weights_to_device=False, sync_weights_from_device=False):
        _, xrt_kernel, insts_bo, insts_len = self.xrt_kernels["full_training"]
        if not self._insts_synced:
            insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._insts_synced = True

        sync_to_device = ["x_raw", "labels"]
        if sync_weights_to_device:
            sync_to_device = ["x_raw", "embed_wt", "res_wt", "head_wt", "labels"]
        self._sync_buffers(sync_to_device, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        start = time.perf_counter()
        run = xrt_kernel(3, insts_bo, insts_len, *self._full_training_run_args())
        result = run.wait()
        elapsed = time.perf_counter() - start
        if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(
                f"Kernel full_training did not complete correctly: {result}"
            )

        sync_from_device = ["labels"]
        if sync_weights_from_device:
            sync_from_device.extend(["embed_wt", "res_wt", "head_wt"])
        self._sync_buffers(
            sync_from_device,
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
        )
        return elapsed

    def sync_resident_weights_from_device(self):
        self._sync_buffers(
            ["embed_wt", "res_wt", "head_wt"],
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
        )
