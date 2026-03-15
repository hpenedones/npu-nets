from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    InstsBinArtifact,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
    XclbinArtifact,
)

from resmlp.artifact_utils import sgd_lr_token, source_fingerprint
from simplecnn.config import (
    BATCH_SIZE,
    C1,
    C2,
    C3,
    HEAD_W_ELEMS,
    IMG_ELEMS,
    LABELS_IO_ELEMS,
    TOTAL_WEIGHT_ELEMS,
)


class SimpleCNNTrainingPipeline(AIEOperatorBase):
    def __init__(self, sgd_lr=0.0005, context=None):
        self.B = BATCH_SIZE
        self.sgd_lr = sgd_lr
        super().__init__(context=context)

    def get_artifacts(self, prefix="simplecnn_training_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        kernels_dir = project_dir / "aie_kernels"

        lr_tag = sgd_lr_token(self.sgd_lr)
        kernel_fp = source_fingerprint(
            operator_dir / "config.py",
            kernels_dir / "conv1_train.cc",
            kernels_dir / "conv2_train.cc",
            kernels_dir / "conv3_train.cc",
            kernels_dir / "gap_pool.cc",
            kernels_dir / "copy_buffer.cc",
            kernels_dir / "simple_head.cc",
        )
        design_fp = source_fingerprint(
            operator_dir / "training_design.py",
            operator_dir / "training_op.py",
            operator_dir / "config.py",
        )
        build_fp = source_fingerprint(
            operator_dir / "training_design.py",
            operator_dir / "training_op.py",
            operator_dir / "config.py",
            kernels_dir / "conv1_train.cc",
            kernels_dir / "conv2_train.cc",
            kernels_dir / "conv3_train.cc",
            kernels_dir / "gap_pool.cc",
            kernels_dir / "copy_buffer.cc",
            kernels_dir / "simple_head.cc",
        )

        archive_name = (
            f"simplecnn_training_kernels_c1{C1}_c2{C2}_c3{C3}_{lr_tag}_{kernel_fp}.a"
        )
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}b{self.B}_{lr_tag}_{build_fp}.mlir",
            import_path=operator_dir / "training_design.py",
            callback_fn="simplecnn_training_pipeline",
            callback_kwargs={"archive_name": archive_name, "sgd_lr": self.sgd_lr},
            requires_context=False,
        )

        def copy_flags(total: int, name: str):
            return [f"-DCOPY_TOTAL={total}", f"-DCOPY_BUFFER_NAME={name}"]

        head_flags = [
            f"-DBATCH_SIZE={self.B}",
            f"-DDIM_H={C3}",
            "-DNUM_CLASSES=10",
            f"-DSGD_LR={self.sgd_lr:.8g}f",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}b{self.B}_{lr_tag}_{build_fp}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    archive_name,
                    depends=[
                        KernelObjectArtifact.new(
                            f"conv1_{lr_tag}_{kernel_fp}.o",
                            extra_flags=[f"-DSGD_LR={self.sgd_lr:.8g}f"],
                            depends=[SourceArtifact.new(kernels_dir / "conv1_train.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"conv2_{lr_tag}_{kernel_fp}.o",
                            extra_flags=[f"-DSGD_LR={self.sgd_lr:.8g}f"],
                            depends=[SourceArtifact.new(kernels_dir / "conv2_train.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"conv3_{lr_tag}_{kernel_fp}.o",
                            extra_flags=[f"-DSGD_LR={self.sgd_lr:.8g}f"],
                            depends=[SourceArtifact.new(kernels_dir / "conv3_train.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"gap_{kernel_fp}.o",
                            extra_flags=[
                                f"-DBATCH_SIZE={self.B}",
                                f"-DIN_C={C3}",
                                "-DIN_H=4",
                                "-DIN_W=4",
                            ],
                            depends=[SourceArtifact.new(kernels_dir / "gap_pool.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"copy_conv1_{kernel_fp}.o",
                            extra_flags=copy_flags(36, "copy_conv1_weight_bf16"),
                            depends=[SourceArtifact.new(kernels_dir / "copy_buffer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"copy_conv2_{kernel_fp}.o",
                            extra_flags=copy_flags(288, "copy_conv2_weight_bf16"),
                            depends=[SourceArtifact.new(kernels_dir / "copy_buffer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"copy_conv3_{kernel_fp}.o",
                            extra_flags=copy_flags(1152, "copy_conv3_weight_bf16"),
                            depends=[SourceArtifact.new(kernels_dir / "copy_buffer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"copy_head_{kernel_fp}.o",
                            extra_flags=copy_flags(HEAD_W_ELEMS, "copy_head_weight_bf16"),
                            depends=[SourceArtifact.new(kernels_dir / "copy_buffer.cc")],
                        ),
                        KernelObjectArtifact.new(
                            f"simple_head_{lr_tag}_{kernel_fp}.o",
                            extra_flags=head_flags,
                            depends=[SourceArtifact.new(kernels_dir / "simple_head.cc")],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}b{self.B}_{lr_tag}_{build_fp}.bin",
            depends=[mlir_artifact],
        )
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    def set_up_runtime(self):
        self.add_kernel(
            "simplecnn_training",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("images", IMG_ELEMS)
        self.add_buffer("weights_in", TOTAL_WEIGHT_ELEMS)
        self.add_buffer("weights_out", TOTAL_WEIGHT_ELEMS)
        self.add_buffer("labels_io", LABELS_IO_ELEMS, dtype="int32")
        self.add_to_runlist(
            "simplecnn_training",
            "images",
            "weights_in",
            "weights_out",
            "labels_io",
        )
