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

from resmlp.training_design import ROWS_PER_COL

class TrainingPipeline(AIEOperatorBase):
    """32-tile pipeline that executes both forward and backward pass with on-chip SGD."""

    def __init__(self, H=160, B=8, num_cols=8, context=None):
        self.H = H
        self.B = B
        self.num_cols = num_cols
        super().__init__(context=context)

    def get_artifacts(self, prefix="resmlp_training_"):
        operator_dir = Path(__file__).parent
        project_dir = operator_dir.parent
        H, B, cols = self.H, self.B, self.num_cols

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{prefix}{cols}cols_{B}x{H}.mlir",
            import_path=operator_dir / "training_design.py",
            callback_fn="training_pipeline",
            callback_kwargs={"H": H, "B": B, "num_cols": cols},
            requires_context=False,
        )

        kernel_flags = [
            f"-DDIM_M={B}",
            f"-DDIM_K={H}",
            f"-DDIM_N={H}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        xclbin_artifact = XclbinArtifact.new(
            f"{prefix}{cols}cols_{B}x{H}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    "resmlp_training_kernels.a",
                    depends=[
                        KernelObjectArtifact.new(
                            "matmul_relu_skip.o",
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    project_dir / "aie_kernels" / "matmul_relu_skip.cc"
                                )
                            ],
                        ),
                        KernelObjectArtifact.new(
                            "residual_backward.o",
                            extra_flags=kernel_flags,
                            depends=[
                                SourceArtifact.new(
                                    project_dir / "aie_kernels" / "residual_backward.cc"
                                )
                            ],
                        ),
                        KernelObjectArtifact.new(
                            "copy_activation.o",
                            extra_flags=[f"-DDIM_M={B}", f"-DDIM_K={H}"],
                            depends=[
                                SourceArtifact.new(
                                    project_dir / "aie_kernels" / "copy_activation.cc"
                                )
                            ],
                        ),
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{prefix}{cols}cols_{B}x{H}.bin",
            depends=[mlir_artifact],
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        xclbin, insts = self.get_artifacts()
        self.xclbin_artifact = xclbin
        self.insts_artifact = insts
        self.add_artifacts([xclbin, insts])

    def set_up_runtime(self):
        H, B, cols = self.H, self.B, self.num_cols
        num_tiles = cols * ROWS_PER_COL
        
        self.add_kernel(
            "training_pipeline",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("act_in", B * H)
        self.add_buffer("weights_in", num_tiles * H * H)
        self.add_buffer("act_out", B * H)
        self.add_buffer("grad_in", B * H)
        self.add_buffer("grad_out", B * H)
        self.add_to_runlist(
            "training_pipeline",
            "act_in",
            "weights_in",
            "act_out",
            "grad_in",
            "grad_out",
        )
