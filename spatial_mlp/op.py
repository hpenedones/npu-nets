# SPDX-License-Identifier: Apache-2.0
"""
AIE operator class for the Spatial MLP pipeline.
Handles compilation artifacts, runtime buffer management, and kernel invocation.

Uses IRON's proven mm.cc for matmul (tiled layout) + custom relu_inplace.cc.
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIESpatialMLP(AIEOperatorBase):
    """
    4-layer pipelined MLP running on all 32 AIE tiles.

    Architecture:
      - 4 rows (pipeline stages) × 8 columns (parallel pipelines)
      - Each tile: zero → matmul → relu_inplace (proven IRON kernel pattern)
      - Data flows through ObjectFIFOs, never returns to DDR between layers
      - Data is in tiled (blocked) layout: 8×8 blocks for bf16 with BFP16 emulation
    """

    def __init__(
        self,
        H: int = 128,
        B: int = 16,
        num_layers: int = 4,
        num_pipelines: int = 8,
        num_batches: int = 1,
        context=None,
    ):
        self.H = H
        self.B = B
        self.num_layers = num_layers
        self.num_pipelines = num_pipelines
        self.num_batches = num_batches

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        project_dir = Path(__file__).parent.parent  # npu-spatial-nets/
        operator_dir = Path(__file__).parent  # spatial_mlp/
        iron_dir = Path(self.context.base_dir)  # IRON/

        name = (f"spatial_mlp_{self.H}h_{self.B}b_"
                f"{self.num_layers}l_{self.num_pipelines}p")

        # Source files: IRON's mm.cc + our relu_inplace.cc
        mm_source = str(iron_dir / "aie_kernels" / "aie2p" / "mm.cc")
        relu_source = str(project_dir / "aie_kernels" / "mlp_kernels.cc")

        # mm.cc compile flags: set tile dimensions and bf16 type
        mm_defines = [
            f"-DDIM_M={self.B}",
            f"-DDIM_K={self.H}",
            f"-DDIM_N={self.H}",
            "-Dbf16_bf16_ONLY",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        relu_defines = [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ]

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{name}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="spatial_mlp",
            callback_kwargs={
                "H": self.H,
                "B": self.B,
                "num_layers": self.num_layers,
                "num_pipelines": self.num_pipelines,
                "num_batches": self.num_batches,
            },
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{name}.xclbin",
            depends=[
                mlir_artifact,
                KernelArchiveArtifact.new(
                    "mlp_kernels.a",
                    depends=[
                        KernelObjectArtifact.new(
                            "mlp_mm.o",
                            extra_flags=mm_defines,
                            depends=[SourceArtifact.new(mm_source)],
                        ),
                        KernelObjectArtifact.new(
                            "mlp_relu.o",
                            extra_flags=relu_defines,
                            depends=[SourceArtifact.new(relu_source)],
                        ),
                    ],
                ),
            ],
            extra_flags=["--dynamic-objFifos"],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{name}.bin",
            depends=[mlir_artifact],
            extra_flags=["--dynamic-objFifos"],
        )

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        H, B = self.H, self.B
        np_ = self.num_pipelines
        nl = self.num_layers

        self.add_kernel(
            "mlp",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )

        # add_buffer(name, count, dtype=bfloat16) — count is element count
        input_count = np_ * B * H
        weights_count = nl * H * H
        output_count = np_ * B * H

        self.add_buffer("input", input_count)
        self.add_buffer("weights", weights_count)
        self.add_buffer("output", output_count)

        self.add_to_runlist("mlp", "input", "weights", "output")
