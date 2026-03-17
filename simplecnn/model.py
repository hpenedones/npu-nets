from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ml_dtypes import bfloat16

from simplecnn.config import (
    C1,
    C2,
    C3,
    CONV1_W_ELEMS,
    CONV2_W_ELEMS,
    CONV3_W_ELEMS,
    HEAD_W_ELEMS,
    N_CLASSES,
    TOTAL_WEIGHT_ELEMS,
    WEIGHT_OFFSETS,
)


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(C1, C2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(C2, C3, kernel_size=3, stride=2, padding=1, bias=False)
        self.head = nn.Linear(C3, N_CLASSES, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.mean(dim=(2, 3))
        return self.head(x)

    def scale_initial_weights(self, conv_scale: float = 0.25, head_scale: float = 0.25) -> None:
        with torch.no_grad():
            self.conv1.weight.mul_(conv_scale)
            self.conv2.weight.mul_(conv_scale)
            self.conv3.weight.mul_(conv_scale)
            self.head.weight.mul_(head_scale)

    @staticmethod
    def _export_conv_weight(weight: torch.Tensor) -> np.ndarray:
        return (
            weight.detach()
            .cpu()
            .float()
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
            .astype(bfloat16)
            .reshape(-1)
        )

    @staticmethod
    def _load_conv_weight(array: np.ndarray, shape: torch.Size) -> torch.Tensor:
        out_c, in_c, k_h, k_w = shape
        return torch.from_numpy(
            np.asarray(array, dtype=np.float32)
            .reshape(out_c, k_h, k_w, in_c)
            .transpose(0, 3, 1, 2)
        )

    def export_packed_weights(self) -> np.ndarray:
        conv1 = self._export_conv_weight(self.conv1.weight)
        conv2 = self._export_conv_weight(self.conv2.weight)
        conv3 = self._export_conv_weight(self.conv3.weight)
        head = self.export_head_weight().reshape(-1)
        packed = np.concatenate([conv1, conv2, conv3, head])
        assert packed.size == TOTAL_WEIGHT_ELEMS
        return packed

    def export_head_weight(self) -> np.ndarray:
        return self.head.weight.detach().cpu().float().numpy().T.astype(bfloat16)

    def load_packed_weights(self, packed: np.ndarray) -> None:
        array = np.asarray(packed, dtype=np.float32).reshape(-1)
        if array.size != TOTAL_WEIGHT_ELEMS:
            raise ValueError(
                f"Packed weight buffer has {array.size} elements, expected {TOTAL_WEIGHT_ELEMS}"
            )

        conv1 = array[
            WEIGHT_OFFSETS.conv1 : WEIGHT_OFFSETS.conv1 + CONV1_W_ELEMS
        ]
        conv2 = array[
            WEIGHT_OFFSETS.conv2 : WEIGHT_OFFSETS.conv2 + CONV2_W_ELEMS
        ]
        conv3 = array[
            WEIGHT_OFFSETS.conv3 : WEIGHT_OFFSETS.conv3 + CONV3_W_ELEMS
        ]
        head = array[
            WEIGHT_OFFSETS.head : WEIGHT_OFFSETS.head + HEAD_W_ELEMS
        ].reshape(C3, N_CLASSES)

        with torch.no_grad():
            self.conv1.weight.copy_(self._load_conv_weight(conv1, self.conv1.weight.shape))
            self.conv2.weight.copy_(self._load_conv_weight(conv2, self.conv2.weight.shape))
            self.conv3.weight.copy_(self._load_conv_weight(conv3, self.conv3.weight.shape))
            self.head.weight.copy_(torch.from_numpy(head.T))
