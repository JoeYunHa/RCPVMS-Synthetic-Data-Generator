"""1D hierarchical CNN for RCP VMS time-series fault classification.

Mirrors HierarchicalOrbitCNN architecture in 1D — shared backbone + two heads.
HierarchicalLoss and _make_head are reused directly from orbit_cnn.py.

Input  : (B, 8, 40000)  — 8 bearing channels × 1 s @ 40 kHz
Output : binary_logits (B, 2),  fault_logits (B, 3)

Architecture
------------
  Stem :  Conv1d(8→64,   k=15, s=4)  → (B,  64, 10000)
  Block1: ResBlock1D(64→128,  s=4)   → (B, 128,  2500)
  Block2: ResBlock1D(128→256, s=4)   → (B, 256,   625)
  Block3: ResBlock1D(256→256, s=4)   → (B, 256,   156)
  Block4: ResBlock1D(256→256, s=4)   → (B, 256,    39)
  GAP                                → (B, 256)
  binary_head / fault_head           (FC 256→128→n_classes, same as orbit_cnn)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Reuse head factory + loss from orbit_cnn — keeps both models consistent
from src.models.orbit_cnn import HierarchicalLoss, _make_head  # noqa: F401


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBnRelu1D(nn.Sequential):
    """Conv1d → BatchNorm1d → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock1D(nn.Module):
    """Two-layer 1D residual block with optional channel/stride projection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = _ConvBnRelu1D(in_ch, out_ch, kernel=kernel_size, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return self.relu(out + self.shortcut(x))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class SignalBackbone1D(nn.Module):
    """Shared 1D feature extractor for 8-channel bearing signals.

    Temporal pyramid:
      40000 → 10000 → 2500 → 625 → 156 → 39 → (GAP) → 1
    Output dimension: 256-D feature vector
    """

    def __init__(self, in_channels: int = 8) -> None:
        super().__init__()
        self.stem   = _ConvBnRelu1D(in_channels, 64,  kernel=15, stride=4)  # →10000
        self.layer1 = _ResBlock1D(64,  128, stride=4)                        # →2500
        self.layer2 = _ResBlock1D(128, 256, stride=4)                        # →625
        self.layer3 = _ResBlock1D(256, 256, stride=4)                        # →156
        self.layer4 = _ResBlock1D(256, 256, stride=4)                        # →39
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).squeeze(-1)  # (B, 256)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HierarchicalSignalCNN(nn.Module):
    """Two-stage hierarchical 1D CNN for RCP VMS fault classification.

    Parameters
    ----------
    in_channels : number of input channels (8 for bearing X+Y pairs)
    dropout     : dropout rate in classifier heads (default 0.4)

    Forward
    -------
    binary_logits, fault_logits = model(x)   # (B, 2), (B, 3)

    Inference helper
    ----------------
    binary_pred, fault_pred = model.predict(x)
    # fault_pred[i] == -1  if  binary_pred[i] == 0 (normal)
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.4) -> None:
        super().__init__()
        self.backbone    = SignalBackbone1D(in_channels)
        feat_dim         = self.backbone.out_dim
        self.binary_head = _make_head(feat_dim, 2, dropout)
        self.fault_head  = _make_head(feat_dim, 3, dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return self.binary_head(feats), self.fault_head(feats)

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated two-stage prediction.

        Returns
        -------
        binary_pred : LongTensor (B,) — 0 = normal, 1 = fault
        fault_pred  : LongTensor (B,) — 0/1/2 = fault type; -1 for normals
        """
        binary_logits, fault_logits = self.forward(x)
        binary_pred = binary_logits.argmax(dim=1)
        fault_pred  = fault_logits.argmax(dim=1)
        fault_pred  = torch.where(
            binary_pred == 1,
            fault_pred,
            torch.full_like(fault_pred, -1),
        )
        return binary_pred, fault_pred
