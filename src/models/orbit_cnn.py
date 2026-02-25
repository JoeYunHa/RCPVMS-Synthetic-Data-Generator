"""Two-stage hierarchical orbit CNN for RCP VMS fault classification.

Stage 1 (binary_head) : normal (0)  vs  fault (1)
Stage 2 (fault_head)  : unbalance (0) / misalignment (1) / oil_whip (2)

Architecture
------------
- Shared backbone: stem (7×7) → 4 residual blocks with stride-2 downsampling
  Input 4×256×256  →  256-D feature vector after global average pooling
- Two independent FC heads, each with one hidden layer + dropout

Training loss
-------------
  L_total = L_binary  +  λ · L_fault (averaged over fault-only samples)

Inference
---------
  binary_pred, fault_pred = model.predict(x)
  fault_pred == -1  for samples predicted as normal
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    """Two-layer residual block with optional channel projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _ConvBnRelu(in_ch, out_ch, kernel=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
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

class OrbitBackbone(nn.Module):
    """Shared feature extractor for 4-channel (N_bearings × H × W) orbit stacks.

    Spatial pyramid : 256 → 128 → 64 → 32 → 16 → 8
    Output dimension: 256-D feature vector (after global average pool)
    """

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.stem    = _ConvBnRelu(in_channels, 32, kernel=7, stride=2)  # →128
        self.layer1  = _ResBlock(32,  64,  stride=2)                      # →64
        self.layer2  = _ResBlock(64,  128, stride=2)                      # →32
        self.layer3  = _ResBlock(128, 256, stride=2)                      # →16
        self.layer4  = _ResBlock(256, 256, stride=2)                      # →8
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)  # (B, 256)


# ---------------------------------------------------------------------------
# Classifier heads
# ---------------------------------------------------------------------------

def _make_head(feat_dim: int, n_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feat_dim, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, n_classes),
    )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HierarchicalOrbitCNN(nn.Module):
    """Two-stage hierarchical CNN for RCP VMS orbit images.

    Parameters
    ----------
    in_channels : number of input channels (= number of bearing pairs, 4)
    dropout     : dropout rate in classifier heads (default 0.4)

    Forward
    -------
    binary_logits, fault_logits = model(x)   # (B, 2), (B, 3)

    Inference helper
    ----------------
    binary_pred, fault_pred = model.predict(x)
    # fault_pred[i] == -1  if  binary_pred[i] == 0 (normal)
    """

    def __init__(self, in_channels: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.backbone    = OrbitBackbone(in_channels)
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
        binary_pred : LongTensor (B,)  — 0 = normal, 1 = fault
        fault_pred  : LongTensor (B,)  — 0/1/2 = fault type; -1 for normals
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


# ---------------------------------------------------------------------------
# Combined training loss
# ---------------------------------------------------------------------------

class HierarchicalLoss(nn.Module):
    """L_total = L_binary  +  λ · L_fault (fault-only samples only).

    Parameters
    ----------
    lambda_fault        : weight for the fault-type CE term (default 1.0)
    binary_class_weight : optional 1-D FloatTensor [w_normal, w_fault] passed
                          to CrossEntropyLoss for binary stage.  Use this
                          instead of WeightedRandomSampler to handle class
                          imbalance while keeping natural batch distribution.
    """

    def __init__(
        self,
        lambda_fault: float = 1.0,
        binary_class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_fault = lambda_fault
        self.ce_binary = nn.CrossEntropyLoss(
            weight=binary_class_weight, label_smoothing=label_smoothing
        )
        self.ce_fault = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        binary_logits: torch.Tensor,
        fault_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        fault_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_loss, binary_loss, fault_loss).

        fault_loss is 0.0 when no fault samples exist in the batch.
        """
        l_binary = self.ce_binary(binary_logits, binary_labels)

        fault_mask = binary_labels == 1
        if fault_mask.any():
            l_fault = self.ce_fault(
                fault_logits[fault_mask], fault_labels[fault_mask]
            )
        else:
            l_fault = torch.tensor(0.0, device=binary_logits.device)

        l_total = l_binary + self.lambda_fault * l_fault
        return l_total, l_binary, l_fault
