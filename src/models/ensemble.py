"""Ensemble of HierarchicalSignalCNN (1D) + HierarchicalOrbitCNN (2D).

두 모델의 logit을 결합하여 최종 이진 / 결함 분류를 수행한다.

Method A  (현재 기본): 고정 가중치 평균
  logit_final = α · logit_1d  +  (1 - α) · logit_2d     (default α = 0.5)

Method B  (고도화 대상): 학습 가능한 소프트맥스 가중치
  w = softmax([w₁, w₂])
  logit_final = w[0] · logit_1d  +  w[1] · logit_2d
  → 두 backbone은 frozen, 가중치 파라미터만 학습

사용법
------
# Method A — 즉시 사용 가능
model = EnsembleCNN(signal_ckpt="...", orbit_ckpt="...", method="A")
signal_batch   # (B, 8,  40000)
orbit_batch    # (B, 4, 256, 256)
binary_logits, fault_logits = model(signal_batch, orbit_batch)
binary_pred, fault_pred     = model.predict(signal_batch, orbit_batch)

# Method B — fine-tune 후 사용 (train_ensemble.py)
model = EnsembleCNN(signal_ckpt="...", orbit_ckpt="...", method="B")
# model.signal_cnn, model.orbit_cnn 은 frozen
# model.log_weights 만 학습 가능
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.signal_cnn import HierarchicalSignalCNN
from src.models.orbit_cnn import HierarchicalOrbitCNN


def _load_model(
    model: nn.Module,
    ckpt_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """체크포인트를 로드하고 eval 모드로 설정한다."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


class EnsembleCNN(nn.Module):
    """1D Signal CNN + 2D Orbit CNN 앙상블 모델.

    Parameters
    ----------
    signal_ckpt  : path to HierarchicalSignalCNN checkpoint (.pt)
    orbit_ckpt   : path to HierarchicalOrbitCNN  checkpoint (.pt)
    method       : "A" = fixed average (default),  "B" = learnable weights
    alpha        : fixed weight for 1D model when method="A" (default 0.5)
    device       : inference device
    """

    def __init__(
        self,
        signal_ckpt: str | Path,
        orbit_ckpt: str | Path,
        method: str = "A",
        alpha: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.method = method.upper()
        self.alpha  = alpha

        # ── 두 모델 로드 ──────────────────────────────────────────────────
        self.signal_cnn = _load_model(
            HierarchicalSignalCNN(in_channels=8), signal_ckpt, device
        )
        self.orbit_cnn = _load_model(
            HierarchicalOrbitCNN(in_channels=4), orbit_ckpt, device
        )

        # Method B: 학습 가능한 log-weights (초기값 = 균등 가중치)
        # log_weights[0] → 1D 모델 가중치,  log_weights[1] → 2D 모델 가중치
        if self.method == "B":
            self.log_weights = nn.Parameter(torch.zeros(2))
            # backbone 은 frozen
            for p in self.signal_cnn.parameters():
                p.requires_grad_(False)
            for p in self.orbit_cnn.parameters():
                p.requires_grad_(False)
        else:
            # Method A: 파라미터 없음, 모든 weight frozen
            for p in self.parameters():
                p.requires_grad_(False)

    def _get_weights(self) -> tuple[float, float]:
        """(w_1d, w_2d) 반환 — method에 따라 고정 또는 학습된 값."""
        if self.method == "B":
            w = F.softmax(self.log_weights, dim=0)
            return w[0].item(), w[1].item()
        return self.alpha, 1.0 - self.alpha

    def forward(
        self,
        signal_batch: torch.Tensor,   # (B, 8, 40000)
        orbit_batch: torch.Tensor,    # (B, 4, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """앙상블 logit 반환: (binary_logits, fault_logits).

        Method A: w₁·logit_1d + w₂·logit_2d  (w₁+w₂=1, 고정)
        Method B: softmax([w₁,w₂])·[logit_1d, logit_2d]  (학습됨)
        """
        # 1D branch
        bin_1d, flt_1d = self.signal_cnn(signal_batch)

        # 2D branch
        bin_2d, flt_2d = self.orbit_cnn(orbit_batch)

        if self.method == "B":
            w = F.softmax(self.log_weights, dim=0)
            w1, w2 = w[0], w[1]
        else:
            w1 = torch.tensor(self.alpha,           device=signal_batch.device)
            w2 = torch.tensor(1.0 - self.alpha,     device=signal_batch.device)

        binary_logits = w1 * bin_1d + w2 * bin_2d
        fault_logits  = w1 * flt_1d + w2 * flt_2d

        return binary_logits, fault_logits

    @torch.no_grad()
    def predict(
        self,
        signal_batch: torch.Tensor,
        orbit_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """게이팅된 2단계 예측.

        Returns
        -------
        binary_pred : LongTensor (B,) — 0 = normal, 1 = fault
        fault_pred  : LongTensor (B,) — 0/1/2 = fault type; -1 = normal
        """
        binary_logits, fault_logits = self.forward(signal_batch, orbit_batch)
        binary_pred = binary_logits.argmax(dim=1)
        fault_pred  = fault_logits.argmax(dim=1)
        fault_pred  = torch.where(
            binary_pred == 1,
            fault_pred,
            torch.full_like(fault_pred, -1),
        )
        return binary_pred, fault_pred

    def weights_info(self) -> str:
        w1, w2 = self._get_weights()
        return f"method={self.method}  w_1d={w1:.4f}  w_2d={w2:.4f}"
