"""Deep SVDD (Support Vector Data Description) — one-class anomaly detection.

Two-stage training:
  Stage 1 — SignalAutoencoder  : encoder pre-training via MSE reconstruction
  Stage 2 — DeepSVDD           : hypersphere compression (SVDD loss)

Collapse prevention (Ruff et al., ICML 2018):
  - Projection head: Linear(..., bias=False)
  - Projection head: no BatchNorm
  - Center c: fixed after initialization, never updated by optimizer
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Shared building blocks ────────────────────────────────────────────────────

class _ConvBnRelu1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 9, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 9) -> None:
        super().__init__()
        p = kernel_size // 2
        self.conv1 = _ConvBnRelu1D(in_ch, out_ch, kernel=kernel_size, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=p, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class _DecoderBlock(nn.Module):
    """Upsample × scale then Conv1d (decoder building block)."""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 4) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=scale, mode="linear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 9, padding=4, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ── Encoder (shared by AE and SVDD) ──────────────────────────────────────────

class SignalEncoder(nn.Module):
    """1D ResNet encoder: (B, 8, W) → (B, latent_dim).

    Temporal pyramid:  40000 → 10000 → 2500 → 625 → 156 → 39 → GAP → 256-D → latent_dim-D
    """

    def __init__(self, in_channels: int = 8, latent_dim: int = 128) -> None:
        super().__init__()
        self.stem   = _ConvBnRelu1D(in_channels, 64,  kernel=15, stride=4)
        self.layer1 = _ResBlock1D(64,  128, stride=4)
        self.layer2 = _ResBlock1D(128, 256, stride=4)
        self.layer3 = _ResBlock1D(256, 256, stride=4)
        self.layer4 = _ResBlock1D(256, 256, stride=4)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        # bias=False: collapse prevention — bias in this layer provides a
        # free additive shortcut that undermines the SVDD projection head's
        # bias=False constraint (Ruff et al., 2018)
        self.proj   = nn.Linear(256, latent_dim, bias=False)
        self.out_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)   # (B, 256)
        return self.proj(x)             # (B, latent_dim)


# ── Stage 1: Autoencoder ──────────────────────────────────────────────────────

class SignalDecoder(nn.Module):
    """Mirror decoder: (B, latent_dim) → (B, out_channels, out_length).

    Upsampling path: 39 → 156 → 624 → 2496 → 9984 → out_length
    """

    _BASE_LEN: int = 39   # temporal length after 4 stride-4 downsamples of 40000

    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 8,
        out_length: int = 40_000,
    ) -> None:
        super().__init__()
        self.out_length = out_length
        self.fc     = nn.Linear(latent_dim, 256 * self._BASE_LEN)
        self.layer1 = _DecoderBlock(256, 256, scale=4)   # 39   → 156
        self.layer2 = _DecoderBlock(256, 128, scale=4)   # 156  → 624
        self.layer3 = _DecoderBlock(128, 64,  scale=4)   # 624  → 2496
        self.layer4 = _DecoderBlock(64,  32,  scale=4)   # 2496 → 9984
        self.final  = nn.Sequential(
            nn.Upsample(size=out_length, mode="linear", align_corners=False),
            nn.Conv1d(32, out_channels, 15, padding=7, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), 256, self._BASE_LEN)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.final(x)   # (B, out_channels, out_length)


class SignalAutoencoder(nn.Module):
    """Stage-1 autoencoder for encoder representation pre-training.

    forward() returns (reconstruction, latent_vector).
    Only reconstruction MSE is minimised during Stage 1.
    """

    def __init__(
        self,
        in_channels: int = 8,
        latent_dim: int = 128,
        window_samples: int = 40_000,
    ) -> None:
        super().__init__()
        self.encoder = SignalEncoder(in_channels, latent_dim)
        self.decoder = SignalDecoder(latent_dim, in_channels, window_samples)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ── Stage 2: Deep SVDD ────────────────────────────────────────────────────────

class DeepSVDD(nn.Module):
    """One-class anomaly detector via minimum enclosing hypersphere.

    Collapse prevention:
      - proj head: bias=False, no BatchNorm
      - center c:  fixed buffer (never receives gradients)

    Parameters
    ----------
    in_channels : signal channels (default 8)
    latent_dim  : SignalEncoder output dimension
    svdd_dim    : hypersphere embedding dimension (bias=False projection)
    """

    def __init__(
        self,
        in_channels: int = 8,
        latent_dim: int = 128,
        svdd_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = SignalEncoder(in_channels, latent_dim)
        # No bias, no BN — collapse prevention
        self.proj = nn.Linear(latent_dim, svdd_dim, bias=False)
        # Fixed center (not a Parameter)
        self.register_buffer("center", torch.zeros(svdd_dim))
        self._center_initialized = False

    @torch.no_grad()
    def initialize_center(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        eps: float = 1e-4,
    ) -> None:
        """Compute c = mean{φ(x) | x ∈ train_normal}, then fix it.

        Entries near zero are nudged by ±eps to prevent trivial collapse.
        """
        self.eval()
        embeddings: list[torch.Tensor] = []
        for signals, _ in loader:
            z = self.proj(self.encoder(signals.to(device)))
            embeddings.append(z.cpu())
        c = torch.cat(embeddings, dim=0).mean(dim=0)
        # Nudge near-zero components
        c[(c.abs() < eps) & (c >= 0)] =  eps
        c[(c.abs() < eps) & (c <  0)] = -eps
        self.center.copy_(c.to(device))
        self._center_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return squared distances ||φ(xᵢ) - c||² for each sample."""
        if not self._center_initialized:
            raise RuntimeError(
                "Hypersphere center c is not initialized. "
                "Call initialize_center() before forward(), "
                "or load it from a checkpoint via model.center.copy_()."
            )
        z = self.proj(self.encoder(x))
        return torch.sum((z - self.center) ** 2, dim=1)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2 distance (non-negative scalar) per sample."""
        return self.forward(x).sqrt()

    def load_encoder_from_autoencoder(self, ae_state_dict: dict) -> None:
        """Transfer encoder weights from a saved SignalAutoencoder state_dict."""
        encoder_sd = {
            k[len("encoder."):]: v
            for k, v in ae_state_dict.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Encoder weight mismatch — missing: {missing}, unexpected: {unexpected}"
            )
