"""PyTorch Dataset for two-stage orbit image classification.

Loads pre-computed orbit image stacks (*.npy) produced by precompute_orbits.py.

Expected directory layout
-------------------------
orbit_root/
  3600rpm/
    normal/        *.npy  (shape: 4 × H × W, float32)
    unbalance/     *.npy
    misalignment/  *.npy
    oil_whip/      *.npy
  1200rpm/
    ...

Each item returned by __getitem__
----------------------------------
image        : FloatTensor (4, H, W)
binary_label : LongTensor scalar  — 0 = normal, 1 = fault
fault_label  : LongTensor scalar  — 0 / 1 / 2 = unbalance / misalignment / oil_whip
                                    -1 for normal samples (N/A for stage-2)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Fault type → stage-2 label
FAULT_LABELS: dict[str, int] = {
    "unbalance": 0,
    "misalignment": 1,
    "oil_whip": 2,
}

FAULT_NAMES: list[str] = ["unbalance", "misalignment", "oil_whip"]
BINARY_NAMES: list[str] = ["normal", "fault"]


class OrbitDataset(Dataset):
    """Two-stage orbit image classification dataset.

    Parameters
    ----------
    orbit_root       : root directory that contains RPM sub-folders
    rpms             : RPM conditions to include
    include_transient: whether to include *_transient_* files
    transform        : optional callable applied to the image FloatTensor
    """

    def __init__(
        self,
        orbit_root: str | Path,
        rpms: tuple[str, ...] = ("3600rpm", "1200rpm"),
        include_transient: bool = True,
        transform=None,
    ) -> None:
        self.orbit_root = Path(orbit_root)
        self.transform = transform
        # Each entry: (path, binary_label, fault_label)
        self.samples: list[tuple[Path, int, int]] = []
        self._build_index(rpms, include_transient)

    def _build_index(
        self, rpms: tuple[str, ...], include_transient: bool
    ) -> None:
        for rpm in rpms:
            rpm_dir = self.orbit_root / rpm
            if not rpm_dir.exists():
                continue

            # Normal samples
            normal_dir = rpm_dir / "normal"
            if normal_dir.exists():
                for p in sorted(normal_dir.glob("*.npy")):
                    self.samples.append((p, 0, -1))

            # Fault samples
            for fault_name, fault_idx in FAULT_LABELS.items():
                fault_dir = rpm_dir / fault_name
                if not fault_dir.exists():
                    continue
                for p in sorted(fault_dir.glob("*.npy")):
                    if not include_transient and "transient" in p.name:
                        continue
                    self.samples.append((p, 1, fault_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, binary_label, fault_label = self.samples[idx]

        image = np.load(path)  # (4, H, W), float32
        image_tensor = torch.from_numpy(image)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return (
            image_tensor,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(fault_label, dtype=torch.long),
        )
