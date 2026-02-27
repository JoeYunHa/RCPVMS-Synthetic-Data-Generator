"""PyTorch Dataset for 1D multi-channel time-series fault classification.

BIN 파일에서 직접 8채널을 읽는다 (signal_cache 사전계산 불필요).
RCPVMSParser.read_channel() 을 이용해 필요한 채널만 seek 읽기한다.

Directory layout expected:
  data/raw/normal_1200rpm/   *.BIN
  data/synthetic/1200rpm/{unbalance,misalignment,oil_whip}/  *.bin

Each __getitem__ returns:
  signal_tensor : FloatTensor (8, window_samples)   DC-removed, per-channel
  binary_label  : LongTensor scalar  — 0 = normal, 1 = fault
  fault_label   : LongTensor scalar  — 0/1/2 = unbalance/misalignment/oil_whip
                                        -1 for normal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.rcpvms_parser import RCPVMSParser
# Reuse label maps from orbit_dataset for consistency
from src.datasets.orbit_dataset import FAULT_LABELS

# Bearing XY channel pairs — same as BEARING_PAIRS in orbit.py (0-indexed)
SIGNAL_CHANNELS: tuple[int, ...] = (0, 1, 4, 5, 10, 11, 16, 17)

DEFAULT_WINDOW_SAMPLES: int = 40_000   # 1 s @ 40 kHz

DATA_ROOT = Path("data")
RPM_NORMAL_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "raw" / "normal_1200rpm",
}
RPM_FAULT_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "synthetic" / "1200rpm",
}


class SignalDataset(Dataset):
    """1D multi-channel windowed signal dataset (direct BIN loading).

    Parameters
    ----------
    rpms              : RPM conditions to include
    window_samples    : samples per window (default 40000 = 1 s @ 40 kHz)
    include_transient : whether to include *transient* files
    training          : True  → random window start each call (augmentation)
                        False → fixed last window (reproducible evaluation)
    transform         : optional callable applied to signal FloatTensor
    """

    def __init__(
        self,
        rpms: tuple[str, ...] = ("1200rpm",),
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        include_transient: bool = True,
        training: bool = True,
        transform=None,
    ) -> None:
        self.window_samples = window_samples
        self.training = training
        self.transform = transform
        # Each entry: (bin_path, binary_label, fault_label)
        self.samples: list[tuple[Path, int, int]] = []
        self._build_index(rpms, include_transient)

    def _build_index(
        self, rpms: tuple[str, ...], include_transient: bool
    ) -> None:
        for rpm in rpms:
            # Normal — case-insensitive glob (Windows safe)
            normal_dir = RPM_NORMAL_DIRS.get(rpm)
            if normal_dir and normal_dir.exists():
                bins = sorted(
                    set(normal_dir.glob("*.BIN")) | set(normal_dir.glob("*.bin"))
                )
                for p in bins:
                    self.samples.append((p, 0, -1))

            # Faults
            fault_root = RPM_FAULT_DIRS.get(rpm)
            if fault_root and fault_root.exists():
                for fault_name, fault_idx in FAULT_LABELS.items():
                    fault_dir = fault_root / fault_name
                    if not fault_dir.exists():
                        continue
                    for p in sorted(fault_dir.glob("*.bin")):
                        if not include_transient and "transient" in p.name:
                            continue
                        self.samples.append((p, 1, fault_idx))

    def _read_signal(self, bin_path: Path) -> np.ndarray:
        """8채널을 BIN 파일에서 직접 읽어 (8, N) float32 배열 반환.

        RCPVMSParser.read_channel()은 채널 단위 seek 읽기를 사용하므로
        파일 전체(38 MB)가 아닌 채널당 ~1.6 MB만 읽는다.
        """
        parser = RCPVMSParser(str(bin_path))
        arrays = []
        for ch in SIGNAL_CHANNELS:
            data = parser.read_channel(ch)
            if len(data) == 0:
                raise RuntimeError(
                    f"Empty channel {ch} in {bin_path.name}"
                )
            arrays.append(data.astype(np.float32))

        # Trim to common length (should all be equal)
        min_len = min(len(a) for a in arrays)
        sig = np.stack([a[:min_len] for a in arrays], axis=0)  # (8, N)

        # DC removal per channel
        sig -= sig.mean(axis=1, keepdims=True)
        return sig

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bin_path, binary_label, fault_label = self.samples[idx]

        sig = self._read_signal(bin_path)   # (8, N_total)
        n_total = sig.shape[1]

        if n_total <= self.window_samples:
            # Pad if file is shorter than one window
            pad = self.window_samples - n_total
            sig_window = np.pad(sig, ((0, 0), (0, pad)))
        elif self.training:
            # Random window start for augmentation
            max_start = n_total - self.window_samples
            start = int(np.random.randint(0, max_start + 1))
            sig_window = sig[:, start : start + self.window_samples]
        else:
            # Fixed: last 1-second window (consistent with validate_synthetic.py)
            start = n_total - self.window_samples
            sig_window = sig[:, start : start + self.window_samples]

        signal_tensor = torch.from_numpy(sig_window.copy())

        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)

        return (
            signal_tensor,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(fault_label, dtype=torch.long),
        )
