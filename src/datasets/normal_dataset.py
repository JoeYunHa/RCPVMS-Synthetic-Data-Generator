"""Dataset for real normal-only BIN files (Deep SVDD training).

No synthetic or fault data is loaded.
Returns 8-channel windowed signals with RPM metadata.

Directory layout:
  data/raw/normal/           *.BIN  (RPM 미분류)
  data/raw/normal_1200rpm/   *.BIN
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.rcpvms_parser import RCPVMSParser

SIGNAL_CHANNELS: tuple[int, ...] = (0, 1, 4, 5, 10, 11, 16, 17)
DEFAULT_WINDOW_SAMPLES: int = 40_000

DATA_ROOT = Path("data")

# (directory_path, rpm_label)
# rpm_label: 0=unknown, 1=1200rpm
NORMAL_DIRS: dict[str, tuple[Path, int]] = {
    "unknown": (DATA_ROOT / "raw" / "normal",        0),
    "1200rpm": (DATA_ROOT / "raw" / "normal_1200rpm", 1),
}
RPM_LABEL_NAMES: dict[int, str] = {0: "unknown", 1: "1200rpm"}


class NormalDataset(Dataset):
    """Real measured normal-only BIN file dataset for one-class learning.

    Parameters
    ----------
    rpms           : RPM keys to include (subset of NORMAL_DIRS keys)
    window_samples : samples per window (default 40 000 = 1 s @ 40 kHz)
    training       : True → random window each call; False → fixed last window

    Returns (per __getitem__)
    -------------------------
    signal_tensor : FloatTensor (8, window_samples)  DC-removed
    rpm_label     : LongTensor scalar  0=unknown / 1=1200rpm
    """

    def __init__(
        self,
        rpms: tuple[str, ...] = ("1200rpm",),
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        training: bool = True,
    ) -> None:
        self.window_samples = window_samples
        self.training = training
        self.samples: list[tuple[Path, int]] = []
        self._build_index(rpms)

    def _build_index(self, rpms: tuple[str, ...]) -> None:
        import warnings
        for rpm_key in rpms:
            entry = NORMAL_DIRS.get(rpm_key)
            if entry is None:
                continue
            dir_path, rpm_label = entry
            if not dir_path.exists():
                warnings.warn(
                    f"NormalDataset: directory for '{rpm_key}' not found: {dir_path}"
                )
                continue
            bins = sorted(
                set(dir_path.glob("*.BIN")) | set(dir_path.glob("*.bin"))
            )
            for p in bins:
                self.samples.append((p, rpm_label))

    def _read_signal(self, bin_path: Path) -> np.ndarray:
        """Read 8 bearing channels and return (8, N) float32, DC-removed.

        Uses read_all_channels() (single file open) instead of per-channel
        read_channel() (8 separate opens) to minimise I/O overhead.
        """
        parser = RCPVMSParser(str(bin_path))
        all_ch = parser.read_all_channels()   # single open, single seek
        if not all_ch:
            raise RuntimeError(f"read_all_channels() returned empty for {bin_path.name}")
        arrays = []
        for ch in SIGNAL_CHANNELS:
            if ch >= len(all_ch) or len(all_ch[ch]) == 0:
                raise RuntimeError(f"Empty channel {ch} in {bin_path.name}")
            arrays.append(all_ch[ch].astype(np.float32))
        min_len = min(len(a) for a in arrays)
        sig = np.stack([a[:min_len] for a in arrays], axis=0)  # (8, N)
        sig -= sig.mean(axis=1, keepdims=True)                  # DC removal
        return sig

    def _window(self, sig: np.ndarray) -> np.ndarray:
        n_total = sig.shape[1]
        if n_total <= self.window_samples:
            pad = self.window_samples - n_total
            return np.pad(sig, ((0, 0), (0, pad)))
        if self.training:
            max_start = n_total - self.window_samples
            start = int(np.random.randint(0, max_start + 1))
        else:
            start = n_total - self.window_samples
        return sig[:, start: start + self.window_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        bin_path, rpm_label = self.samples[idx]
        sig = self._read_signal(bin_path)
        sig_window = self._window(sig)
        return (
            torch.from_numpy(sig_window.copy()),
            torch.tensor(rpm_label, dtype=torch.long),
        )

    def split_by_file(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[_SubsetNormalDataset, _SubsetNormalDataset]:
        """Split train/val by file index to prevent window-level leakage."""
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.samples))
        n_val   = max(1, int(len(indices) * val_ratio))
        val_idx = sorted(indices[:n_val].tolist())
        tr_idx  = sorted(indices[n_val:].tolist())
        return (
            _SubsetNormalDataset(self, tr_idx,  training=True),
            _SubsetNormalDataset(self, val_idx, training=False),
        )

    def summary(self) -> str:
        counts = {}
        for _, rpm_label in self.samples:
            counts[rpm_label] = counts.get(rpm_label, 0) + 1
        parts = [f"{RPM_LABEL_NAMES[k]}={v}" for k, v in sorted(counts.items())]
        return f"NormalDataset({len(self.samples)} files: {', '.join(parts)})"


class _SubsetNormalDataset(Dataset):
    """File-index subset of NormalDataset with independent training flag."""

    def __init__(
        self,
        base: NormalDataset,
        indices: list[int],
        training: bool,
    ) -> None:
        self.base     = base
        self.indices  = indices
        self.training = training

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        bin_path, rpm_label = self.base.samples[self.indices[idx]]
        sig = self.base._read_signal(bin_path)
        n_total = sig.shape[1]
        ws = self.base.window_samples

        if n_total <= ws:
            sig_window = np.pad(sig, ((0, 0), (0, ws - n_total)))
        elif self.training:
            start = int(np.random.randint(0, n_total - ws + 1))
            sig_window = sig[:, start: start + ws]
        else:
            sig_window = sig[:, n_total - ws:]

        return (
            torch.from_numpy(sig_window.copy()),
            torch.tensor(rpm_label, dtype=torch.long),
        )
