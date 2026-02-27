"""PyTorch Dataset for two-stage orbit image classification.

On-the-fly orbit image generation directly from BIN files.
precompute_orbits.py / data/orbit_images/ 는 더 이상 필요하지 않다.

Directory layout expected:
  data/raw/normal_1200rpm/   *.BIN
  data/synthetic/1200rpm/{unbalance,misalignment,oil_whip}/  *.bin

Each __getitem__ returns:
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
from tqdm import tqdm

from src.core.rcpvms_parser import RCPVMSParser
from src.utils.orbit import (
    BEARING_PAIRS,
    DEFAULT_AXIS_LIM,
    DEFAULT_IMG_SIZE,
    make_orbit_stack,
)

# Fault type → stage-2 label
FAULT_LABELS: dict[str, int] = {
    "unbalance": 0,
    "misalignment": 1,
    "oil_whip": 2,
}

FAULT_NAMES: list[str] = ["unbalance", "misalignment", "oil_whip"]
BINARY_NAMES: list[str] = ["normal", "fault"]

DATA_ROOT = Path("data")
RPM_NORMAL_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "raw" / "normal_1200rpm",
}
RPM_FAULT_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "synthetic" / "1200rpm",
}

# Channels required for orbit images (flattened & sorted BEARING_PAIRS)
_ORBIT_CHANNELS: tuple[int, ...] = tuple(
    sorted({ch for pair in BEARING_PAIRS for ch in pair})
)  # (0, 1, 4, 5, 10, 11, 16, 17)

# Sparse list length needed by make_orbit_stack
_MAX_CH: int = max(max(p) for p in BEARING_PAIRS)


class OrbitDataset(Dataset):
    """On-the-fly orbit image dataset (direct BIN loading).

    Parameters
    ----------
    rpms             : RPM conditions to include
    include_transient: whether to include *transient* files
    axis_lim         : orbit axis half-width in mils
    img_size         : output image pixel size
    transform        : optional callable applied to the image FloatTensor
    orbit_root       : legacy parameter — ignored (kept for call-site compat)
    """

    def __init__(
        self,
        rpms: tuple[str, ...] = ("1200rpm",),
        include_transient: bool = True,
        axis_lim: float = DEFAULT_AXIS_LIM,
        img_size: int = DEFAULT_IMG_SIZE,
        transform=None,
        cache: bool = True,
        orbit_root: str | Path | None = None,  # legacy — unused
    ) -> None:
        self.axis_lim = axis_lim
        self.img_size = img_size
        self.transform = transform
        # Each entry: (bin_path, binary_label, fault_label)
        self.samples: list[tuple[Path, int, int]] = []
        self._build_index(rpms, include_transient)

        # RAM 캐시: 초기화 시 전체 orbit 이미지를 한 번만 생성
        self._cache: list[np.ndarray] | None = None
        if cache:
            self._cache = self._build_cache()

    def _build_index(
        self, rpms: tuple[str, ...], include_transient: bool
    ) -> None:
        for rpm in rpms:
            # Normal — Windows-safe glob dedup
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

    def _build_cache(self) -> list[np.ndarray]:
        """모든 orbit 이미지를 메모리에 사전 생성한다 (초기화 1회만 실행)."""
        cache: list[np.ndarray] = []
        for bin_path, _, _ in tqdm(
            self.samples, desc="Caching orbit images", unit="file", leave=False
        ):
            cache.append(self._make_orbit(bin_path))
        return cache

    def _make_orbit(self, bin_path: Path) -> np.ndarray:
        """8개 베어링 채널을 BIN 파일에서 직접 읽어 orbit 이미지 스택 (4×H×W) 반환.

        read_channel()로 필요한 채널만 선택적으로 읽는다 (~12.8 MB vs 38 MB 전체).
        make_orbit_stack()은 채널 인덱스 기반이므로 sparse list로 전달한다.
        """
        parser = RCPVMSParser(str(bin_path))
        parser.parse_header()

        # mils_per_v: BIN 헤더에서 읽거나 기본값 10.0 사용
        mils_per_v = 10.0
        if parser.header and parser.header.extra_fields:
            mpv = parser.header.extra_fields.get("mils_per_v", 0.0)
            if mpv and mpv > 0:
                mils_per_v = float(mpv)

        # 필요한 채널만 읽어 sparse list 구성 (make_orbit_stack이 인덱스로 접근)
        channels: list[np.ndarray] = [np.array([])] * (_MAX_CH + 1)
        for ch in _ORBIT_CHANNELS:
            channels[ch] = parser.read_channel(ch)

        return make_orbit_stack(
            channels, BEARING_PAIRS, mils_per_v, self.axis_lim, self.img_size
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bin_path, binary_label, fault_label = self.samples[idx]

        if self._cache is not None:
            image = self._cache[idx]
        else:
            image = self._make_orbit(bin_path)
        image_tensor = torch.from_numpy(image)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return (
            image_tensor,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(fault_label, dtype=torch.long),
        )
