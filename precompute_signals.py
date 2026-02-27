#!/usr/bin/env python3
"""Pre-compute 8-channel signal arrays from BIN files.

Reads every BIN file (normal + synthetic), extracts the 8 bearing channels
used by the 1D CNN, removes DC offset, and saves as a float32 .npy file.

Usage
-----
venv\\Scripts\\python.exe precompute_signals.py
venv\\Scripts\\python.exe precompute_signals.py --rpms 3600rpm --dry-run
venv\\Scripts\\python.exe precompute_signals.py --overwrite

Output layout
-------------
data/signal_cache/
  1200rpm/
    normal/        {stem}.npy   shape (8, N_samples), float32, DC-removed
    unbalance/     {stem}.npy
    misalignment/  {stem}.npy
    oil_whip/      {stem}.npy

Storage estimate: ~12.8 MB per file × 1010 files ≈ 12.9 GB
  (8 channels × 400,000 samples × 4 bytes)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.core.rcpvms_parser import RCPVMSParser
from src.datasets.signal_dataset import SIGNAL_CHANNELS

DATA_ROOT    = Path("data")
CACHE_ROOT   = DATA_ROOT / "signal_cache"
FAULT_TYPES  = ("unbalance", "misalignment", "oil_whip")

RPM_NORMAL_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "raw" / "normal_1200rpm",
}
RPM_FAULT_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "synthetic" / "1200rpm",
}


def extract_channels(bin_path: Path) -> np.ndarray | None:
    """Read selected channels from BIN file using per-channel seeks.

    Returns (8, N_samples) float32 array with DC removed, or None on failure.
    Uses read_channel() for each channel — reads only 8/24 of the file data.
    """
    parser = RCPVMSParser(str(bin_path))
    try:
        parser.parse_header()
    except Exception as e:
        print(f"  [WARN] header parse failed: {bin_path.name} — {e}")
        return None

    arrays = []
    for ch in SIGNAL_CHANNELS:
        data = parser.read_channel(ch)
        if len(data) == 0:
            print(f"  [WARN] empty channel {ch}: {bin_path.name}")
            return None
        # DC removal
        data = data - data.mean()
        arrays.append(data.astype(np.float32))

    # Trim to common length (all channels should be equal, but be safe)
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays], axis=0)  # (8, N)


def collect_tasks(rpms: list[str]) -> list[tuple[Path, Path, str]]:
    """Build (bin_path, out_path, label) task list."""
    tasks: list[tuple[Path, Path, str]] = []

    for rpm in rpms:
        # Normal
        normal_dir = RPM_NORMAL_DIRS.get(rpm)
        if normal_dir and normal_dir.exists():
            # Use set union to avoid duplicates on case-insensitive filesystems (Windows)
            bins = sorted(set(normal_dir.glob("*.BIN")) | set(normal_dir.glob("*.bin")))
            for p in bins:
                out = CACHE_ROOT / rpm / "normal" / (p.stem + ".npy")
                tasks.append((p, out, "normal"))

        # Faults
        synthetic_dir = RPM_FAULT_DIRS.get(rpm)
        if synthetic_dir and synthetic_dir.exists():
            for fault in FAULT_TYPES:
                fault_dir = synthetic_dir / fault
                if not fault_dir.exists():
                    continue
                for p in sorted(fault_dir.glob("*.bin")):
                    out = CACHE_ROOT / rpm / fault / (p.stem + ".npy")
                    tasks.append((p, out, fault))

    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-compute 8-channel signal .npy files"
    )
    ap.add_argument(
        "--rpms", nargs="+", default=["1200rpm"],
        help="RPM conditions to process (default: both)",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Re-process already-existing output files",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without writing files",
    )
    args = ap.parse_args()

    tasks = collect_tasks(args.rpms)
    already_done = sum(1 for _, out, _ in tasks if out.exists())

    size_per_file_mb = len(SIGNAL_CHANNELS) * 400_000 * 4 / 1e6
    total_gb = len(tasks) * size_per_file_mb / 1e3
    todo = len(tasks) if args.overwrite else len(tasks) - already_done

    print(f"Tasks       : {len(tasks)}  ({already_done} already done)")
    print(f"To process  : {todo}")
    print(f"Est. storage: {total_gb:.1f} GB  ({size_per_file_mb:.1f} MB/file × {len(tasks)} files)")
    print(f"Output root : {CACHE_ROOT.resolve()}")

    if args.dry_run:
        print("\nFirst 10 tasks:")
        for p, out, label in tasks[:10]:
            status = "SKIP" if (out.exists() and not args.overwrite) else "TODO"
            print(f"  [{status}] {label:15s}  {p.name}  →  {out.name}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    n_ok = n_skip = n_fail = 0
    for bin_path, out_path, _ in tqdm(tasks, unit="file"):
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        arr = extract_channels(bin_path)
        if arr is None:
            n_fail += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
        n_ok += 1

    print(f"\nDone.  written={n_ok}  skipped={n_skip}  failed={n_fail}")
    print(f"Signal cache saved to: {CACHE_ROOT.resolve()}")


if __name__ == "__main__":
    main()
