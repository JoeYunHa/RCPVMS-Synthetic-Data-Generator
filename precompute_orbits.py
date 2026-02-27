#!/usr/bin/env python3
"""Pre-compute orbit image stacks from raw BIN files.

Reads every BIN file in data/raw/normal_* and data/synthetic/,
extracts a 4-bearing orbit stack (4 × 256 × 256, float32), and
writes it as a .npy file under data/orbit_images/.

Usage
-----
venv\\Scripts\\python.exe precompute_orbits.py
venv\\Scripts\\python.exe precompute_orbits.py --rpms 3600rpm --axis_lim 3.0
venv\\Scripts\\python.exe precompute_orbits.py --dry-run

Output layout
-------------
data/orbit_images/
  1200rpm/
    normal/          {stem}.npy   (shape 4×256×256, float32)
    unbalance/
    misalignment/
    oil_whip/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.core.rcpvms_parser import RCPVMSParser
from src.utils.orbit import (
    BEARING_PAIRS,
    DEFAULT_AXIS_LIM,
    DEFAULT_IMG_SIZE,
    make_orbit_stack,
)

DATA_ROOT  = Path("data")
ORBIT_ROOT = DATA_ROOT / "orbit_images"

RPM_NORMAL_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "raw" / "normal_1200rpm",
}
RPM_FAULT_DIRS: dict[str, Path] = {
    "1200rpm": DATA_ROOT / "synthetic" / "1200rpm",
}
FAULT_TYPES = ("unbalance", "misalignment", "oil_whip")

# Required channel count (highest index in BEARING_PAIRS + 1)
_MIN_CHANNELS = max(max(p) for p in BEARING_PAIRS) + 1


def _get_mils_per_v(parser: RCPVMSParser) -> float:
    if parser.header and parser.header.extra_fields:
        mpv = parser.header.extra_fields.get("mils_per_v", 0.0)
        if mpv and mpv > 0:
            return float(mpv)
    return 10.0


def process_file(
    bin_path: Path,
    out_path: Path,
    axis_lim: float,
    img_size: int,
    dry_run: bool,
) -> bool:
    """Parse one BIN file and save its orbit stack as .npy.

    Returns True on success (or if the file was already processed).
    """
    if out_path.exists():
        return True  # already done

    parser = RCPVMSParser(str(bin_path))
    try:
        parser.parse_header()
    except Exception as e:
        print(f"  [WARN] header parse failed: {bin_path.name} — {e}")
        return False

    channels = parser.read_all_channels()
    if len(channels) < _MIN_CHANNELS:
        print(f"  [WARN] only {len(channels)} channels, need {_MIN_CHANNELS}: {bin_path.name}")
        return False

    mils_per_v = _get_mils_per_v(parser)
    stack = make_orbit_stack(channels, BEARING_PAIRS, mils_per_v, axis_lim, img_size)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, stack)

    return True


def collect_tasks(rpms: list[str]) -> list[tuple[Path, Path, str]]:
    """Build the full (bin_path, out_path, label) task list."""
    tasks: list[tuple[Path, Path, str]] = []

    for rpm in rpms:
        # Normal
        normal_dir = RPM_NORMAL_DIRS.get(rpm)
        if normal_dir and normal_dir.exists():
            for p in sorted(set(normal_dir.glob("*.BIN")) | set(normal_dir.glob("*.bin"))):
                out = ORBIT_ROOT / rpm / "normal" / (p.stem + ".npy")
                tasks.append((p, out, "normal"))

        # Faults
        synthetic_dir = RPM_FAULT_DIRS.get(rpm)
        if synthetic_dir and synthetic_dir.exists():
            for fault in FAULT_TYPES:
                fault_dir = synthetic_dir / fault
                if not fault_dir.exists():
                    continue
                for p in sorted(fault_dir.glob("*.bin")):
                    out = ORBIT_ROOT / rpm / fault / (p.stem + ".npy")
                    tasks.append((p, out, fault))

    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-compute orbit images from BIN files"
    )
    ap.add_argument(
        "--rpms", nargs="+", default=["1200rpm"],
        help="RPM conditions to process (default: both)",
    )
    ap.add_argument(
        "--axis_lim", type=float, default=DEFAULT_AXIS_LIM,
        help=f"Orbit axis half-width in mils (default {DEFAULT_AXIS_LIM})",
    )
    ap.add_argument(
        "--img_size", type=int, default=DEFAULT_IMG_SIZE,
        help=f"Orbit image pixel size (default {DEFAULT_IMG_SIZE})",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without writing any files",
    )
    args = ap.parse_args()

    tasks = collect_tasks(args.rpms)
    already_done = sum(1 for _, out, _ in tasks if out.exists())
    print(f"Tasks : {len(tasks)} files  ({already_done} already done)")

    if args.dry_run:
        print("\nFirst 10 tasks:")
        for p, out, label in tasks[:10]:
            status = "SKIP" if out.exists() else "TODO"
            print(f"  [{status}] {label:15s}  {p.name}  →  {out}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    n_ok = n_skip = n_fail = 0
    for bin_path, out_path, _ in tqdm(tasks, unit="file"):
        if out_path.exists():
            n_skip += 1
            continue
        ok = process_file(bin_path, out_path, args.axis_lim, args.img_size, dry_run=False)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone.  written={n_ok}  skipped={n_skip}  failed={n_fail}")
    print(f"Orbit images saved to: {ORBIT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
