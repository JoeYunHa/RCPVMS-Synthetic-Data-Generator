#!/usr/bin/env python3
"""앙상블 추론 평가 스크립트 (Method A — 고정 평균).

두 개의 독립 체크포인트(1D Signal CNN + 2D Orbit CNN)를 로드하고
동일한 파일에 대한 양쪽 예측을 평균 logit으로 결합하여 최종 성능을 보고한다.

Prerequisites
-------------
1. precompute_signals.py  →  data/signal_cache/
2. precompute_orbits.py   →  data/orbit_images/
3. train_signal_cnn.py    →  data/checkpoints/signal_cnn_best.pt
4. train_orbit_cnn.py     →  data/checkpoints/orbit_cnn_best.pt

Usage
-----
venv\\Scripts\\python.exe evaluate_ensemble.py
venv\\Scripts\\python.exe evaluate_ensemble.py --alpha 0.6
venv\\Scripts\\python.exe evaluate_ensemble.py --no_transient

출력 예시
---------
  [Signal CNN 단독]  binary=0.9200  fault=0.8500
  [Orbit  CNN 단독]  binary=0.9100  fault=0.8300
  [앙상블 (α=0.50)]  binary=0.9400  fault=0.8800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.signal_dataset import SignalDataset
from src.datasets.orbit_dataset import OrbitDataset
from src.models.signal_cnn import HierarchicalSignalCNN
from src.models.orbit_cnn import HierarchicalOrbitCNN, HierarchicalLoss
from src.models.ensemble import EnsembleCNN

ORBIT_ROOT = Path("data") / "orbit_images"
CKPT_DIR    = Path("data") / "checkpoints"

FAULT_NAMES = ["unbalance", "misalignment", "oil_whip"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_single(
    model: HierarchicalSignalCNN | HierarchicalOrbitCNN,
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> dict:
    """단일 모델 평가 — binary_acc, fault_acc, confusion matrix."""
    model.eval()
    binary_correct = fault_correct = fault_total = n_total = 0
    # fault confusion: [pred_row × true_col], 3×3
    fault_cm = np.zeros((3, 3), dtype=int)

    with torch.no_grad():
        for batch in loader:
            inputs        = batch[0].to(device)
            binary_labels = batch[1].to(device)
            fault_labels  = batch[2].to(device)

            bin_logits, flt_logits = model(inputs)

            B = inputs.size(0)
            n_total        += B
            binary_pred     = bin_logits.argmax(1)
            binary_correct += (binary_pred == binary_labels).sum().item()

            fault_mask = binary_labels == 1
            if fault_mask.any():
                fp = flt_logits[fault_mask].argmax(1)
                fl = fault_labels[fault_mask]
                fault_correct += (fp == fl).sum().item()
                fault_total   += fault_mask.sum().item()
                for p, t in zip(fp.cpu().numpy(), fl.cpu().numpy()):
                    fault_cm[p, t] += 1

    return {
        "label":        label,
        "binary_acc":   binary_correct / n_total,
        "fault_acc":    fault_correct  / max(fault_total, 1),
        "n_total":      n_total,
        "fault_total":  fault_total,
        "fault_cm":     fault_cm,
    }


def evaluate_ensemble_loader(
    signal_model: HierarchicalSignalCNN,
    orbit_model: HierarchicalOrbitCNN,
    sig_loader: DataLoader,
    orb_loader: DataLoader,
    alpha: float,
    device: torch.device,
) -> dict:
    """파일 단위로 두 모델의 logit을 결합하여 앙상블 정확도를 계산한다.

    Note: sig_loader와 orb_loader는 동일한 파일 인덱스를 사용해야 한다.
    (동일한 random_split seed 사용)
    """
    signal_model.eval()
    orbit_model.eval()

    binary_correct = fault_correct = fault_total = n_total = 0
    fault_cm = np.zeros((3, 3), dtype=int)

    with torch.no_grad():
        for sig_batch, orb_batch in zip(sig_loader, orb_loader):
            sig_in, binary_labels, fault_labels = (
                sig_batch[0].to(device), sig_batch[1].to(device), sig_batch[2].to(device)
            )
            orb_in = orb_batch[0].to(device)

            bin_1d, flt_1d = signal_model(sig_in)
            bin_2d, flt_2d = orbit_model(orb_in)

            bin_logits = alpha * bin_1d + (1.0 - alpha) * bin_2d
            flt_logits = alpha * flt_1d + (1.0 - alpha) * flt_2d

            B = sig_in.size(0)
            n_total        += B
            binary_pred     = bin_logits.argmax(1)
            binary_correct += (binary_pred == binary_labels).sum().item()

            fault_mask = binary_labels == 1
            if fault_mask.any():
                fp = flt_logits[fault_mask].argmax(1)
                fl = fault_labels[fault_mask]
                fault_correct += (fp == fl).sum().item()
                fault_total   += fault_mask.sum().item()
                for p, t in zip(fp.cpu().numpy(), fl.cpu().numpy()):
                    fault_cm[p, t] += 1

    return {
        "label":       f"앙상블 (α={alpha:.2f})",
        "binary_acc":  binary_correct / n_total,
        "fault_acc":   fault_correct  / max(fault_total, 1),
        "n_total":     n_total,
        "fault_total": fault_total,
        "fault_cm":    fault_cm,
    }


def print_result(r: dict) -> None:
    print(
        f"  [{r['label']:20s}]  "
        f"binary={r['binary_acc']:.4f}  "
        f"fault={r['fault_acc']:.4f}  "
        f"(n={r['n_total']}, fault_n={r['fault_total']})"
    )
    if r["fault_cm"].sum() > 0:
        print("    Fault confusion (pred↓ / true→):")
        header = f"    {'':12s}" + "".join(f"{n:>14s}" for n in FAULT_NAMES)
        print(header)
        for i, row_name in enumerate(FAULT_NAMES):
            row_str = "".join(f"{r['fault_cm'][i, j]:>14d}" for j in range(3))
            print(f"    {row_name:12s}{row_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="앙상블 추론 평가")
    ap.add_argument("--signal_ckpt", default=str(CKPT_DIR / "signal_cnn_best.pt"))
    ap.add_argument("--orbit_ckpt",  default=str(CKPT_DIR / "orbit_cnn_best.pt"))
    ap.add_argument("--alpha",       type=float, default=0.5,
                    help="1D 모델 가중치 (0~1, 기본 0.5). --ensemble_ckpt 사용 시 무시됨")
    ap.add_argument("--ensemble_ckpt", default=None,
                    help="train_ensemble.py 출력 체크포인트 (Method B 학습 가중치)")
    ap.add_argument("--val_split",   type=float, default=0.2)
    ap.add_argument("--batch_size",  type=int,   default=16)
    ap.add_argument("--rpms",        nargs="+",  default=["1200rpm"])
    ap.add_argument("--no_transient", action="store_true")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    # ---- Device ----------------------------------------------------------
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # ---- Check checkpoints -----------------------------------------------
    for ckpt, name in [(args.signal_ckpt, "signal"), (args.orbit_ckpt, "orbit")]:
        if not Path(ckpt).exists():
            print(f"ERROR: {name} checkpoint not found: {ckpt}", file=sys.stderr)
            sys.exit(1)

    # ---- Datasets --------------------------------------------------------
    sig_ds = SignalDataset(
        rpms=tuple(args.rpms),
        include_transient=not args.no_transient, training=False,
    )
    orb_ds = OrbitDataset(
        orbit_root=ORBIT_ROOT, rpms=tuple(args.rpms),
        include_transient=not args.no_transient,
    )

    for name, ds in [("Signal", sig_ds), ("Orbit", orb_ds)]:
        if len(ds) == 0:
            print(f"ERROR: {name} dataset is empty.", file=sys.stderr)
            sys.exit(1)

    # ── 동일 seed로 분할 → 파일 순서 대응 보장 ─────────────────────────────
    gen = torch.Generator().manual_seed(42)
    n_val_sig = int(len(sig_ds) * args.val_split)
    _, sig_val = random_split(sig_ds, [len(sig_ds) - n_val_sig, n_val_sig], generator=gen)

    gen = torch.Generator().manual_seed(42)
    n_val_orb = int(len(orb_ds) * args.val_split)
    _, orb_val = random_split(orb_ds, [len(orb_ds) - n_val_orb, n_val_orb], generator=gen)

    sig_loader = DataLoader(sig_val, batch_size=args.batch_size, shuffle=False)
    orb_loader = DataLoader(orb_val, batch_size=args.batch_size, shuffle=False)

    # ---- Models ----------------------------------------------------------
    signal_model = HierarchicalSignalCNN(in_channels=8).to(device)
    orbit_model  = HierarchicalOrbitCNN(in_channels=4).to(device)

    sig_ckpt = torch.load(args.signal_ckpt, map_location=device, weights_only=True)
    orb_ckpt = torch.load(args.orbit_ckpt,  map_location=device, weights_only=True)
    signal_model.load_state_dict(sig_ckpt["model_state_dict"])
    orbit_model.load_state_dict(orb_ckpt["model_state_dict"])

    print(f"\n  Signal checkpoint: epoch={sig_ckpt['epoch']}  val_loss={sig_ckpt['val_loss']:.4f}")
    print(f"  Orbit  checkpoint: epoch={orb_ckpt['epoch']}  val_loss={orb_ckpt['val_loss']:.4f}")

    # ---- Resolve alpha ---------------------------------------------------
    alpha = args.alpha
    if args.ensemble_ckpt is not None:
        ens_ckpt_path = Path(args.ensemble_ckpt)
        if not ens_ckpt_path.exists():
            print(f"ERROR: ensemble checkpoint not found: {ens_ckpt_path}",
                  file=sys.stderr)
            sys.exit(1)
        ens_ckpt = torch.load(ens_ckpt_path, map_location=device, weights_only=True)
        alpha = ens_ckpt["w_1d"]
        print(f"  Ensemble checkpoint: epoch={ens_ckpt['epoch']}  "
              f"w_1d={ens_ckpt['w_1d']:.4f}  w_2d={ens_ckpt['w_2d']:.4f}")

    # ---- Evaluate --------------------------------------------------------
    print("\n" + "=" * 64)
    print("  평가 결과")
    print("=" * 64)

    r_sig = evaluate_single(signal_model, sig_loader, device, "Signal CNN (1D)")
    r_orb = evaluate_single(orbit_model,  orb_loader, device, "Orbit  CNN (2D)")
    r_ens = evaluate_ensemble_loader(
        signal_model, orbit_model,
        sig_loader, orb_loader,
        alpha=alpha, device=device,
    )

    print_result(r_sig)
    print_result(r_orb)
    print()
    print_result(r_ens)
    print("=" * 64)


if __name__ == "__main__":
    main()
