#!/usr/bin/env python3
"""Method B 앙상블 파인튜닝: log_weights (2개) 만 학습한다.

두 백본은 완전히 frozen 상태이며, 학습 가능한 파라미터는
softmax 가중치 log_weights[0], log_weights[1] 뿐이다.

Prerequisites
-------------
1. data/checkpoints/signal_cnn_best.pt
2. data/checkpoints/orbit_cnn_best.pt

Usage
-----
venv\\Scripts\\python.exe train_ensemble.py
venv\\Scripts\\python.exe train_ensemble.py --epochs 50 --lr 1e-2

Outputs
-------
data/checkpoints/ensemble_best.pt  — learned log_weights + metadata
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.signal_dataset import SignalDataset
from src.datasets.orbit_dataset import OrbitDataset
from src.models.ensemble import EnsembleCNN
from src.models.orbit_cnn import HierarchicalLoss

CKPT_DIR = Path("data") / "checkpoints"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: EnsembleCNN,
    sig_loader: DataLoader,
    orb_loader: DataLoader,
    criterion: HierarchicalLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    # 백본은 항상 eval 모드 (frozen)
    model.signal_cnn.eval()
    model.orbit_cnn.eval()
    if is_train:
        model.log_weights.requires_grad_(True)

    total_loss = 0.0
    binary_correct = fault_correct = fault_total = n_total = 0

    with torch.set_grad_enabled(is_train):
        for sig_batch, orb_batch in zip(sig_loader, orb_loader):
            sig_in        = sig_batch[0].to(device)
            binary_labels = sig_batch[1].to(device)
            fault_labels  = sig_batch[2].to(device)
            orb_in        = orb_batch[0].to(device)

            bin_logits, flt_logits = model(sig_in, orb_in)
            l_total, _, _ = criterion(bin_logits, flt_logits, binary_labels, fault_labels)

            if is_train:
                optimizer.zero_grad()
                l_total.backward()
                optimizer.step()

            B = sig_in.size(0)
            n_total        += B
            total_loss     += l_total.item() * B
            binary_correct += (bin_logits.argmax(1) == binary_labels).sum().item()

            fault_mask = binary_labels == 1
            if fault_mask.any():
                fp = flt_logits[fault_mask].argmax(1)
                fl = fault_labels[fault_mask]
                fault_correct += (fp == fl).sum().item()
                fault_total   += fault_mask.sum().item()

    return {
        "loss":       total_loss     / n_total,
        "binary_acc": binary_correct / n_total,
        "fault_acc":  fault_correct  / max(fault_total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train EnsembleCNN Method B (log_weights only)")
    ap.add_argument("--signal_ckpt",  default=str(CKPT_DIR / "signal_cnn_best.pt"))
    ap.add_argument("--orbit_ckpt",   default=str(CKPT_DIR / "orbit_cnn_best.pt"))
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=1e-2)
    ap.add_argument("--val_split",    type=float, default=0.2)
    ap.add_argument("--patience",     type=int,   default=10)
    ap.add_argument("--rpms",         nargs="+",  default=["3600rpm", "1200rpm"])
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

    # ---- Checkpoints -----------------------------------------------------
    for ckpt, name in [(args.signal_ckpt, "signal"), (args.orbit_ckpt, "orbit")]:
        if not Path(ckpt).exists():
            print(f"ERROR: {name} checkpoint not found: {ckpt}", file=sys.stderr)
            sys.exit(1)

    # ---- Datasets --------------------------------------------------------
    sig_ds = SignalDataset(
        rpms=tuple(args.rpms),
        include_transient=not args.no_transient,
        training=False,   # 고정 윈도우 — orb_ds와 대응 보장
    )
    orb_ds = OrbitDataset(
        rpms=tuple(args.rpms),
        include_transient=not args.no_transient,
        cache=True,
    )

    for name, ds in [("Signal", sig_ds), ("Orbit", orb_ds)]:
        if len(ds) == 0:
            print(f"ERROR: {name} dataset is empty.", file=sys.stderr)
            sys.exit(1)

    # 두 데이터셋 크기가 다르면 동일 seed 분할로도 인덱스 대응이 깨짐 → 사전 검증
    if len(sig_ds) != len(orb_ds):
        print(
            f"ERROR: Signal ({len(sig_ds)}) and Orbit ({len(orb_ds)}) dataset sizes differ. "
            "Index correspondence cannot be guaranteed — check that both datasets "
            "scan the same BIN files.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 동일 seed + 동일 크기 분할 → 인덱스 대응 보장
    gen = torch.Generator().manual_seed(42)
    n_val_sig = int(len(sig_ds) * args.val_split)
    sig_train, sig_val = random_split(
        sig_ds, [len(sig_ds) - n_val_sig, n_val_sig], generator=gen
    )

    gen = torch.Generator().manual_seed(42)
    n_val_orb = int(len(orb_ds) * args.val_split)
    orb_train, orb_val = random_split(
        orb_ds, [len(orb_ds) - n_val_orb, n_val_orb], generator=gen
    )

    # 분할 결과 크기 일치 재확인 (방어적 검증)
    assert len(sig_train) == len(orb_train) and len(sig_val) == len(orb_val), (
        f"Split size mismatch after random_split: "
        f"sig=({len(sig_train)},{len(sig_val)}) orb=({len(orb_train)},{len(orb_val)})"
    )

    # shuffle=False: zip 대응을 위해 순서 고정
    sig_train_loader = DataLoader(sig_train, batch_size=args.batch_size, shuffle=False)
    orb_train_loader = DataLoader(orb_train, batch_size=args.batch_size, shuffle=False)
    sig_val_loader   = DataLoader(sig_val,   batch_size=args.batch_size, shuffle=False)
    orb_val_loader   = DataLoader(orb_val,   batch_size=args.batch_size, shuffle=False)

    print(f"Dataset: {len(sig_ds)} total  |  train={len(sig_train)}  val={len(sig_val)}")

    # ---- Model -----------------------------------------------------------
    model = EnsembleCNN(
        signal_ckpt=args.signal_ckpt,
        orbit_ckpt=args.orbit_ckpt,
        method="B",
        device=device,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params : {n_trainable}  (log_weights only)")

    criterion = HierarchicalLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # ---- Training loop ---------------------------------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve    = 0

    header = (
        f"{'Epoch':>5} | {'TrainLoss':>9} | {'ValLoss':>8} | "
        f"{'BinAcc(tr)':>10} | {'BinAcc(v)':>9} | {'FltAcc(v)':>9} | "
        f"{'w_1d':>6} | {'w_2d':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0  = time.time()
        tr  = run_epoch(model, sig_train_loader, orb_train_loader, criterion, optimizer, device)
        val = run_epoch(model, sig_val_loader,   orb_val_loader,   criterion, None,      device)
        elapsed = time.time() - t0

        w1, w2 = model._get_weights()
        print(
            f"{epoch:5d} | {tr['loss']:9.4f} | {val['loss']:8.4f} | "
            f"{tr['binary_acc']:10.4f} | {val['binary_acc']:9.4f} | "
            f"{val['fault_acc']:9.4f} | {w1:.4f} | {w2:.4f}   ({elapsed:.1f}s)"
        )

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            no_improve    = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "log_weights": model.log_weights.data.clone(),
                    "val_loss":    best_val_loss,
                    "w_1d":        w1,
                    "w_2d":        w2,
                },
                CKPT_DIR / "ensemble_best.pt",
            )
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ---- Summary ---------------------------------------------------------
    best = torch.load(
        CKPT_DIR / "ensemble_best.pt", map_location=device, weights_only=True
    )
    print(f"\nBest epoch      : {best['epoch']}  val_loss={best['val_loss']:.4f}")
    print(f"Learned weights : w_1d={best['w_1d']:.4f}  w_2d={best['w_2d']:.4f}")
    print(f"Checkpoint      : {CKPT_DIR / 'ensemble_best.pt'}")
    print(f"\n→ 평가: venv\\Scripts\\python.exe evaluate_ensemble.py "
          f"--ensemble_ckpt data/checkpoints/ensemble_best.pt")


if __name__ == "__main__":
    main()
