#!/usr/bin/env python3
"""Training script for HierarchicalSignalCNN (1D time-series path).

Prerequisites
-------------
Run precompute_signals.py first to generate data/signal_cache/.

Usage
-----
venv\\Scripts\\python.exe train_signal_cnn.py
venv\\Scripts\\python.exe train_signal_cnn.py --epochs 100 --batch_size 32 --lr 5e-4
venv\\Scripts\\python.exe train_signal_cnn.py --rpms 3600rpm --no_transient

Outputs
-------
data/checkpoints/signal_cnn_best.pt   — best checkpoint (lowest val loss)
data/checkpoints/signal_train_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.signal_dataset import SignalDataset
from src.models.signal_cnn import HierarchicalSignalCNN
from src.models.orbit_cnn import HierarchicalLoss

CKPT_DIR = Path("data") / "checkpoints"


# ---------------------------------------------------------------------------
# Helpers  (identical structure to train_orbit_cnn.py)
# ---------------------------------------------------------------------------

def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    """Balance normal / fault in each mini-batch via weighted sampling."""
    binary_labels = torch.tensor([s[1] for s in samples])
    class_counts  = torch.bincount(binary_labels)
    weights       = 1.0 / class_counts.float()
    sample_weights = weights[binary_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def run_epoch(
    model: HierarchicalSignalCNN,
    loader: DataLoader,
    criterion: HierarchicalLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = binary_loss_sum = fault_loss_sum = 0.0
    binary_correct = fault_correct = fault_total = n_samples = 0

    with torch.set_grad_enabled(is_train):
        for signals, binary_labels, fault_labels in loader:
            signals       = signals.to(device)
            binary_labels = binary_labels.to(device)
            fault_labels  = fault_labels.to(device)

            binary_logits, fault_logits = model(signals)
            l_total, l_bin, l_fault = criterion(
                binary_logits, fault_logits, binary_labels, fault_labels
            )

            if is_train:
                optimizer.zero_grad()
                l_total.backward()
                optimizer.step()

            B = signals.size(0)
            n_samples       += B
            total_loss      += l_total.item() * B
            binary_loss_sum += l_bin.item() * B

            binary_correct += (binary_logits.argmax(1) == binary_labels).sum().item()

            fault_mask = binary_labels == 1
            if fault_mask.any():
                fault_loss_sum += l_fault.item() * fault_mask.sum().item()
                fault_correct  += (
                    fault_logits[fault_mask].argmax(1) == fault_labels[fault_mask]
                ).sum().item()
                fault_total += fault_mask.sum().item()

    return {
        "loss":        total_loss      / n_samples,
        "binary_loss": binary_loss_sum / n_samples,
        "fault_loss":  fault_loss_sum  / max(fault_total, 1),
        "binary_acc":  binary_correct  / n_samples,
        "fault_acc":   fault_correct   / max(fault_total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train HierarchicalSignalCNN")
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--val_split",    type=float, default=0.2)
    ap.add_argument("--lambda_fault", type=float, default=1.0,
                    help="Weight for fault-type CE loss term")
    ap.add_argument("--dropout",      type=float, default=0.4)
    ap.add_argument("--window",       type=int,   default=40_000,
                    help="Window samples for 1D input (default 40000 = 1 s @ 40 kHz)")
    ap.add_argument("--rpms",         nargs="+",  default=["3600rpm", "1200rpm"])
    ap.add_argument("--no_transient", action="store_true",
                    help="Exclude transient fault files from training")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader num_workers (0 = main process, safe on Windows)")
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

    # ---- Dataset ---------------------------------------------------------
    # Build two datasets sharing the same files but with different training flags:
    #   train_ds_base : training=True  (random window augmentation)
    #   val_ds_base   : training=False (fixed last window, reproducible)
    kw = dict(
        rpms=tuple(args.rpms),
        window_samples=args.window,
        include_transient=not args.no_transient,
    )
    train_ds_base = SignalDataset(**kw, training=True)
    val_ds_base   = SignalDataset(**kw, training=False)

    if len(train_ds_base) == 0:
        print(
            "ERROR: No BIN files found. Check data/raw/normal_* and data/synthetic/.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Deterministic index split — same indices applied to both base datasets
    n_total = len(train_ds_base)
    n_val   = int(n_total * args.val_split)
    n_train = n_total - n_val
    perm    = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_train].tolist()
    val_idx   = perm[n_train:].tolist()

    train_ds = Subset(train_ds_base, train_idx)
    val_ds   = Subset(val_ds_base,   val_idx)

    train_samples = [train_ds_base.samples[i] for i in train_idx]
    sampler = make_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    n_normal = sum(1 for _, b, _ in train_ds_base.samples if b == 0)
    n_fault  = sum(1 for _, b, _ in train_ds_base.samples if b == 1)
    print(f"Dataset: {n_total} total  "
          f"(normal={n_normal}, fault={n_fault})  "
          f"|  train={n_train}  val={n_val}")

    # ---- Model -----------------------------------------------------------
    model     = HierarchicalSignalCNN(in_channels=8, dropout=args.dropout).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = HierarchicalLoss(lambda_fault=args.lambda_fault)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    print(f"Model  : HierarchicalSignalCNN  ({n_params:,} params)")

    # ---- Training loop ---------------------------------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history: list[dict] = []

    header = (
        f"{'Epoch':>5} | {'TrainLoss':>9} | {'ValLoss':>8} | "
        f"{'BinAcc(tr)':>10} | {'BinAcc(v)':>9} | {'FltAcc(v)':>9}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0  = time.time()
        tr  = run_epoch(model, train_loader, criterion, optimizer, device)
        val = run_epoch(model, val_loader,   criterion, None,      device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "lr":    scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}":   v for k, v in val.items()},
        })

        print(
            f"{epoch:5d} | {tr['loss']:9.4f} | {val['loss']:8.4f} | "
            f"{tr['binary_acc']:10.4f} | {val['binary_acc']:9.4f} | "
            f"{val['fault_acc']:9.4f}   ({elapsed:.1f}s)"
        )

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            ckpt_path = CKPT_DIR / "signal_cnn_best.pt"
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             best_val_loss,
                    "args":                 vars(args),
                },
                ckpt_path,
            )

    # ---- Save history ----------------------------------------------------
    history_path = CKPT_DIR / "signal_train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {CKPT_DIR / 'signal_cnn_best.pt'}")
    print(f"History       : {history_path}")


if __name__ == "__main__":
    main()
