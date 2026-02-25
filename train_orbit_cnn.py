#!/usr/bin/env python3
"""Training script for HierarchicalOrbitCNN.

Prerequisites
-------------
Run precompute_orbits.py first to generate data/orbit_images/.

Usage
-----
venv\\Scripts\\python.exe train_orbit_cnn.py
venv\\Scripts\\python.exe train_orbit_cnn.py --epochs 100 --batch_size 32 --lr 5e-4
venv\\Scripts\\python.exe train_orbit_cnn.py --rpms 3600rpm --no_transient

Outputs
-------
data/checkpoints/orbit_cnn_best.pt   — best checkpoint (lowest val loss)
data/checkpoints/train_history.json  — per-epoch metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.orbit_dataset import OrbitDataset
from src.models.orbit_cnn import HierarchicalLoss, HierarchicalOrbitCNN

ORBIT_ROOT = Path("data") / "orbit_images"
CKPT_DIR   = Path("data") / "checkpoints"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    """Balance normal / fault in each mini-batch via weighted sampling."""
    binary_labels = torch.tensor([s[1] for s in samples])
    class_counts  = torch.bincount(binary_labels)
    weights       = 1.0 / class_counts.float()
    sample_weights = weights[binary_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def run_epoch(
    model: HierarchicalOrbitCNN,
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
        for images, binary_labels, fault_labels in loader:
            images        = images.to(device)
            binary_labels = binary_labels.to(device)
            fault_labels  = fault_labels.to(device)

            binary_logits, fault_logits = model(images)
            l_total, l_bin, l_fault = criterion(
                binary_logits, fault_logits, binary_labels, fault_labels
            )

            if is_train:
                optimizer.zero_grad()
                l_total.backward()
                optimizer.step()

            B = images.size(0)
            n_samples     += B
            total_loss    += l_total.item() * B
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
        "loss":       total_loss    / n_samples,
        "binary_loss": binary_loss_sum / n_samples,
        "fault_loss":  fault_loss_sum  / max(fault_total, 1),
        "binary_acc":  binary_correct  / n_samples,
        "fault_acc":   fault_correct   / max(fault_total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train HierarchicalOrbitCNN")
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--val_split",    type=float, default=0.2)
    ap.add_argument("--lambda_fault", type=float, default=1.0,
                    help="Weight for fault-type CE loss term")
    ap.add_argument("--dropout",      type=float, default=0.4)
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
    full_ds = OrbitDataset(
        orbit_root=ORBIT_ROOT,
        rpms=tuple(args.rpms),
        include_transient=not args.no_transient,
    )
    if len(full_ds) == 0:
        print(
            "ERROR: No orbit images found. Run precompute_orbits.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_val   = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Build a lightweight proxy to pass samples list into make_weighted_sampler
    train_samples = [full_ds.samples[i] for i in train_ds.indices]
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

    print(f"Dataset: {len(full_ds)} total  |  train={n_train}  val={n_val}")

    # ---- Model -----------------------------------------------------------
    model     = HierarchicalOrbitCNN(in_channels=4, dropout=args.dropout).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = HierarchicalLoss(lambda_fault=args.lambda_fault)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    print(f"Model  : HierarchicalOrbitCNN  ({n_params:,} params)")

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

        # Save best checkpoint
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            ckpt_path = CKPT_DIR / "orbit_cnn_best.pt"
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            best_val_loss,
                    "args":                vars(args),
                },
                ckpt_path,
            )

    # ---- Save history ----------------------------------------------------
    history_path = CKPT_DIR / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {CKPT_DIR / 'orbit_cnn_best.pt'}")
    print(f"History       : {history_path}")


if __name__ == "__main__":
    main()
