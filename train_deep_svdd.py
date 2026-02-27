#!/usr/bin/env python3
"""Two-stage Deep SVDD training using real normal data only.

Stage 1 — Autoencoder pre-training (MSE reconstruction loss)
Stage 2 — Deep SVDD fine-tuning   (hypersphere compression loss)

No synthetic or fault data is used in any phase.

Usage
-----
venv\\Scripts\\python.exe train_deep_svdd.py
venv\\Scripts\\python.exe train_deep_svdd.py --ae_epochs 30 --svdd_epochs 50
venv\\Scripts\\python.exe train_deep_svdd.py --skip_ae   # Stage 2 only (load existing AE)

Outputs
-------
data/checkpoints/autoencoder_best.pt   — best AE checkpoint (lowest val MSE)
data/checkpoints/ae_train_history.json
data/checkpoints/svdd_best.pt          — best SVDD checkpoint (lowest val dist)
data/checkpoints/svdd_train_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.normal_dataset import NormalDataset
from src.models.deep_svdd import DeepSVDD, SignalAutoencoder

CKPT_DIR = Path("data") / "checkpoints"


# ── Stage 1: Autoencoder ─────────────────────────────────────────────────────

def train_autoencoder(
    model: SignalAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_ae, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.ae_epochs, eta_min=args.lr_ae * 0.01
    )
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    no_improve   = 0
    history: list[dict] = []

    print("\n" + "=" * 55)
    print("  Stage 1 — Autoencoder Pre-training")
    print("=" * 55)
    header = f"{'Epoch':>5} | {'TrainMSE':>10} | {'ValMSE':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.ae_epochs + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        tr_loss, n_tr = 0.0, 0
        for signals, _ in train_loader:
            signals = signals.to(device)
            recon, _ = model(signals)
            loss = criterion(recon, signals)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * signals.size(0)
            n_tr    += signals.size(0)
        scheduler.step()
        tr_loss /= n_tr

        # ── val ──
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for signals, _ in val_loader:
                signals  = signals.to(device)
                recon, _ = model(signals)
                val_loss += criterion(recon, signals).item() * signals.size(0)
                n_val    += signals.size(0)
        val_loss /= n_val

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": val_loss})
        print(f"{epoch:5d} | {tr_loss:10.6f} | {val_loss:10.6f}   ({elapsed:.1f}s)")

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            no_improve   = 0
            torch.save(model.state_dict(), CKPT_DIR / "autoencoder_best.pt")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    with open(CKPT_DIR / "ae_train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Restore best weights into the model in-place so the caller
    # does not need a separate reload step (I5)
    model.load_state_dict(
        torch.load(CKPT_DIR / "autoencoder_best.pt", map_location="cpu", weights_only=True)
    )

    print(f"\nBest AE val MSE : {best_val_mse:.6f}")
    print(f"Checkpoint      : {CKPT_DIR / 'autoencoder_best.pt'}")


# ── Stage 2: Deep SVDD ───────────────────────────────────────────────────────

def train_svdd(
    model: DeepSVDD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    # Initialise center using a fixed-window (deterministic) loader so
    # the center is not biased by stochastic window sampling.
    from src.datasets.normal_dataset import _SubsetNormalDataset
    ds = train_loader.dataset
    if not isinstance(ds, _SubsetNormalDataset):
        raise RuntimeError(
            f"train_loader.dataset must be _SubsetNormalDataset "
            f"(produced by NormalDataset.split_by_file()), "
            f"got {type(ds).__name__}. Do not bypass split_by_file()."
        )
    center_loader = DataLoader(
        _SubsetNormalDataset(ds.base, ds.indices, training=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    print("\nInitialising hypersphere center c ...")
    model.initialize_center(center_loader, device)
    c_norm = model.center.norm().item()
    print(f"  ||c|| = {c_norm:.4f}   dim = {model.center.shape[0]}")
    if c_norm < 0.01:
        print("  WARNING: ||c|| is very small — collapse risk. "
              "Consider re-running Stage 1 with more epochs.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_svdd, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.svdd_epochs, eta_min=args.lr_svdd * 0.01
    )

    best_val_dist = float("inf")
    no_improve    = 0
    history: list[dict] = []

    print("\n" + "=" * 65)
    print("  Stage 2 — Deep SVDD Fine-tuning")
    print("=" * 65)
    header = (
        f"{'Epoch':>5} | {'TR dist':>8} | {'VL dist':>8} | "
        f"{'VL std':>7} | {'R(p99)':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.svdd_epochs + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        sq_dists: list[torch.Tensor] = []
        for signals, _ in train_loader:
            d2   = model(signals.to(device))      # squared distances
            loss = d2.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sq_dists.append(d2.detach().cpu())
        scheduler.step()
        tr_dists = torch.cat(sq_dists).sqrt()
        tr_mean  = tr_dists.mean().item()

        # ── val ──
        model.eval()
        val_dists_list: list[torch.Tensor] = []
        with torch.no_grad():
            for signals, _ in val_loader:
                d2 = model(signals.to(device))
                val_dists_list.append(d2.cpu().sqrt())
        vd      = torch.cat(val_dists_list)
        vl_mean = vd.mean().item()
        # correction=0 (population std) avoids NaN when batch has 1 sample (R2)
        vl_std  = vd.std(correction=0).item() if len(vd) > 1 else 0.0
        vl_r99  = float(vd.quantile(0.99))

        elapsed = time.time() - t0
        history.append({
            "epoch":   epoch,
            "tr_dist": tr_mean,
            "vl_dist": vl_mean,
            "vl_std":  vl_std,
            "vl_r99":  vl_r99,
        })
        print(
            f"{epoch:5d} | {tr_mean:8.4f} | {vl_mean:8.4f} | "
            f"{vl_std:7.4f} | {vl_r99:7.4f}   ({elapsed:.1f}s)"
        )

        if vl_mean < best_val_dist:
            best_val_dist = vl_mean
            no_improve    = 0
            torch.save(
                {
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "center":        model.center.cpu(),
                    "val_dist_mean": torch.tensor(vl_mean),
                    "val_dist_std":  torch.tensor(vl_std),
                    "val_r99":       torch.tensor(vl_r99),
                },
                CKPT_DIR / "svdd_best.pt",
            )
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    with open(CKPT_DIR / "svdd_train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Report suggested thresholds (M6: weights_only=True safe — all values are tensors)
    best = torch.load(
        CKPT_DIR / "svdd_best.pt", map_location="cpu", weights_only=True
    )
    mu    = float(best["val_dist_mean"])
    sigma = float(best["val_dist_std"])
    r99   = float(best["val_r99"])
    print(f"\nBest val dist   : {mu:.4f} ± {sigma:.4f}")
    print(f"Sphere R (p99)  : {r99:.4f}")
    # C4 fix: tighter sphere (smaller θ) → more anomalies caught → higher recall, higher FPR
    print(f"Threshold θ(k=2): {mu + 2 * sigma:.4f}  (tighter sphere → higher recall, higher FPR)")
    print(f"Threshold θ(k=3): {mu + 3 * sigma:.4f}  (looser sphere  → lower recall,  lower FPR)")
    print(f"Checkpoint      : {CKPT_DIR / 'svdd_best.pt'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-stage Deep SVDD training (real normal data only)"
    )
    ap.add_argument("--ae_epochs",   type=int,   default=50,
                    help="Stage 1 autoencoder epochs")
    ap.add_argument("--svdd_epochs", type=int,   default=100,
                    help="Stage 2 SVDD fine-tuning epochs")
    ap.add_argument("--lr_ae",       type=float, default=1e-3,
                    help="Stage 1 learning rate")
    ap.add_argument("--lr_svdd",     type=float, default=1e-4,
                    help="Stage 2 learning rate")
    ap.add_argument("--latent_dim",  type=int,   default=128,
                    help="Encoder latent dimension")
    ap.add_argument("--svdd_dim",    type=int,   default=64,
                    help="Hypersphere embedding dimension (proj head output)")
    ap.add_argument("--batch_size",  type=int,   default=8,
                    help="Mini-batch size (small due to limited normal data)")
    ap.add_argument("--val_ratio",   type=float, default=0.2,
                    help="Fraction of files held out for validation")
    ap.add_argument("--patience",    type=int,   default=10,
                    help="Early stopping patience (0 = off)")
    ap.add_argument("--window",      type=int,   default=40_000,
                    help="Window length in samples")
    ap.add_argument("--rpms",        nargs="+",
                    default=["unknown", "1200rpm", "3600rpm"],
                    help="RPM keys to include (subsets of NORMAL_DIRS)")
    ap.add_argument("--device",      default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--workers",     type=int,   default=0,
                    help="DataLoader num_workers (0 = safe on Windows)")
    ap.add_argument("--skip_ae",     action="store_true",
                    help="Skip Stage 1 and load existing autoencoder_best.pt")
    args = ap.parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # ── Dataset ──
    full_ds = NormalDataset(
        rpms=tuple(args.rpms),
        window_samples=args.window,
    )
    if len(full_ds) == 0:
        print("ERROR: No normal BIN files found.", file=sys.stderr)
        sys.exit(1)

    train_ds, val_ds = full_ds.split_by_file(val_ratio=args.val_ratio)
    print(full_ds.summary())
    print(f"Split  : train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Autoencoder ──
    ae_model = SignalAutoencoder(
        in_channels=8,
        latent_dim=args.latent_dim,
        window_samples=args.window,
    ).to(device)
    n_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print(f"Autoencoder    : {n_ae:,} params")

    if args.skip_ae:
        ae_ckpt = CKPT_DIR / "autoencoder_best.pt"
        if not ae_ckpt.exists():
            print(f"ERROR: {ae_ckpt} not found. Run without --skip_ae first.",
                  file=sys.stderr)
            sys.exit(1)
        ae_model.load_state_dict(
            torch.load(ae_ckpt, map_location=device, weights_only=True)
        )
        print(f"Loaded existing autoencoder from {ae_ckpt}")
    else:
        # train_autoencoder restores best weights into ae_model before returning (I5)
        train_autoencoder(ae_model, train_loader, val_loader, args, device)

    # ── Stage 2: Deep SVDD ──
    svdd = DeepSVDD(
        in_channels=8,
        latent_dim=args.latent_dim,
        svdd_dim=args.svdd_dim,
    ).to(device)
    svdd.load_encoder_from_autoencoder(ae_model.state_dict())

    n_svdd = sum(p.numel() for p in svdd.parameters() if p.requires_grad)
    print(f"Deep SVDD      : {n_svdd:,} params  (encoder + proj head)")

    train_svdd(svdd, train_loader, val_loader, args, device)


if __name__ == "__main__":
    main()
