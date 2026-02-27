"""Method A: 정상 실측 데이터 군집 분석 (FFT 피처 + 차원 축소 시각화)

특징 벡터 구성 (파일당 32차원):
  - 베어링 채널 8개: (0,1), (4,5), (10,11), (16,17)
  - 채널당 4가지 피처: 1X 진폭, 2X 진폭, 3X 진폭, RMS
  - 3600rpm: 1X=60Hz / 1200rpm: 1X=20Hz

출력:
  data/analysis/normal_cluster/pca_tsne.png
  data/analysis/normal_cluster/pca_components.png  (분산 기여율)
  data/analysis/normal_cluster/features.npy         (32-dim 피처)
  data/analysis/normal_cluster/labels.npy           (RPM 레이블)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.core.rcpvms_parser import RCPVMSParser

# ── 설정 ─────────────────────────────────────────────────────────────────────
FS = 40_000                        # 샘플링 주파수 (Hz)
BEARING_CHANNELS = (0, 1, 4, 5, 10, 11, 16, 17)
CHANNEL_NAMES    = ["ch0(X)", "ch1(Y)", "ch4(X)", "ch5(Y)",
                    "ch10(X)", "ch11(Y)", "ch16(X)", "ch17(Y)"]

RPM_CONFIG = {
    "3600rpm": {"dir": Path("data/raw/normal_3600rpm"), "1x_hz": 60.0, "label": 2},
    "1200rpm": {"dir": Path("data/raw/normal_1200rpm"), "1x_hz": 20.0, "label": 1},
}
RPM_COLORS  = {1: "#2196F3", 2: "#F44336"}   # 파랑=1200rpm, 빨강=3600rpm
RPM_NAMES   = {1: "1200 rpm", 2: "3600 rpm"}

OUT_DIR = Path("data/analysis/normal_cluster")
HARM_BW = 2.0   # 고조파 탐색 대역폭 ±Hz


# ── 피처 추출 함수 ─────────────────────────────────────────────────────────────

def harmonic_amplitude(sig: np.ndarray, freq_hz: float, fs: int,
                        bw: float = HARM_BW) -> float:
    """FFT에서 freq_hz ± bw 범위의 최대 진폭(단면 스펙트럼) 반환."""
    n = len(sig)
    if n == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(sig - sig.mean())) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= freq_hz - bw) & (freqs <= freq_hz + bw)
    return float(spec[mask].max()) if mask.any() else 0.0


def extract_features(bin_path: Path, one_x_hz: float) -> np.ndarray | None:
    """BIN 파일 → 32차원 피처 벡터."""
    parser = RCPVMSParser(str(bin_path))
    try:
        parser.parse_header()
    except Exception:
        return None

    feats: list[float] = []
    for ch in BEARING_CHANNELS:
        sig = parser.read_channel(ch)
        if len(sig) < FS:        # 최소 1초 필요
            return None
        # 1X, 2X, 3X 진폭
        for k in (1, 2, 3):
            feats.append(harmonic_amplitude(sig, one_x_hz * k, FS))
        # RMS (DC 제거)
        feats.append(float(np.sqrt(np.mean((sig - sig.mean()) ** 2))))

    return np.array(feats, dtype=np.float32)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def build_feature_matrix() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """전체 정상 파일로부터 피처 행렬 X (N×32) 와 레이블 y (N,) 구성."""
    X_rows, y_rows, paths = [], [], []
    for rpm_key, cfg in RPM_CONFIG.items():
        d = cfg["dir"]
        if not d.exists():
            print(f"[WARN] 경로 없음: {d}")
            continue
        bins = sorted(set(d.glob("*.BIN")) | set(d.glob("*.bin")))
        for p in tqdm(bins, desc=f"{rpm_key} ({len(bins)}개)", unit="file"):
            vec = extract_features(p, cfg["1x_hz"])
            if vec is None:
                print(f"[SKIP] {p.name}")
                continue
            X_rows.append(vec)
            y_rows.append(cfg["label"])
            paths.append(p.name)

    return np.stack(X_rows), np.array(y_rows, dtype=np.int32), paths


def plot_scatter(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray,
                 title: str) -> None:
    for lbl, color in RPM_COLORS.items():
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=RPM_NAMES[lbl],
                   s=40, alpha=0.75, edgecolors="none")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def main(tsne_perplexity: int = 30, random_state: int = 42) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 피처 추출
    print("=== [1/4] FFT 피처 추출 ===")
    X, y, paths = build_feature_matrix()
    print(f"  → X shape: {X.shape}, labels: { {int(k): int((y==k).sum()) for k in np.unique(y)} }")

    np.save(OUT_DIR / "features.npy", X)
    np.save(OUT_DIR / "labels.npy",   y)

    # 2. 표준화
    print("=== [2/4] 표준화 (StandardScaler) ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA
    print("=== [3/4] PCA ===")
    pca = PCA(random_state=random_state)
    X_pca_full = pca.fit_transform(X_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(explained, 0.90)) + 1
    print(f"  → 90% 분산 설명에 필요한 PC 수: {n_90}")

    X_pca2 = X_pca_full[:, :2]

    # 4. t-SNE (PCA 10차원 → t-SNE)
    print("=== [4/4] t-SNE (perplexity={}) ===".format(tsne_perplexity))
    n_pca_pre = min(10, X_pca_full.shape[1])
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity,
                random_state=random_state, max_iter=1000,
                init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_pca_full[:, :n_pca_pre])

    # ── 시각화 A: PCA 2D + t-SNE 2D 나란히 ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Normal Data Cluster Analysis (Method A: FFT Features)",
                 fontsize=13, fontweight="bold")

    plot_scatter(axes[0], X_pca2,  y, f"PCA (PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})")
    plot_scatter(axes[1], X_tsne,  y, f"t-SNE (perplexity={tsne_perplexity})")

    plt.tight_layout()
    out_path = OUT_DIR / "pca_tsne.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {out_path}")

    # ── 시각화 B: PCA 분산 기여율 ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    n_show = min(32, len(explained))
    ax2.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show],
            alpha=0.7, color="#5C6BC0", label="개별 분산")
    ax2.plot(range(1, n_show + 1), explained[:n_show],
             color="#E53935", marker="o", markersize=3, label="누적 분산")
    ax2.axvline(n_90, color="gray", linestyle="--", linewidth=1)
    ax2.axhline(0.90, color="gray", linestyle="--", linewidth=1)
    ax2.text(n_90 + 0.3, 0.05, f"n={n_90}\n(90%)", fontsize=8, color="gray")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title("PCA Explained Variance")
    ax2.legend()
    plt.tight_layout()
    out_path2 = OUT_DIR / "pca_components.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"  → 저장: {out_path2}")

    # ── 요약 출력 ──────────────────────────────────────────────────────────
    print("\n===== 요약 =====")
    print(f"  파일 수: {len(y)}  (1200rpm={int((y==1).sum())}, 3600rpm={int((y==2).sum())})")
    print(f"  피처 차원: {X.shape[1]}")
    print(f"  PC1 분산: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2 분산: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  90% 분산 PC 수: {n_90}")
    print(f"  결과 저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Normal data cluster analysis (Method A)")
    ap.add_argument("--perplexity", type=int, default=30,
                    help="t-SNE perplexity (default: 30)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(tsne_perplexity=args.perplexity, random_state=args.seed)
