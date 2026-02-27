"""Method B: GMM BIC/AIC 기반 군집 수 정량 분석

Method A에서 저장된 피처(features.npy, labels.npy)를 재사용한다.

분석 대상:
  (1) 전체 (1200rpm + 3600rpm 합산)
  (2) 1200rpm 단독
  (3) 3600rpm 단독

출력:
  data/analysis/normal_cluster/gmm_bic_aic.png      (BIC/AIC 곡선)
  data/analysis/normal_cluster/gmm_ellipses.png     (PCA 2D + GMM 타원)
  data/analysis/normal_cluster/gmm_summary.txt      (최적 군집 수 요약)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ── 설정 ─────────────────────────────────────────────────────────────────────
FEAT_PATH  = Path("data/analysis/normal_cluster/features.npy")
LABEL_PATH = Path("data/analysis/normal_cluster/labels.npy")
OUT_DIR    = Path("data/analysis/normal_cluster")

MAX_COMPONENTS = 6
N_INIT         = 10      # GMM 초기화 반복 수 (안정적인 수렴)
RANDOM_STATE   = 42

RPM_COLORS = {1: "#2196F3", 2: "#F44336"}
RPM_NAMES  = {1: "1200 rpm", 2: "3600 rpm"}

SUBSETS = {
    "전체":     None,    # 필터 없음
    "1200rpm": 1,
    "3600rpm": 2,
}


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def fit_gmm_sweep(X_scaled: np.ndarray, max_k: int) -> tuple[list, list, list]:
    """k=1..max_k에 대해 GMM 학습 후 BIC/AIC 반환."""
    ks, bics, aics = [], [], []
    for k in range(1, max_k + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=N_INIT, random_state=RANDOM_STATE)
        gm.fit(X_scaled)
        ks.append(k)
        bics.append(gm.bic(X_scaled))
        aics.append(gm.aic(X_scaled))
    return ks, bics, aics


def best_k(scores: list[float]) -> int:
    """점수 리스트에서 최솟값 인덱스 → 최적 k 반환."""
    return int(np.argmin(scores)) + 1


def draw_ellipse(ax: plt.Axes, mean: np.ndarray, cov: np.ndarray,
                 color: str, n_std: float = 2.0, alpha: float = 0.25) -> None:
    """PCA 2D 공분산 타원 그리기."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor=color, edgecolor=color,
                  alpha=alpha, linewidth=1.5, linestyle="--")
    ax.add_patch(ell)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 피처 로드
    if not FEAT_PATH.exists():
        raise FileNotFoundError(
            f"{FEAT_PATH} 없음. 먼저 analyze_normal_cluster.py를 실행하세요.")

    X_raw = np.load(FEAT_PATH)
    y     = np.load(LABEL_PATH)
    print(f"피처 로드: {X_raw.shape}, labels={dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 1. 전체 / RPM별 BIC/AIC 스윕 ────────────────────────────────────────
    print("\n=== [1/3] GMM BIC/AIC 스윕 ===")
    results: dict[str, dict] = {}

    for name, lbl_filter in SUBSETS.items():
        mask = np.ones(len(y), dtype=bool) if lbl_filter is None else (y == lbl_filter)
        X_sub = X_raw[mask]
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_sub)

        ks, bics, aics = fit_gmm_sweep(X_sc, MAX_COMPONENTS)
        k_bic = best_k(bics)
        k_aic = best_k(aics)
        results[name] = dict(X_sub=X_sub, X_sc=X_sc, y_sub=y[mask],
                              ks=ks, bics=bics, aics=aics,
                              k_bic=k_bic, k_aic=k_aic)
        print(f"  [{name:6s}]  n={mask.sum():3d}  최적 k (BIC)={k_bic}  (AIC)={k_aic}")

    # ── 2. BIC/AIC 곡선 플롯 ─────────────────────────────────────────────────
    print("\n=== [2/3] BIC/AIC 곡선 저장 ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("GMM Component Selection (BIC / AIC)", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        ks   = res["ks"]
        bics = np.array(res["bics"])
        aics = np.array(res["aics"])
        # 정규화하여 비교 용이하게
        bics_n = (bics - bics.min()) / (bics.max() - bics.min() + 1e-9)
        aics_n = (aics - aics.min()) / (aics.max() - aics.min() + 1e-9)

        ax.plot(ks, bics_n, "o-", color="#E53935", label=f"BIC (k*={res['k_bic']})")
        ax.plot(ks, aics_n, "s--", color="#1E88E5", label=f"AIC (k*={res['k_aic']})")
        ax.axvline(res["k_bic"], color="#E53935", linewidth=0.8, linestyle=":")
        ax.axvline(res["k_aic"], color="#1E88E5", linewidth=0.8, linestyle=":")
        ax.set_title(f"{name}  (n={len(res['X_sub'])})")
        ax.set_xlabel("Number of Components (k)")
        ax.set_ylabel("Normalized Score")
        ax.set_xticks(ks)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out1 = OUT_DIR / "gmm_bic_aic.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {out1}")

    # ── 3. PCA 2D + GMM 타원 오버레이 ────────────────────────────────────────
    print("\n=== [3/3] GMM 타원 오버레이 저장 ===")

    # PCA는 각 subset에서 독립 학습
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("GMM Ellipses on PCA 2D  (2σ, optimal k by BIC)",
                  fontsize=13, fontweight="bold")

    ELLIPSE_PALETTE = ["#FF7043", "#26C6DA", "#66BB6A", "#AB47BC", "#FFA726", "#8D6E63"]

    for ax, (name, res) in zip(axes2, results.items()):
        X_sc  = res["X_sc"]
        y_sub = res["y_sub"]
        k_opt = res["k_bic"]

        # PCA 2D
        pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
        X_2d = pca2.fit_transform(X_sc)

        # GMM 최적 k로 재학습 (PCA 전 스케일된 공간 기준)
        gm = GaussianMixture(n_components=k_opt, covariance_type="full",
                              n_init=N_INIT, random_state=RANDOM_STATE)
        gm.fit(X_sc)
        labels_gmm = gm.predict(X_sc)

        # 산점도: RPM 색상
        for lbl, color in RPM_COLORS.items():
            mask = y_sub == lbl
            if mask.any():
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=color, label=RPM_NAMES[lbl],
                           s=35, alpha=0.75, edgecolors="none", zorder=3)

        # GMM 타원: PCA 2D 공간으로 공분산 투영
        W = pca2.components_          # (2, D)
        for i in range(k_opt):
            mean_2d = W @ gm.means_[i]
            cov_2d  = W @ gm.covariances_[i] @ W.T
            draw_ellipse(ax, mean_2d, cov_2d,
                         color=ELLIPSE_PALETTE[i % len(ELLIPSE_PALETTE)])
            ax.plot(*mean_2d, "x", color=ELLIPSE_PALETTE[i % len(ELLIPSE_PALETTE)],
                    markersize=8, markeredgewidth=2, zorder=4)

        pv1 = pca2.explained_variance_ratio_[0]
        pv2 = pca2.explained_variance_ratio_[1]
        ax.set_title(f"{name}  k*={k_opt} (BIC)\nPC1={pv1:.1%} PC2={pv2:.1%}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out2 = OUT_DIR / "gmm_ellipses.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"  → 저장: {out2}")

    # ── 텍스트 요약 저장 ─────────────────────────────────────────────────────
    summary_lines = ["===== Method B: GMM BIC/AIC 군집 수 분석 요약 =====\n"]
    for name, res in results.items():
        summary_lines.append(
            f"[{name}]  n={len(res['X_sub'])}  "
            f"BIC 최적 k={res['k_bic']}  AIC 최적 k={res['k_aic']}"
        )
    summary_text = "\n".join(summary_lines)
    (OUT_DIR / "gmm_summary.txt").write_text(summary_text, encoding="utf-8")

    print("\n" + summary_text)
    print(f"\n결과 저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
