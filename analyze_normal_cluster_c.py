"""Method C: Pairwise 거리 분포 + KDE 분석

군집 가정 없이 거리 분포 형태(unimodal vs multimodal)를 직접 확인한다.

분석 내용:
  (1) 전체 pairwise 거리 KDE  (intra-1200 / intra-3600 / inter-RPM 분리)
  (2) 각 RPM 단독 pairwise 거리 KDE + unimodal 검정 (Hartigan's dip test)
  (3) 각 파일의 최근접 이웃 거리 분포 (nearest-neighbor distance)
  (4) 거리 행렬 히트맵 (RPM 레이블 정렬)

출력:
  data/analysis/normal_cluster/pairwise_kde.png
  data/analysis/normal_cluster/pairwise_heatmap.png
  data/analysis/normal_cluster/nn_distance.png
  data/analysis/normal_cluster/distance_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

# ── 설정 ─────────────────────────────────────────────────────────────────────
FEAT_PATH  = Path("data/analysis/normal_cluster/features.npy")
LABEL_PATH = Path("data/analysis/normal_cluster/labels.npy")
OUT_DIR    = Path("data/analysis/normal_cluster")

RPM_COLORS = {1: "#2196F3", 2: "#F44336"}
RPM_NAMES  = {1: "1200 rpm", 2: "3600 rpm"}


# ── Hartigan Dip Test (unimodality) ──────────────────────────────────────────
def dip_test(x: np.ndarray, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    """Bootstrap Hartigan dip statistic.
    dip > 0.05 and p < 0.05 → multimodal 가능성.
    Returns (dip_stat, p_value).
    """
    rng = np.random.default_rng(seed)
    x_sorted = np.sort(x)
    n = len(x_sorted)

    def _dip(arr: np.ndarray) -> float:
        ecdf = np.arange(1, len(arr) + 1) / len(arr)
        uniform = np.linspace(arr[0], arr[-1], len(arr))
        uniform_cdf = (uniform - arr[0]) / (arr[-1] - arr[0] + 1e-12)
        return float(np.max(np.abs(ecdf - uniform_cdf)))

    obs_dip = _dip(x_sorted)
    boot_dips = np.array([
        _dip(np.sort(rng.uniform(x_sorted[0], x_sorted[-1], n)))
        for _ in range(n_boot)
    ])
    p_val = float((boot_dips >= obs_dip).mean())
    return obs_dip, p_val


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    if not FEAT_PATH.exists():
        raise FileNotFoundError(
            f"{FEAT_PATH} 없음. 먼저 analyze_normal_cluster.py를 실행하세요.")

    X_raw = np.load(FEAT_PATH)
    y     = np.load(LABEL_PATH)
    print(f"피처 로드: {X_raw.shape}, labels={dict(zip(*np.unique(y, return_counts=True)))}")

    # 표준화
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    m1 = y == 1   # 1200rpm
    m2 = y == 2   # 3600rpm

    # ── 1. 전체 pairwise 거리 행렬 ──────────────────────────────────────────
    print("\n=== [1/4] Pairwise 거리 계산 ===")
    D_full = squareform(pdist(X, metric="euclidean"))

    # 마스크별 거리 벡터 (대각 제외)
    idx_all  = np.arange(len(y))
    def off_diag_pairs(mask_a, mask_b):
        ia = np.where(mask_a)[0]
        ib = np.where(mask_b)[0]
        vals = []
        for i in ia:
            for j in ib:
                if i != j:
                    vals.append(D_full[i, j])
        return np.array(vals)

    d_intra1 = off_diag_pairs(m1, m1)   # 1200 vs 1200
    d_intra2 = off_diag_pairs(m2, m2)   # 3600 vs 3600
    d_inter  = off_diag_pairs(m1, m2)   # 1200 vs 3600

    print(f"  intra-1200: {len(d_intra1)} pairs, mean={d_intra1.mean():.3f}, std={d_intra1.std():.3f}")
    print(f"  intra-3600: {len(d_intra2)} pairs, mean={d_intra2.mean():.3f}, std={d_intra2.std():.3f}")
    print(f"  inter-RPM:  {len(d_inter)}  pairs, mean={d_inter.mean():.3f},  std={d_inter.std():.3f}")

    # ── 2. KDE 플롯 ─────────────────────────────────────────────────────────
    print("\n=== [2/4] KDE 플롯 저장 ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Pairwise Distance Distribution (KDE)", fontsize=13, fontweight="bold")

    def plot_kde(ax, dist_dict: dict, title: str, xlim=None):
        for label, (d, color, ls) in dist_dict.items():
            if len(d) < 3:
                continue
            kde = gaussian_kde(d, bw_method="scott")
            xs  = np.linspace(d.min(), d.max(), 500)
            ax.plot(xs, kde(xs), color=color, linestyle=ls, linewidth=2, label=label)
            ax.axvline(d.mean(), color=color, linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        if xlim:
            ax.set_xlim(xlim)

    # (a) 전체: intra/inter 분리
    x_max = max(d_intra1.max(), d_intra2.max(), d_inter.max())
    plot_kde(axes[0], {
        "intra-1200rpm": (d_intra1, "#2196F3", "-"),
        "intra-3600rpm": (d_intra2, "#F44336", "-"),
        "inter-RPM":     (d_inter,  "#4CAF50", "--"),
    }, "All: intra vs inter RPM", xlim=(0, x_max * 1.05))

    # (b) 1200rpm 단독
    plot_kde(axes[1], {
        "intra-1200rpm": (d_intra1, "#2196F3", "-"),
    }, f"1200rpm only  (n=82)")

    # (c) 3600rpm 단독
    plot_kde(axes[2], {
        "intra-3600rpm": (d_intra2, "#F44336", "-"),
    }, f"3600rpm only  (n=28)")

    plt.tight_layout()
    out1 = OUT_DIR / "pairwise_kde.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"  -> 저장: {out1}")

    # ── 3. Nearest-Neighbor 거리 분포 ────────────────────────────────────────
    print("\n=== [3/4] Nearest-Neighbor 거리 ===")
    np.fill_diagonal(D_full, np.inf)
    nn_dist = D_full.min(axis=1)
    np.fill_diagonal(D_full, 0)

    nn1 = nn_dist[m1]
    nn2 = nn_dist[m2]

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    fig2.suptitle("Nearest-Neighbor Distance Distribution", fontsize=13, fontweight="bold")

    for ax, nn, color, name, n in [
        (axes2[0], nn1, "#2196F3", "1200rpm", m1.sum()),
        (axes2[1], nn2, "#F44336", "3600rpm", m2.sum()),
    ]:
        ax.hist(nn, bins=15, color=color, alpha=0.55, edgecolor="white",
                density=True, label="histogram")
        if len(nn) >= 3:
            kde = gaussian_kde(nn, bw_method="scott")
            xs  = np.linspace(nn.min() * 0.9, nn.max() * 1.1, 300)
            ax.plot(xs, kde(xs), color=color, linewidth=2.5, label="KDE")
        ax.axvline(nn.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={nn.mean():.2f}")
        ax.set_title(f"{name}  (n={n})")
        ax.set_xlabel("NN Distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out2 = OUT_DIR / "nn_distance.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"  -> 저장: {out2}")

    # ── 4. 거리 행렬 히트맵 (RPM 정렬) ──────────────────────────────────────
    print("\n=== [4/4] 거리 행렬 히트맵 저장 ===")
    sort_idx = np.argsort(y)          # 1200rpm 먼저, 3600rpm 다음
    D_sorted = D_full[np.ix_(sort_idx, sort_idx)]
    n1, n2   = m1.sum(), m2.sum()

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    im = ax3.imshow(D_sorted, aspect="auto", cmap="viridis_r",
                    interpolation="nearest")
    plt.colorbar(im, ax=ax3, label="Euclidean Distance")

    # RPM 경계선
    ax3.axhline(n1 - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax3.axvline(n1 - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax3.text(n1 / 2, -2.5, "1200rpm", ha="center", color="white",
             fontsize=8, fontweight="bold")
    ax3.text(n1 + n2 / 2, -2.5, "3600rpm", ha="center", color="white",
             fontsize=8, fontweight="bold")
    ax3.set_title("Pairwise Distance Matrix (sorted by RPM)")
    ax3.set_xlabel("File index")
    ax3.set_ylabel("File index")

    plt.tight_layout()
    out3 = OUT_DIR / "pairwise_heatmap.png"
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"  -> 저장: {out3}")

    # ── Dip Test ─────────────────────────────────────────────────────────────
    print("\n=== Hartigan Dip Test (unimodality) ===")
    dip_results = {}
    for name, d in [("intra-1200rpm", d_intra1), ("intra-3600rpm", d_intra2),
                    ("inter-RPM", d_inter)]:
        dip_stat, p_val = dip_test(d)
        multimodal = "multimodal 가능성" if p_val < 0.05 else "unimodal"
        print(f"  {name:15s}  dip={dip_stat:.4f}  p={p_val:.3f}  -> {multimodal}")
        dip_results[name] = (dip_stat, p_val)

    # ── 텍스트 요약 ───────────────────────────────────────────────────────────
    lines = [
        "===== Method C: Pairwise Distance 분석 요약 =====\n",
        f"intra-1200rpm  mean={d_intra1.mean():.3f}  std={d_intra1.std():.3f}",
        f"intra-3600rpm  mean={d_intra2.mean():.3f}  std={d_intra2.std():.3f}",
        f"inter-RPM      mean={d_inter.mean():.3f}  std={d_inter.std():.3f}",
        f"inter/intra-1200 ratio = {d_inter.mean()/d_intra1.mean():.2f}",
        f"inter/intra-3600 ratio = {d_inter.mean()/d_intra2.mean():.2f}",
        "\n[Dip Test]",
    ]
    for name, (ds, pv) in dip_results.items():
        lines.append(f"  {name:15s}  dip={ds:.4f}  p={pv:.3f}")
    summary = "\n".join(lines)
    (OUT_DIR / "distance_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\n결과 저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
