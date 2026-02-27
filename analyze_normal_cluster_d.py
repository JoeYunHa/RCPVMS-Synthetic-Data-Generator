"""Method D: Orbit 이미지 기반 시각 검사 + 픽셀 공간 차원 축소

Deep SVDD의 실제 입력(orbit 이미지)과 동일한 공간에서 군집 구조를 확인한다.

분석 내용:
  (1) RPM별 orbit 이미지 샘플 갤러리 (베어링 쌍 4개 × 파일 4개)
  (2) 픽셀 공간 PCA → t-SNE 2D 산점도 (img_size=64 다운샘플)
  (3) 채널별(베어링 쌍별) 평균 orbit 이미지 비교 (1200 vs 3600)

출력:
  data/analysis/normal_cluster/orbit_gallery_1200rpm.png
  data/analysis/normal_cluster/orbit_gallery_3600rpm.png
  data/analysis/normal_cluster/orbit_pca_tsne.png
  data/analysis/normal_cluster/orbit_mean_compare.png
"""

from __future__ import annotations

import random
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
from src.utils.orbit import make_orbit_stack, BEARING_PAIRS

# ── 설정 ─────────────────────────────────────────────────────────────────────
OUT_DIR   = Path("data/analysis/normal_cluster")
IMG_SIZE  = 64       # PCA용 축소 해상도 (4×64×64 = 16384 dims)
IMG_FULL  = 128      # 갤러리/평균 비교용 해상도
GALLERY_N = 4        # RPM별 갤러리 파일 수
RANDOM_SEED = 42

RPM_CONFIG = {
    "3600rpm": {"dir": Path("data/raw/normal_3600rpm"), "label": 2},
    "1200rpm": {"dir": Path("data/raw/normal_1200rpm"), "label": 1},
}
RPM_COLORS = {1: "#2196F3", 2: "#F44336"}
RPM_NAMES  = {1: "1200 rpm", 2: "3600 rpm"}
PAIR_NAMES = ["Bearing 1 (ch0,1)", "Bearing 2 (ch4,5)",
              "Bearing 3 (ch10,11)", "Bearing 4 (ch16,17)"]
_MAX_CH    = 17
_ORBIT_CHS = [ch for pair in BEARING_PAIRS for ch in pair]


# ── BIN → orbit 이미지 ────────────────────────────────────────────────────────
def load_orbit(bin_path: Path, img_size: int) -> np.ndarray | None:
    """BIN 파일 → (4, img_size, img_size) orbit 스택. 실패 시 None."""
    parser = RCPVMSParser(str(bin_path))
    try:
        parser.parse_header()
    except Exception:
        return None

    mils_per_v = 10.0
    if parser.header and parser.header.extra_fields:
        mpv = parser.header.extra_fields.get("mils_per_v", 0.0)
        if mpv and mpv > 0:
            mils_per_v = float(mpv)

    channels: list[np.ndarray] = [np.array([])] * (_MAX_CH + 1)
    for ch in _ORBIT_CHS:
        channels[ch] = parser.read_channel(ch)

    stack = make_orbit_stack(channels, BEARING_PAIRS, mils_per_v,
                             img_size=img_size)
    if stack.max() == 0:
        return None
    return stack


def collect_orbits(img_size: int) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """전체 정상 파일 orbit 이미지 수집."""
    stacks, labels, names = [], [], []
    for rpm_key, cfg in RPM_CONFIG.items():
        d = cfg["dir"]
        if not d.exists():
            print(f"[WARN] 경로 없음: {d}")
            continue
        bins = sorted(set(d.glob("*.BIN")) | set(d.glob("*.bin")))
        for p in tqdm(bins, desc=f"{rpm_key}", unit="file"):
            st = load_orbit(p, img_size)
            if st is None:
                continue
            stacks.append(st)
            labels.append(cfg["label"])
            names.append(p.name)
    return stacks, np.array(labels, dtype=np.int32), names


# ── 1. 갤러리 ─────────────────────────────────────────────────────────────────
def plot_gallery(bins: list[Path], rpm_name: str, color: str) -> plt.Figure:
    """GALLERY_N개 파일 × 4 베어링 쌍 격자 출력."""
    rng = random.Random(RANDOM_SEED)
    chosen = rng.sample(bins, min(GALLERY_N, len(bins)))

    fig, axes = plt.subplots(GALLERY_N, 4, figsize=(10, GALLERY_N * 2.5))
    fig.suptitle(f"Orbit Image Gallery — {rpm_name}", fontsize=12,
                 fontweight="bold", color=color)

    for row, p in enumerate(chosen):
        st = load_orbit(p, IMG_FULL)
        if st is None:
            st = np.zeros((4, IMG_FULL, IMG_FULL), dtype=np.float32)
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(st[col], cmap="hot", vmin=0, vmax=1,
                      interpolation="nearest", origin="lower")
            if row == 0:
                ax.set_title(PAIR_NAMES[col], fontsize=7)
            ax.set_xlabel(p.name[:18], fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    return fig


# ── 2. 평균 orbit 비교 ────────────────────────────────────────────────────────
def plot_mean_orbit(stacks: list[np.ndarray],
                    labels: np.ndarray) -> plt.Figure:
    """1200rpm / 3600rpm 각 베어링 쌍의 평균 orbit 이미지 비교."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Mean Orbit Image per RPM (4 bearing pairs)",
                 fontsize=12, fontweight="bold")

    for row, (lbl, rpm_name, color) in enumerate(
        [(1, "1200 rpm", "#2196F3"), (2, "3600 rpm", "#F44336")]
    ):
        mask = labels == lbl
        group = np.stack([stacks[i] for i in np.where(mask)[0]])  # (n, 4, H, W)
        mean_img = group.mean(axis=0)                              # (4, H, W)
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(mean_img[col], cmap="hot", vmin=0,
                      interpolation="nearest", origin="lower")
            if row == 0:
                ax.set_title(PAIR_NAMES[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{rpm_name}\n(n={mask.sum()})",
                              fontsize=8, color=color, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    return fig


# ── 3. 픽셀 PCA + t-SNE ──────────────────────────────────────────────────────
def plot_pixel_pca_tsne(stacks: list[np.ndarray],
                        labels: np.ndarray) -> plt.Figure:
    """픽셀 공간 PCA 2D + t-SNE 2D 산점도."""
    X_flat = np.stack([s.flatten() for s in stacks])   # (N, 4*64*64)
    print(f"  픽셀 행렬: {X_flat.shape}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_flat)

    # PCA
    pca = PCA(n_components=10, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_sc)
    pv = pca.explained_variance_ratio_
    print(f"  PC1={pv[0]:.1%}  PC2={pv[1]:.1%}  (PC1+2 합계={pv[:2].sum():.1%})")

    # t-SNE
    n = len(labels)
    perp = min(30, max(5, n // 3))
    print(f"  t-SNE perplexity={perp}")
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                init="pca", learning_rate="auto", random_state=RANDOM_SEED)
    X_tsne = tsne.fit_transform(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Orbit Pixel Space: PCA & t-SNE (img_size=64)",
                 fontsize=12, fontweight="bold")

    for ax, coords, title in [
        (axes[0], X_pca[:, :2],
         f"PCA  (PC1={pv[0]:.1%}, PC2={pv[1]:.1%})"),
        (axes[1], X_tsne,
         f"t-SNE  (perplexity={perp})"),
    ]:
        for lbl, color in RPM_COLORS.items():
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, label=RPM_NAMES[lbl],
                       s=40, alpha=0.75, edgecolors="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 갤러리 (full resolution) ─────────────────────────────────────────────
    print("=== [1/4] 갤러리 생성 ===")
    for rpm_key, cfg in RPM_CONFIG.items():
        d = cfg["dir"]
        if not d.exists():
            continue
        bins = sorted(set(d.glob("*.BIN")) | set(d.glob("*.bin")))
        fig = plot_gallery(bins, RPM_NAMES[cfg["label"]],
                           RPM_COLORS[cfg["label"]])
        out = OUT_DIR / f"orbit_gallery_{rpm_key}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  -> {out}")

    # ── 전체 orbit 수집 (PCA용 64×64) ────────────────────────────────────────
    print("\n=== [2/4] Orbit 이미지 수집 (64×64) ===")
    stacks, labels, _ = collect_orbits(IMG_SIZE)
    print(f"  -> {len(stacks)}개 로드  "
          f"(1200rpm={int((labels==1).sum())}, 3600rpm={int((labels==2).sum())})")

    # ── 평균 orbit 비교 ──────────────────────────────────────────────────────
    print("\n=== [3/4] 평균 Orbit 비교 ===")
    # 평균 비교는 full resolution으로 별도 수집
    stacks_full, labels_full, _ = collect_orbits(IMG_FULL)
    fig3 = plot_mean_orbit(stacks_full, labels_full)
    out3 = OUT_DIR / "orbit_mean_compare.png"
    fig3.savefig(out3, dpi=130)
    plt.close(fig3)
    print(f"  -> {out3}")

    # ── 픽셀 PCA + t-SNE ─────────────────────────────────────────────────────
    print("\n=== [4/4] 픽셀 공간 PCA + t-SNE ===")
    fig4 = plot_pixel_pca_tsne(stacks, labels)
    out4 = OUT_DIR / "orbit_pca_tsne.png"
    fig4.savefig(out4, dpi=150)
    plt.close(fig4)
    print(f"  -> {out4}")

    print(f"\n완료. 결과 저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
