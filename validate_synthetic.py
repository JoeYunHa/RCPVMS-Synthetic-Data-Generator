#!/usr/bin/env python3
"""
validate_synthetic.py
---------------------
RCPVMSParser를 사용하여 합성 데이터의 주파수 특성을 검증한다.

Usage:
    python validate_synthetic.py [--data_dir PATH] [--n_files N] [--fs FS]
"""

import os
import sys
import glob
import json
import argparse
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.core.rcpvms_parser import RCPVMSParser

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── 운전 조건별 소스 경로 정의 ────────────────────────────────────────────────
# condition → {class: (subdir, glob_pattern)}
CONDITION_SOURCES = {
    # 혼합(기존): 분리 전 원본 구조
    "mixed": {
        "normal":       ("raw/normal",                   "*.BIN"),
        "unbalance":    ("synthetic/unbalance",          "*.bin"),
        "misalignment": ("synthetic/misalignment",       "*.bin"),
        "oil_whip":     ("synthetic/oil_whip",           "*.bin"),
        "abnormal":     ("raw/abnormal",                 "*.BIN"),
    },
    # 3600 RPM (2506-2507 계열)
    "3600rpm": {
        "normal":       ("raw/normal_3600rpm",           "*.BIN"),
        "unbalance":    ("synthetic/3600rpm/unbalance",  "*.bin"),
        "misalignment": ("synthetic/3600rpm/misalignment","*.bin"),
        "oil_whip":     ("synthetic/3600rpm/oil_whip",   "*.bin"),
        "abnormal":     ("raw/abnormal",                 "*.BIN"),
    },
    # 1200 RPM (2511 계열)
    "1200rpm": {
        "normal":       ("raw/normal_1200rpm",           "*.BIN"),
        "unbalance":    ("synthetic/1200rpm/unbalance",  "*.bin"),
        "misalignment": ("synthetic/1200rpm/misalignment","*.bin"),
        "oil_whip":     ("synthetic/1200rpm/oil_whip",   "*.bin"),
        "abnormal":     ("raw/abnormal",                 "*.BIN"),
    },
}

# 클래스별 이론 지배 주파수 배수 (None = 자동 판정 없음)
FAULT_DOMINANT = {
    "normal":       None,
    "unbalance":    1.00,
    "misalignment": 2.00,
    "oil_whip":     0.45,
    "abnormal":     None,
}

# 기본값: 3600rpm (mixed는 RPM 분리 전 구 경로 구조용으로만 유지)
SOURCES = CONDITION_SOURCES["3600rpm"]

SEP = "=" * 66


# ── FFT 유틸리티 ───────────────────────────────────────────────────────────────

def compute_fft(sig: np.ndarray, fs: int):
    n = len(sig)
    mags = np.abs(np.fft.rfft(sig)) * 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mags


def get_band_amplitude(freqs: np.ndarray, mags: np.ndarray,
                       center_hz: float, half_bw_hz: float = 2.0) -> float:
    mask = np.abs(freqs - center_hz) <= half_bw_hz
    return float(np.max(mags[mask])) if np.any(mask) else 0.0


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────

def load_signals(class_name: str, data_dir: str,
                 n_files: int, fs: int, seg_sec: float = 1.0,
                 _sources: dict | None = None):
    """
    BIN 파일에서 신호를 로드한다.
    seg_sec 구간을 파일 내 가능한 한 뒤쪽(9 s 지점)에서 추출한다.
    """
    src = _sources if _sources is not None else SOURCES
    subdir, pattern = src[class_name]
    class_path = os.path.join(data_dir, subdir)
    bin_files = sorted(glob.glob(os.path.join(class_path, pattern)))[:n_files]

    if not bin_files:
        print(f"  [경고] {class_name}: 파일 없음 ({class_path})")
        return [], 0

    signals = []
    for bin_path in bin_files:
        try:
            parser = RCPVMSParser(bin_path)
            fs_file = parser.header.sampling_rate if parser.header.sampling_rate > 0 else fs
            channels = parser.read_all_channels()
            if not channels:
                continue

            seg_n = int(seg_sec * fs_file)
            ref_len = len(channels[0])
            s = min(9 * fs_file, max(0, ref_len - seg_n))
            e = s + seg_n

            for ch_data in channels:
                if len(ch_data) >= e and seg_n > 0:
                    signals.append(ch_data[s:e].astype(np.float32))
        except Exception as ex:
            print(f"    [오류] {os.path.basename(bin_path)}: {ex}")

    print(f"  {class_name:<15}: {len(bin_files):>3} 파일 → {len(signals):>4} 신호")
    return signals, len(bin_files)


# ── 1X 주파수 자동 검출 ────────────────────────────────────────────────────────

def detect_1x_freq(signals: list, fs: int,
                   search: tuple = (20.0, 70.0)) -> float:
    sum_mags = None
    ref_freqs = None
    for sig in signals:
        freqs, mags = compute_fft(sig, fs)
        if sum_mags is None:
            sum_mags, ref_freqs = mags.copy(), freqs
        else:
            sum_mags += mags
    if sum_mags is None:
        return 30.0
    mask = (ref_freqs >= search[0]) & (ref_freqs <= search[1])
    return float(ref_freqs[mask][np.argmax(sum_mags[mask])])


# ── FFT 분석 ───────────────────────────────────────────────────────────────────

def run_fft_analysis(signals_by_class: dict, freq_1x: float, fs: int) -> dict:
    bands = {
        "0.45x": freq_1x * 0.45,
        "1x":    freq_1x * 1.00,
        "2x":    freq_1x * 2.00,
        "3x":    freq_1x * 3.00,
    }
    results = {}
    for class_name, signals in signals_by_class.items():
        acc = {k: [] for k in bands}
        for sig in signals:
            freqs, mags = compute_fft(sig, fs)
            for band, center in bands.items():
                acc[band].append(get_band_amplitude(freqs, mags, center))
        results[class_name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in acc.items()
        }
    return results


# ── RMS 통계 ───────────────────────────────────────────────────────────────────

def run_rms_stats(signals_by_class: dict) -> dict:
    results = {}
    for class_name, signals in signals_by_class.items():
        rms_vals = [float(np.sqrt(np.mean(s ** 2))) for s in signals]
        results[class_name] = {
            "mean": float(np.mean(rms_vals)),
            "std":  float(np.std(rms_vals)),
            "n":    len(rms_vals),
        }
    return results


# ── 타당성 판정 ────────────────────────────────────────────────────────────────

def check_validity(fft_results: dict) -> dict:
    band_key = {0.45: "0.45x", 1.00: "1x", 2.00: "2x", 3.00: "3x"}
    validity = {}
    for class_name, metrics in fft_results.items():
        dominant_mult = FAULT_DOMINANT.get(class_name)
        if dominant_mult is None:
            validity[class_name] = {"pass": None, "reason": "판정 기준 없음 (수동 확인)"}
            continue

        amps = {m: metrics[band_key[m]]["mean"] for m in band_key}
        actual_max = max(amps, key=amps.get)
        passed = abs(actual_max - dominant_mult) < 0.1

        if passed:
            validity[class_name] = {
                "pass": True,
                "reason": (
                    f"{dominant_mult}X = {amps[dominant_mult]:.5f}  "
                    f"(최대 대역 확인 ✓)"
                ),
            }
        else:
            validity[class_name] = {
                "pass": False,
                "reason": (
                    f"예상 {dominant_mult}X = {amps[dominant_mult]:.5f},  "
                    f"실제 최대 {actual_max}X = {amps[actual_max]:.5f}"
                ),
            }
    return validity


# ── 보고서 출력 ────────────────────────────────────────────────────────────────

def print_report(fft_results, rms_stats, validity, freq_1x, out_dir,
                 sample_counts):
    print(f"\n{SEP}")
    print(f"  검증 보고서   (1X = {freq_1x:.2f} Hz / {freq_1x*60:.0f} RPM)")
    print(f"  분석 신호 수: {sum(sample_counts.values())} "
          f"({', '.join(f'{k}:{v}' for k,v in sample_counts.items())})")

    # FFT 진폭 테이블
    print(f"\n  {'-'*60}")
    print("  [1] FFT 주파수 진폭 (mean ± std)")
    print(f"  {'클래스':<15} {'0.45X':>12} {'1X':>12} {'2X':>12} {'3X':>12}")
    print(f"  {'-'*59}")
    for cn, m in fft_results.items():
        def fmt(k): return f"{m[k]['mean']:.5f}±{m[k]['std']:.5f}"
        print(f"  {cn:<15} {fmt('0.45x'):>12} {fmt('1x'):>12} "
              f"{fmt('2x'):>12} {fmt('3x'):>12}")

    # RMS 통계
    print(f"\n  {'-'*60}")
    print("  [2] RMS 통계")
    print(f"  {'클래스':<15} {'RMS mean':>12} {'RMS std':>10} {'n':>5}")
    print(f"  {'-'*44}")
    for cn, s in rms_stats.items():
        print(f"  {cn:<15} {s['mean']:>12.5f} {s['std']:>10.5f} {s['n']:>5}")

    # 타당성 판정
    print(f"\n  {'-'*60}")
    print("  [3] 타당성 자동 판정")
    pass_n, total_n = 0, 0
    for cn, v in validity.items():
        if v["pass"] is True:
            sym = "✓ PASS"; pass_n += 1; total_n += 1
        elif v["pass"] is False:
            sym = "✗ FAIL"; total_n += 1
        else:
            sym = "? N/A "
        print(f"  [{sym}] {cn:<15} {v['reason']}")

    # 종합 결론
    print(f"\n  {'-'*60}")
    if total_n == 0:
        print("  → 판정 대상 없음")
    elif pass_n == total_n:
        print(f"  → 전체 PASS ({pass_n}/{total_n}) — 모든 합성 클래스 이론 특성 충족 ✓")
    else:
        failed = [cn for cn, v in validity.items() if v["pass"] is False]
        print(f"  → 일부 FAIL ({pass_n}/{total_n}) — 재검토 필요: {', '.join(failed)}")
    print(SEP)

    # JSON 저장
    report = {
        "freq_1x_hz":  freq_1x,
        "freq_1x_rpm": round(freq_1x * 60),
        "sample_counts": sample_counts,
        "fft_results": fft_results,
        "rms_stats":   rms_stats,
        "validity":    validity,
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "validation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  보고서 저장: {json_path}")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="합성 데이터 타당성 검증")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "data"),
        help="data/raw, data/synthetic 이 위치한 상위 디렉토리",
    )
    p.add_argument(
        "--condition",
        choices=list(CONDITION_SOURCES.keys()),
        default="3600rpm",
        help=(
            "운전 조건 선택: '3600rpm'(2506계), '1200rpm'(2511계), 'mixed'(구 경로 호환). "
            "각 조건에 맞는 정상/합성 파일 경로가 자동 설정됩니다."
        ),
    )
    p.add_argument("--n_files", type=int, default=10,
                   help="클래스당 최대 파일 수")
    p.add_argument("--fs", type=int, default=40_000,
                   help="기본 샘플링 주파수 (파일 헤더 우선)")
    p.add_argument(
        "--out_dir",
        default=None,
        help="검증 보고서 출력 디렉토리 (기본: data/validation/<condition>/)",
    )
    args = p.parse_args()

    sources = CONDITION_SOURCES[args.condition]
    out_dir = args.out_dir or os.path.join(
        SCRIPT_DIR, "data", "validation", args.condition
    )

    t_start = time.time()
    print(f"\n{'='*66}")
    print("  RCPVMS 합성 데이터 타당성 검증")
    print(f"  condition: {args.condition}")
    print(f"  data_dir : {args.data_dir}")
    print(f"  n_files  : {args.n_files} per class")
    print(f"{'='*66}")

    # [1] 로딩
    print("\n[1] 데이터 로딩")
    signals_by_class = {}
    sample_counts = {}
    for cls in sources:
        sigs, _ = load_signals(cls, args.data_dir, args.n_files, args.fs,
                               _sources=sources)
        signals_by_class[cls] = sigs
        sample_counts[cls] = len(sigs)

    if not signals_by_class.get("normal"):
        print("\n[오류] 정상 신호를 찾을 수 없습니다. --data_dir 을 확인하세요.")
        sys.exit(1)

    # [2] 1X 주파수 검출
    print("\n[2] 1X 회전 주파수 자동 검출 (정상 신호 FFT)")
    freq_1x = detect_1x_freq(signals_by_class["normal"], args.fs)
    print(f"  → 1X = {freq_1x:.2f} Hz  ({freq_1x*60:.0f} RPM)")

    # [3] FFT 분석
    print("\n[3] FFT 주파수 분석")
    fft_results = run_fft_analysis(signals_by_class, freq_1x, args.fs)

    # [4] RMS 통계
    print("\n[4] RMS 통계")
    rms_stats = run_rms_stats(signals_by_class)

    # [5] 타당성 판정
    print("\n[5] 타당성 판정")
    validity = check_validity(fft_results)
    for cn, v in validity.items():
        sym = "✓" if v["pass"] is True else ("✗" if v["pass"] is False else "?")
        print(f"  [{sym}] {cn:<15} {v['reason']}")

    # [6] 보고서
    print_report(fft_results, rms_stats, validity, freq_1x,
                 out_dir, sample_counts)

    print(f"\n  총 소요 시간: {time.time() - t_start:.1f}s\n")


if __name__ == "__main__":
    main()
