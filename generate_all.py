#!/usr/bin/env python3
"""
RCPVMS 합성 데이터 일괄 생성 스크립트
=======================================
두 RPM 운전 조건별로 전체 결함 유형 + 과도(Transient) 변형을 생성한다.

출력 구조:
  data/raw/normal_3600rpm/   ← 원본 hard link (250 계열, 28 파일)
  data/raw/normal_1200rpm/   ← 원본 hard link (251 계열, 82 파일)
  data/synthetic/3600rpm/unbalance/
  data/synthetic/3600rpm/misalignment/
  data/synthetic/3600rpm/oil_whip/
  data/synthetic/1200rpm/unbalance/
  data/synthetic/1200rpm/misalignment/
  data/synthetic/1200rpm/oil_whip/

Usage:
    venv\\Scripts\\python.exe generate_all.py [--dry-run]
"""

import sys
import os
import glob
import shutil
import subprocess
import argparse
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── 설정 ─────────────────────────────────────────────────────────────────────

PYTHON      = Path("venv/Scripts/python.exe")
NORMAL_DIR  = Path("data/raw/normal")
FAULT_TYPES = ["unbalance", "misalignment", "oil_whip"]

# RPM 계열 파일명 패턴
RPM_PATTERNS = {
    "3600rpm": "RCPVMS_BKG_250*.BIN",   # 2506~2507 계열
    "1200rpm": "RCPVMS_BKG_251*.BIN",   # 2511 계열
}

# 생성 수량 / 심각도
GEN = {
    "continuous": {
        "count":    100,
        "sev_min":  0.3,
        "sev_max":  3.0,
        "transient": False,
    },
    "transient": {
        "count":    50,
        "sev_min":  0.3,
        "sev_max":  1.5,
        "transient": True,
        "active_cycles": 3.0,
        "silent_cycles": 10.0,
    },
}

SEP = "=" * 64


# ── RPM 별 디렉토리 구성 ──────────────────────────────────────────────────────

def setup_rpm_dirs(dry_run: bool = False) -> dict[str, Path]:
    """
    NORMAL_DIR 안의 파일을 파일명 패턴으로 분류해
    normal_3600rpm/, normal_1200rpm/ 디렉토리를 구성한다.
    Hard link 우선; 실패 시 copy 로 폴백.
    """
    result = {}
    for rpm_cond, pattern in RPM_PATTERNS.items():
        src_files = sorted(NORMAL_DIR.glob(pattern))
        dest_dir  = Path(f"data/raw/normal_{rpm_cond}")

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        linked, skipped = 0, 0
        for src in src_files:
            dst = dest_dir / src.name
            if dst.exists():
                skipped += 1
                continue
            if dry_run:
                linked += 1
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            linked += 1

        result[rpm_cond] = dest_dir
        print(f"  [{rpm_cond}] {len(src_files)} 파일 → {dest_dir}"
              f"  (신규 {linked}, 기존 {skipped})")
    return result


# ── 단일 생성 실행 ────────────────────────────────────────────────────────────

def run_gen(
    fault: str,
    rpm_cond: str,
    mode_name: str,
    cfg: dict,
    dry_run: bool = False,
) -> float:
    """main.py를 호출해 fault 데이터를 생성한다. 소요 시간(초)을 반환."""
    normal_dir = f"data/raw/normal_{rpm_cond}"
    out_dir    = f"data/synthetic/{rpm_cond}/{fault}"

    cmd = [
        str(PYTHON), "main.py",
        "--fault",        fault,
        "--severity-min", str(cfg["sev_min"]),
        "--severity-max", str(cfg["sev_max"]),
        "--count",        str(cfg["count"]),
        "--normal-dir",   normal_dir,
        "--output-dir",   out_dir,
    ]

    if cfg.get("transient"):
        cmd += [
            "--transient",
            "--active-cycles", str(cfg.get("active_cycles", 3.0)),
            "--silent-cycles", str(cfg.get("silent_cycles", 10.0)),
        ]

    label = f"{rpm_cond} / {fault} / {mode_name}"
    print(f"\n  ▶ {label}  (count={cfg['count']}, "
          f"sev=[{cfg['sev_min']},{cfg['sev_max']}])")

    if dry_run:
        print(f"    [DRY-RUN] {' '.join(cmd)}")
        return 0.0

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"    완료  {elapsed:.1f}s")
    return elapsed


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RCPVMS 합성 데이터 일괄 생성")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="실제 생성 없이 실행 계획만 출력",
    )
    ap.add_argument(
        "--rpm", choices=["3600rpm", "1200rpm", "both"], default="both",
        help="생성할 RPM 조건 (기본: both)",
    )
    ap.add_argument(
        "--mode", choices=["continuous", "transient", "all"], default="all",
        help="연속(continuous) / 과도(transient) / 전체(all)",
    )
    args = ap.parse_args()

    rpm_targets  = list(RPM_PATTERNS.keys()) if args.rpm == "both" else [args.rpm]
    mode_targets = list(GEN.keys())          if args.mode == "all"  else [args.mode]

    total_files = sum(
        GEN[m]["count"] * len(FAULT_TYPES) * len(rpm_targets)
        for m in mode_targets
    )

    print(SEP)
    print(" RCPVMS 합성 데이터 일괄 생성")
    print(f" RPM 조건 : {rpm_targets}")
    print(f" 생성 모드: {mode_targets}")
    print(f" 예상 파일: {total_files} 개")
    print(SEP)

    # 1. RPM 별 normal 디렉토리 구성
    print("\n[1] RPM 별 정상 데이터 디렉토리 구성")
    setup_rpm_dirs(dry_run=args.dry_run)

    # 2. 생성 실행
    print(f"\n[2] 합성 결함 데이터 생성")
    total_elapsed = 0.0
    completed     = 0

    for mode_name, cfg in GEN.items():
        if mode_name not in mode_targets:
            continue

        print(f"\n  {'─'*56}")
        print(f"  모드: {mode_name.upper()}"
              f"  (count={cfg['count']}, "
              f"sev=[{cfg['sev_min']},{cfg['sev_max']}]"
              f"{'  [Transient]' if cfg.get('transient') else ''})")

        for rpm_cond in rpm_targets:
            for fault in FAULT_TYPES:
                try:
                    elapsed = run_gen(
                        fault, rpm_cond, mode_name, cfg,
                        dry_run=args.dry_run,
                    )
                    total_elapsed += elapsed
                    completed     += cfg["count"]
                except subprocess.CalledProcessError as e:
                    print(f"    [오류] {rpm_cond}/{fault}/{mode_name}: {e}",
                          file=sys.stderr)

    # 3. 결과 요약
    print(f"\n{SEP}")
    print(f" 생성 완료: {completed} 파일  |  총 소요: {total_elapsed:.0f}s")

    if not args.dry_run:
        print("\n 출력 디렉토리별 파일 수:")
        for rpm_cond in rpm_targets:
            for fault in FAULT_TYPES:
                out = Path(f"data/synthetic/{rpm_cond}/{fault}")
                cnt = len(list(out.glob("*.bin"))) if out.exists() else 0
                print(f"  {out}:  {cnt} 파일")
    print(SEP)


if __name__ == "__main__":
    main()
