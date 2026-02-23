"""
RCPVMS Synthetic Fault Data Generator
======================================
Pipeline: Parse → Profile → Generate → Inject → Save

Usage:
    python main.py <input.bin> [--fault FAULT] [--severity SEVERITY] [--output PATH]

Examples:
    python main.py data/raw/normal.bin --fault unbalance --severity WARNING
    python main.py data/raw/normal.bin --fault oil_whip --severity CRITICAL --output out.bin
"""

import sys
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.core.rcpvms_parser import RCPVMSParser
from src.core.generator import FaultGenerator, TransientConfig
from src.core.synthesizer import RCPVMSSynthesizer
from src.models.fault_configs import FAULT_MODELS, Severity

# ── Constants ─────────────────────────────────────────────────────────────────

NORMAL_DATA_DIR = Path("data/raw/normal")  # Default directory for normal-state BIN files

RPM_JITTER_HZ = 0.05         # ±0.05 Hz steady-state RPM variation for RCPs
RPM_SEARCH_MIN_HZ = 20.0     # Lower bound for 1X frequency search (FFT)
RPM_SEARCH_MAX_HZ = 50.0     # Upper bound for 1X frequency search (FFT)
RPM_FALLBACK_HZ = 30.0       # Nominal fallback: 1800 RPM
ADC_FULL_SCALE_V = 10.0      # Standard ICP sensor ADC full-scale voltage (+-10 V)
SATURATION_FALLBACK = 4.0    # Fallback multiplier x base_rms when header sensitivity is absent


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_1x_frequency(signal: np.ndarray, fs: int) -> float:
    """
    Detects the dominant rotation frequency (1X) from a normal signal via FFT.
    Searches within [RPM_SEARCH_MIN_HZ, RPM_SEARCH_MAX_HZ] to avoid noise peaks.
    Falls back to RPM_FALLBACK_HZ if no peak is found in range.
    """
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)

    mask = (freqs >= RPM_SEARCH_MIN_HZ) & (freqs <= RPM_SEARCH_MAX_HZ)
    if not np.any(mask):
        return RPM_FALLBACK_HZ

    peak_idx = np.argmax(fft_mag[mask])
    return float(freqs[mask][peak_idx])


def compute_noise_floor(signal: np.ndarray, fs: int, rpm_hz: float) -> float:
    """
    Estimates the background noise standard deviation by removing 1X and 2X
    harmonic components from the signal and measuring the residual spread.
    """
    n = len(signal)
    fft_complex = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    freq_res = fs / n  # Hz per FFT bin

    for harmonic in [1.0, 2.0]:
        center = harmonic * rpm_hz
        mask = np.abs(freqs - center) <= 2 * freq_res
        fft_complex[mask] = 0.0

    residual = np.fft.irfft(fft_complex, n=n)
    return float(np.std(residual))


def compute_clip_range(extra: dict, base_rms: float) -> tuple[float, float]:
    """
    Derives physical saturation limits from the header's sensor sensitivity.

    Uses g_per_v (accelerometer sensitivity in g/V) and the known ADC full-scale
    voltage (ADC_FULL_SCALE_V) to compute the measurable physical range.
    Falls back to SATURATION_FALLBACK * base_rms if the header value is unusable.
    """
    g_per_v = float(extra.get("g_per_v", 0.0))
    if g_per_v > 0.0:
        clip_max = ADC_FULL_SCALE_V * g_per_v
    else:
        clip_max = base_rms * SATURATION_FALLBACK
    return -clip_max, clip_max


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(
    input_path: str,
    fault_type: str,
    severity_value: float,
    output_path: str,
    transient_cfg: TransientConfig | None = None,
) -> None:
    fault_cfg = FAULT_MODELS[fault_type]

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    tqdm.write(f"[1/5] Parsing: {input_path}")
    parser = RCPVMSParser(input_path)

    fs = parser.header.sampling_rate
    total_ch = parser.header.total_ch
    extra = parser.header.extra_fields or {}
    data_start = int(extra.get("data_start_offset", 0))

    tqdm.write(f"      fs={fs} Hz  |  channels={total_ch}  |  data_offset=0x{data_start:X}")

    channels_raw = parser.read_all_channels()

    if not channels_raw or len(channels_raw[0]) == 0:
        raise RuntimeError("Failed to read channel data.")

    n_samples = len(channels_raw[0])
    tqdm.write(f"      samples/channel={n_samples}")

    # ── 2. Feature Profiling ──────────────────────────────────────────────────
    tqdm.write("[2/5] Profiling normal signal...")

    ref = channels_raw[0]  # X-axis as reference channel
    rpm_hz = extract_1x_frequency(ref, fs)
    base_rms = float(np.sqrt(np.mean(ref ** 2)))
    noise_sigma = compute_noise_floor(ref, fs, rpm_hz)

    tqdm.write(f"      1X={rpm_hz:.3f} Hz  |  RMS={base_rms:.5f}  |  noise σ={noise_sigma:.5f}")

    # ── 3. Fault Generation ───────────────────────────────────────────────────
    transient_info = (
        f"transient(active={transient_cfg.active_cycles}cyc, silent={transient_cfg.silent_cycles}cyc)"
        if transient_cfg is not None
        else "continuous"
    )
    tqdm.write(f"[3/5] Generating fault  →  type={fault_type}  |  severity={severity_value:.3f}  |  mode={transient_info}")

    generator = FaultGenerator(
        fs=fs,
        rpm_hz=rpm_hz,
        n_samples=n_samples,
        jitter_hz=RPM_JITTER_HZ,
    )

    fault_signals = []
    for ch_idx in range(total_ch):
        # Within each X/Y bearing-plane pair, offset the Y-channel (odd index)
        # by 90° so the injected component forms a circular orbit.
        # Channels beyond the first pair reuse the same 0°/90° pattern.
        phase = np.pi / 2 if (ch_idx % 2 == 1) else 0.0

        if fault_type == "unbalance":
            f_sig = generator.generate_unbalance(severity=severity_value, phase=phase, transient=transient_cfg)
        elif fault_type == "misalignment":
            f_sig = generator.generate_misalignment(severity=severity_value, phase=phase, transient=transient_cfg)
        elif fault_type == "oil_whip":
            f_sig = generator.generate_oil_whip(severity=severity_value, phase=phase, transient=transient_cfg)
        else:
            raise ValueError(f"Unknown fault type: {fault_type!r}")

        fault_signals.append(f_sig)

    # ── 4. Fault Injection ────────────────────────────────────────────────────
    tqdm.write("[4/5] Injecting fault...")

    synthesizer = RCPVMSSynthesizer(parser)
    clip_range = compute_clip_range(extra, base_rms)
    tqdm.write(f"      clip_range=[{clip_range[0]:.4f}, {clip_range[1]:.4f}]")

    synthetic_channels = []
    for real, fault in zip(channels_raw, fault_signals):
        # severity.value       → amplitude scale of the fault waveform (FaultGenerator)
        # fault_cfg.default_gain → mixing ratio of fault into real signal (inject_fault)
        synthetic = synthesizer.inject_fault(
            real_signal=real,
            fault_signal=fault,
            gain=fault_cfg.default_gain,
            clip_range=clip_range,
        )
        synthetic_channels.append(synthetic)

    # ── 5. Re-package BIN ─────────────────────────────────────────────────────
    tqdm.write(f"[5/5] Saving → {output_path}")

    with open(input_path, "rb") as f:
        header_bytes = f.read(data_start)

    synthesizer.save_as_bin(output_path, header_bytes, synthetic_channels)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="RCPVMS Synthetic Fault Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Path to a real, normal-state .bin file. "
            f"If omitted, a file is chosen at random from {NORMAL_DATA_DIR} each iteration."
        ),
    )
    arg_parser.add_argument(
        "--fault",
        choices=list(FAULT_MODELS.keys()),
        default="unbalance",
        help="Fault type to inject",
    )
    arg_parser.add_argument(
        "--severity",
        choices=[s.name for s in Severity],
        default="WARNING",
        help=(
            "ISO 20816-7 severity zone (NORMAL=0.5, WARNING=1.5, CRITICAL=3.0). "
            "Ignored if --severity-value is specified."
        ),
    )
    arg_parser.add_argument(
        "--severity-value",
        type=float,
        default=None,
        metavar="V",
        help=(
            "Fixed severity amplitude scale (any positive float). "
            "Overrides --severity. Incompatible with --severity-min/max."
        ),
    )
    arg_parser.add_argument(
        "--severity-min",
        type=float,
        default=None,
        metavar="V",
        help=(
            "Lower bound for random severity sampling per file (batch mode). "
            "Must be used with --severity-max. e.g. 0.2 for incipient faults."
        ),
    )
    arg_parser.add_argument(
        "--severity-max",
        type=float,
        default=None,
        metavar="V",
        help=(
            "Upper bound for random severity sampling per file (batch mode). "
            "Must be used with --severity-min. e.g. 3.0 for critical faults."
        ),
    )
    arg_parser.add_argument(
        "--count",
        type=int,
        default=100,
        metavar="N",
        help="Number of synthetic files to generate (batch mode). Use --count 1 for single output.",
    )
    arg_parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Auto-generated under data/synthetic/<fault>/ if omitted. "
            "Files are named <fault>_<severity>_<NNNN>.bin."
        ),
    )
    arg_parser.add_argument(
        "--transient",
        action="store_true",
        default=False,
        help=(
            "Enable intermittent transient fault mode. The fault appears and "
            "disappears at specific rotation cycles rather than persisting "
            "continuously, simulating early-stage defects."
        ),
    )
    arg_parser.add_argument(
        "--active-cycles",
        type=float,
        default=3.0,
        metavar="N",
        help="[transient] Rotation cycles per fault burst. (default: 3.0)",
    )
    arg_parser.add_argument(
        "--silent-cycles",
        type=float,
        default=10.0,
        metavar="N",
        help="[transient] Silent rotation cycles between bursts. (default: 10.0)",
    )

    args = arg_parser.parse_args()

    if args.count < 1:
        arg_parser.error("--count must be at least 1.")

    # Resolve candidate pool for random input selection
    if args.input is not None:
        candidates = [Path(args.input)]
    else:
        candidates = sorted(NORMAL_DATA_DIR.glob("*.bin"))
        if not candidates:
            arg_parser.error(f"No .bin files found in {NORMAL_DATA_DIR}.")

    # Resolve severity mode
    range_mode = args.severity_min is not None or args.severity_max is not None
    if range_mode:
        if args.severity_min is None or args.severity_max is None:
            arg_parser.error("--severity-min and --severity-max must be used together.")
        if args.severity_value is not None:
            arg_parser.error("--severity-value cannot be used with --severity-min/max.")
        if args.severity_min <= 0 or args.severity_max <= 0:
            arg_parser.error("--severity-min and --severity-max must be positive.")
        if args.severity_min >= args.severity_max:
            arg_parser.error("--severity-min must be less than --severity-max.")
        severity_range = (args.severity_min, args.severity_max)
        severity_label_fixed = f"{args.severity_min:.3f}-{args.severity_max:.3f}"
    elif args.severity_value is not None:
        if args.severity_value <= 0:
            arg_parser.error("--severity-value must be a positive float.")
        severity_fixed = args.severity_value
        severity_label_fixed = f"{severity_fixed:.3f}"
    else:
        severity_fixed = Severity[args.severity].value
        severity_label_fixed = args.severity.lower()

    transient_cfg = (
        TransientConfig(
            active_cycles=args.active_cycles,
            silent_cycles=args.silent_cycles,
        )
        if args.transient
        else None
    )

    transient_suffix = "_transient" if transient_cfg is not None else ""

    # Resolve output directory
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("data/synthetic") / args.fault
    out_dir.mkdir(parents=True, exist_ok=True)

    n_digits = len(str(args.count))
    mode_desc = f"range [{severity_label_fixed}]" if range_mode else severity_label_fixed
    print(f"Generating {args.count} × [{args.fault} | severity={mode_desc}{transient_suffix}] → {out_dir}/")

    failed = 0
    for i in tqdm(range(1, args.count + 1), unit="file"):
        # Sample severity per-file in range mode; use fixed value otherwise
        if range_mode:
            severity_value = random.uniform(*severity_range)
            severity_label = f"{severity_value:.3f}"
        else:
            severity_value = severity_fixed
            severity_label = severity_label_fixed

        input_path = str(random.choice(candidates))
        stem = f"{args.fault}_{severity_label}{transient_suffix}"
        output_path = str(out_dir / f"{stem}_{i:0{n_digits}d}.bin")

        try:
            run(
                input_path=input_path,
                fault_type=args.fault,
                severity_value=severity_value,
                output_path=output_path,
                transient_cfg=transient_cfg,
            )
        except Exception as e:
            tqdm.write(f"[Error] {output_path}: {e}", file=sys.stderr)
            failed += 1

    total = args.count
    print(f"\nDone: {total - failed}/{total} succeeded.")


if __name__ == "__main__":
    main()
