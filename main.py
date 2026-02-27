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
from src.core.generator import JeffcottGenerator, JeffcottParams, TransientConfig
from src.core.synthesizer import RCPVMSSynthesizer
from src.models.fault_configs import FAULT_MODELS, Severity

# ── Constants ─────────────────────────────────────────────────────────────────

NORMAL_DATA_DIR = Path("data/raw/normal")  # Default directory for normal-state BIN files

RPM_JITTER_HZ = 0.05         # ±0.05 Hz steady-state RPM variation for RCPs
RPM_SEARCH_MIN_HZ = 20.0     # Lower bound for 1X frequency search (FFT)
RPM_SEARCH_MAX_HZ = 70.0     # Upper bound for 1X frequency search (FFT)
RPM_FALLBACK_HZ = 30.0       # Nominal fallback: 1800 RPM
ADC_FULL_SCALE_V = 10.0      # Standard ICP sensor ADC full-scale voltage (+-10 V)
SATURATION_FALLBACK = 4.0    # Fallback multiplier x base_rms when header sensitivity is absent

# ── Physical Parameter Diversity Ranges ───────────────────────────────────────
# All Jeffcott model parameters are sampled independently per file to produce
# diverse orbit shapes and spectral signatures across the synthetic dataset.
ZETA_RANGE               = (0.03, 0.08)  # Viscous damping ratio ζ
FREQ_RATIO_RANGE         = (0.55, 0.80)  # Running speed / critical speed ratio (r₀)
KAPPA_RANGE              = (0.70, 1.00)  # Bearing stiffness anisotropy (ωny/ωnx)
OIL_WHIP_TAU_RANGE       = (1.0,  5.0)  # Self-excitation amplitude growth τ [s]
OIL_WHIP_LOCKIN_RANGE    = (5.0, 20.0)  # Lock-in acquisition cycles

MISALIGN_1X_RATIO_RANGE   = (0.15, 0.35)  # Residual 1X / dominant 2X output amplitude ratio
OIL_WHIP_FREQ_RATIO_RANGE = (0.43, 0.48)  # Sub-synchronous whirl / running speed ratio

# Per-bearing-pair independent amplitude scale (simulates spatial fault propagation)
BEARING_AMP_SCALE_RANGE  = (0.50, 1.50)


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


def compute_jeffcott_forcing(
    signal: np.ndarray,
    fs: int,
    rpm_hz: float,
    params: JeffcottParams,
    fault_type: str = "unbalance",
) -> float:
    """
    Calibrates the Jeffcott forcing amplitude to the fault-specific frequency band.

    Strategy (per fault type):
        unbalance    → calibrate from 1X band, magnify at 1X
                       severity=1 ≡ fault 1X response = existing 1X amplitude
        misalignment → calibrate from 1X amplitude, magnify at 2X
                       severity=1 ≡ fault 2X response = existing 1X amplitude
                       → guarantees 2X dominates 1X for severity ≥ 1.5
        oil_whip     → calibrate from 1X amplitude, magnify at 0.45X
                       severity=1 ≡ fault 0.45X response = existing 1X amplitude
                       → guarantees 0.45X dominates 1X for severity ≥ 1.2

    This fault-specific calibration compensates for the H(r) difference between
    fault frequencies, preventing under-injection when the fault frequency's
    magnification factor is lower than 1X (e.g., 2X above critical speed).

    Falls back to signal RMS if the 1X FFT peak cannot be located.
    """
    n = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal)) * 2.0 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Reference amplitude: always the 1X peak of the REAL signal
    mask_1x = np.abs(freqs - rpm_hz) <= 2.0
    if not np.any(mask_1x):
        return float(np.sqrt(np.mean(signal**2)))
    A_1x = float(np.max(fft_mag[mask_1x]))

    omega = 2.0 * np.pi * rpm_hz
    omega_n = omega / params.freq_ratio

    # Select the magnification frequency based on fault type
    if fault_type == "misalignment":
        omega_exc = 2.0 * omega          # fault response is at 2X
    elif fault_type == "oil_whip":
        omega_exc = 0.45 * omega         # fault response is at 0.45X
    else:                                # unbalance
        omega_exc = omega                # fault response is at 1X

    r = omega_exc / omega_n
    H = 1.0 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * params.zeta * r) ** 2)

    if H < 1e-9 or A_1x < 1e-9:
        return float(np.sqrt(np.mean(signal**2)))

    # F₀ = A_1x / H(fault_freq)
    # → severity × F₀ × H(fault_freq) = severity × A_1x
    # → fault component at fault_freq equals (severity × A_1x)
    return A_1x / H


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

    # ── 3. Fault Generation (Jeffcott Rotor Model) ────────────────────────────
    transient_info = (
        f"transient(active={transient_cfg.active_cycles}cyc, silent={transient_cfg.silent_cycles}cyc)"
        if transient_cfg is not None
        else "continuous"
    )

    # Jeffcott calibration:
    #   1. Back-calculate physical forcing amplitude F₀ from the real signal's 1X peak
    #      (F₀ = A_1x / H(ω), where H(ω) is the Jeffcott magnification factor)
    #   2. Scale by severity_value so severity=1.0 means fault forcing = normal 1X forcing
    #   This bounds fault amplitudes within physical limits and prevents over-injection.
    # ── Stage 1: Per-file physical parameter randomization ────────────────────
    jeffcott_params = JeffcottParams(
        zeta=np.random.uniform(*ZETA_RANGE),
        freq_ratio=np.random.uniform(*FREQ_RATIO_RANGE),
        kappa=np.random.uniform(*KAPPA_RANGE),
        oil_whip_growth_tau=np.random.uniform(*OIL_WHIP_TAU_RANGE),
        oil_whip_lockin_cycles=np.random.uniform(*OIL_WHIP_LOCKIN_RANGE),
    )
    forcing_ref = compute_jeffcott_forcing(ref, fs, rpm_hz, jeffcott_params, fault_type)
    scaled_forcing = severity_value * forcing_ref

    tqdm.write(
        f"[3/5] Generating fault  →  type={fault_type}  |  severity={severity_value:.3f}"
        f"  |  forcing={scaled_forcing:.5f}  |  mode={transient_info}"
    )
    tqdm.write(
        f"      jeffcott: zeta={jeffcott_params.zeta:.3f}  r0={jeffcott_params.freq_ratio:.3f}"
        f"  kappa={jeffcott_params.kappa:.3f}  tau={jeffcott_params.oil_whip_growth_tau:.2f}s"
        f"  lockin={jeffcott_params.oil_whip_lockin_cycles:.1f}cyc"
    )

    generator = JeffcottGenerator(
        fs=fs,
        rpm_hz=rpm_hz,
        n_samples=n_samples,
        params=jeffcott_params,
        jitter_hz=RPM_JITTER_HZ,
    )

    # generate_*() returns (x_signal, y_signal): a physically-valid X-Y pair.
    # Phase lag is derived from system damping (not hard-coded), and the orbit
    # shape matches rotor dynamics theory (circle / figure-8 / precessing ellipse).
    # Random phase: each file gets a unique orbit orientation [0, 2π)
    phase = np.random.uniform(0.0, 2.0 * np.pi)
    tqdm.write(f"      phase={phase:.3f} rad  ({np.degrees(phase):.1f}°)")

    if fault_type == "unbalance":
        x_fault, y_fault = generator.generate_unbalance(
            amplitude=scaled_forcing, phase=phase, transient=transient_cfg
        )
    elif fault_type == "misalignment":
        # Stage 1: per-file residual 1X ratio randomization
        residual_1x_ratio = np.random.uniform(*MISALIGN_1X_RATIO_RANGE)
        tqdm.write(f"      residual_1x_ratio={residual_1x_ratio:.3f}")
        x_fault, y_fault = generator.generate_misalignment(
            amplitude=scaled_forcing, phase=phase,
            residual_1x_ratio=residual_1x_ratio,
            transient=transient_cfg,
        )
    elif fault_type == "oil_whip":
        # Stage 1: per-file sub-synchronous frequency ratio randomization
        whip_freq_ratio = np.random.uniform(*OIL_WHIP_FREQ_RATIO_RANGE)
        tqdm.write(f"      oil_whip_freq_ratio={whip_freq_ratio:.3f}")
        x_fault, y_fault = generator.generate_oil_whip(
            amplitude=scaled_forcing, phase=phase,
            freq_ratio=whip_freq_ratio,
            transient=transient_cfg,
        )
    else:
        raise ValueError(f"Unknown fault type: {fault_type!r}")

    # Stage 3: Per-bearing-pair amplitude modulation.
    # Only inject into actual bearing XY channel pairs (BEARING_PAIRS).
    # Non-bearing channels (e.g. ch 2-3, 6-9, 12-15) receive zero fault signal.
    _BEARING_PAIRS = ((0, 1), (4, 5), (10, 11), (16, 17))
    bearing_scales = np.random.uniform(*BEARING_AMP_SCALE_RANGE, size=len(_BEARING_PAIRS))
    tqdm.write(f"      bearing_scales(4 pairs): {np.round(bearing_scales, 3).tolist()}")

    # Initialise all channels with zero fault signal
    fault_signals = [np.zeros(len(channels_raw[ch]), dtype=np.float32)
                     for ch in range(total_ch)]
    # Inject only into valid bearing XY channels
    for pair_idx, (ch_x, ch_y) in enumerate(_BEARING_PAIRS):
        scale = bearing_scales[pair_idx]
        if ch_x < total_ch:
            fault_signals[ch_x] = x_fault * scale
        if ch_y < total_ch:
            fault_signals[ch_y] = y_fault * scale

    # ── 4. Fault Injection ────────────────────────────────────────────────────
    tqdm.write("[4/5] Injecting fault...")

    synthesizer = RCPVMSSynthesizer(parser)
    clip_range = compute_clip_range(extra, base_rms)
    tqdm.write(f"      clip_range=[{clip_range[0]:.4f}, {clip_range[1]:.4f}]")

    synthetic_channels = []
    for real, fault in zip(channels_raw, fault_signals):
        # gain=1.0: the Jeffcott calibration has already set the physically-correct
        # fault amplitude. No additional mixing gain is needed.
        synthetic = synthesizer.inject_fault(
            real_signal=real,
            fault_signal=fault,
            gain=1.0,
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
    arg_parser.add_argument(
        "--normal-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing normal-state BIN files to use as synthesis base. "
            "Overrides the default NORMAL_DATA_DIR. Useful for separating operating "
            "conditions (e.g. 3600 RPM vs 1200 RPM pools)."
        ),
    )
    arg_parser.add_argument(
        "--active-cycles-min",
        type=float,
        default=None,
        metavar="N",
        help=(
            "[transient] Min active cycles for per-file random sampling. "
            "Must be used with --active-cycles-max. Overrides --active-cycles."
        ),
    )
    arg_parser.add_argument(
        "--active-cycles-max",
        type=float,
        default=None,
        metavar="N",
        help="[transient] Max active cycles for per-file random sampling.",
    )
    arg_parser.add_argument(
        "--silent-cycles-min",
        type=float,
        default=None,
        metavar="N",
        help=(
            "[transient] Min silent cycles for per-file random sampling. "
            "Must be used with --silent-cycles-max. Overrides --silent-cycles."
        ),
    )
    arg_parser.add_argument(
        "--silent-cycles-max",
        type=float,
        default=None,
        metavar="N",
        help="[transient] Max silent cycles for per-file random sampling.",
    )

    args = arg_parser.parse_args()

    if args.count < 1:
        arg_parser.error("--count must be at least 1.")

    # Resolve candidate pool for random input selection
    if args.input is not None:
        candidates = [Path(args.input)]
    else:
        normal_dir = Path(args.normal_dir) if args.normal_dir else NORMAL_DATA_DIR
        candidates = sorted(normal_dir.glob("*.bin")) + sorted(normal_dir.glob("*.BIN"))
        candidates = sorted(set(candidates))
        if not candidates:
            arg_parser.error(f"No .bin/.BIN files found in {normal_dir}.")

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

    # Stage 2: detect transient range mode (per-file random sampling)
    _transient_range_mode = (
        args.transient
        and args.active_cycles_min is not None
        and args.active_cycles_max is not None
    )

    transient_suffix = "_transient" if args.transient else ""

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
        # In range mode, severity varies per file — omit it from the filename
        # so alphabetical sort in validate_synthetic.py does not accidentally
        # select only the lowest-severity files.
        if range_mode:
            stem = f"{args.fault}{transient_suffix}"
        else:
            stem = f"{args.fault}_{severity_label}{transient_suffix}"
        output_path = str(out_dir / f"{stem}_{i:0{n_digits}d}.bin")

        # Stage 2: create per-file TransientConfig (fixed or range-sampled)
        if args.transient:
            if _transient_range_mode:
                _act = random.uniform(args.active_cycles_min, args.active_cycles_max)
                _sil_min = args.silent_cycles_min if args.silent_cycles_min is not None else args.silent_cycles
                _sil_max = args.silent_cycles_max if args.silent_cycles_max is not None else args.silent_cycles
                _sil = random.uniform(_sil_min, _sil_max)
                transient_cfg = TransientConfig(active_cycles=_act, silent_cycles=_sil)
            else:
                transient_cfg = TransientConfig(
                    active_cycles=args.active_cycles,
                    silent_cycles=args.silent_cycles,
                )
        else:
            transient_cfg = None

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
