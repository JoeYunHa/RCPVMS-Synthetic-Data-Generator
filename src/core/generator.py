"""
Jeffcott Rotor-Based Vibration Fault Generator
===============================================
Physically-correct synthesis using 2-DOF Jeffcott rotor steady-state response.

Previous approach (sinusoidal injection) problems fixed:
  1. X-Y phase was hard-coded to π/2 regardless of system dynamics
     → orbit phase lag now derived from damping ratio ζ via arctan2(2ζr, 1-r²)
  2. Amplitude was scaled by RMS (not a physical quantity for orbit size)
     → amplitude governed by magnification factor H(r) = 1/√((1-r²)²+(2ζr)²)
  3. Y channel received arbitrary +π/2 offset producing circular orbits always
     → each fault type now produces its physically-correct orbit shape:
       · Unbalance  → circle
       · Misalignment → figure-8 (banana)
       · Oil Whip  → precessing ellipse

Reference: Jeffcott (1919), Bently & Hatch "Fundamentals of Rotating Machinery"
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TransientConfig:
    """
    Parameters for intermittent (transient) fault envelope generation.

    Models early-stage faults that appear and disappear at specific rotation
    cycles rather than persisting continuously throughout the signal.

    The envelope pattern per period:

        Rotation cycles:  0        active_cycles      active_cycles + silent_cycles
                          |-- burst (Hanning) --|------- silence (0) --------|

    Example (30 Hz rotation, active_cycles=3, silent_cycles=10):
        Burst duration  ≈  3 × (1/30 s) ≈  100 ms
        Silence         ≈ 10 × (1/30 s) ≈  333 ms
        Repeat period   ≈ 13 × (1/30 s) ≈  433 ms

    Attributes:
        active_cycles:  Number of rotation cycles the fault is active per burst.
        silent_cycles:  Number of silent rotation cycles between bursts.
    """

    active_cycles: float = 3.0
    silent_cycles: float = 10.0


@dataclass
class JeffcottParams:
    """
    Physical parameters for the 2-DOF Jeffcott rotor model.

    The dimensionless formulation avoids unit assumptions and allows calibration
    purely from real signal statistics (no mass/stiffness in physical units needed).

    Attributes:
        zeta:        Viscous damping ratio (ζ).
                     RCP journal bearings typically: 0.03 – 0.08.
                     Higher ζ → flatter frequency response, less resonance peak.

        freq_ratio:  Running speed / first critical speed  (r₀ = ω_run / ω_n).
                     Most RCPs operate sub-critically: r₀ ≈ 0.5 – 0.80.
                     ω_n is derived internally as ω_run / freq_ratio.
    """

    zeta: float = 0.05
    freq_ratio: float = 0.70

    # Bearing stiffness anisotropy: ωny / ωnx  (1.0 = isotropic, <1 = Y softer)
    # Typical range for asymmetric journal bearings: 0.70 – 0.90.
    # Drives different H(r) and φ(r) in X vs Y → elliptical / tilted orbit.
    kappa: float = 1.0

    # Self-excitation amplitude growth time-constant [s].
    # 0.0 = instant steady-state (legacy behaviour).
    # Finite τ → A(t) = A_max × (1 − e^(−t/τ))  replicates positive-feedback
    # energy injection from the oil-film.
    oil_whip_growth_tau: float = 2.0

    # Number of sub-synchronous whirl cycles over which instantaneous frequency
    # chirps from 0.88×freq_ratio×Ω to the locked value.  0 = pre-locked.
    oil_whip_lockin_cycles: float = 10.0


class JeffcottGenerator:
    """
    Generates physically-valid vibration fault signals using 2-DOF Jeffcott rotor theory.

    Steady-state solution for harmonic excitation F = F₀·e^(jΩt):

        H(r)  = 1 / sqrt((1 - r²)² + (2ζr)²)     [amplitude magnification]
        φ(r)  = arctan2(2ζr, 1 - r²)              [phase lag, rad]

    where r = Ω_excitation / ω_natural.

    Each generate_*() method returns (x_signal, y_signal) — a physically-valid
    X-Y displacement pair whose orbit satisfies rotor dynamics constraints:

        · generate_unbalance()     → circular orbit  (1X)
        · generate_misalignment()  → figure-8 orbit  (2X dominant + 1X residual)
        · generate_oil_whip()      → precessing ellipse (sub-synchronous ~0.45X)
    """

    def __init__(
        self,
        fs: int,
        rpm_hz: float,
        n_samples: int,
        params: "JeffcottParams | None" = None,
        jitter_hz: float = 0.0,
    ):
        """
        Args:
            fs:         Sampling frequency (Hz).
            rpm_hz:     Nominal rotation frequency (Hz), e.g. 30.0 for 1800 RPM.
            n_samples:  Number of samples to generate.
            params:     JeffcottParams. Uses defaults if None.
            jitter_hz:  Max RPM jitter magnitude (Hz). A random offset in
                        [-jitter_hz, +jitter_hz] is applied to rpm_hz to simulate
                        realistic steady-state speed variation. Default 0 (no jitter).
        """
        self.fs = fs
        self.rpm_hz = (
            rpm_hz + np.random.uniform(-jitter_hz, jitter_hz)
            if jitter_hz > 0.0
            else rpm_hz
        )
        self.omega = 2.0 * np.pi * self.rpm_hz  # running speed [rad/s]
        self.n_samples = n_samples
        self.t = np.arange(n_samples, dtype=np.float64) / fs

        p = params if params is not None else JeffcottParams()
        self.zeta = float(p.zeta)
        self.omega_n = self.omega / float(p.freq_ratio)   # X-axis critical speed [rad/s]
        self.omega_ny = self.omega_n * float(p.kappa)     # Y-axis critical speed [rad/s]
        self.kappa = float(p.kappa)
        self.oil_whip_growth_tau = float(p.oil_whip_growth_tau)
        self.oil_whip_lockin_cycles = float(p.oil_whip_lockin_cycles)

    # ── Jeffcott Transfer Functions ───────────────────────────────────────────

    def _magnification(self, omega_exc: float) -> float:
        """
        Steady-state amplitude magnification H(r) at excitation frequency omega_exc.

        Output displacement amplitude = (forcing amplitude) × H(r).
        Below resonance (r < 1): H > 1 and grows toward resonance.
        Above resonance (r > 1): H < 1 and decays toward 0.
        """
        r = omega_exc / self.omega_n
        return 1.0 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * self.zeta * r) ** 2)

    def _phase_lag(self, omega_exc: float) -> float:
        """
        Phase lag φ(r) of displacement response relative to forcing [rad].

        Sub-critical (r < 1): φ ≈ 0  (response nearly in phase with forcing)
        At resonance  (r = 1): φ = π/2
        Super-critical (r > 1): φ → π  (response nearly anti-phase)
        """
        r = omega_exc / self.omega_n
        return np.arctan2(2.0 * self.zeta * r, 1.0 - r**2)

    # ── Y-axis Transfer Functions (anisotropic stiffness) ────────────────────

    def _mag_y(self, omega_exc: float) -> float:
        """Amplitude magnification H(r) for the Y axis (uses omega_ny).

        Identical to _magnification() when kappa == 1.0 (isotropic).
        When kappa < 1.0, omega_ny < omega_n, so the Y-axis resonance peak
        sits at a lower excitation frequency than the X-axis peak.
        """
        r = omega_exc / self.omega_ny
        return 1.0 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * self.zeta * r) ** 2)

    def _phi_y(self, omega_exc: float) -> float:
        """Phase lag φ(r) for the Y axis (uses omega_ny)."""
        r = omega_exc / self.omega_ny
        return np.arctan2(2.0 * self.zeta * r, 1.0 - r**2)

    # ── Transient Envelope ───────────────────────────────────────────────────

    def generate_transient_envelope(self, config: TransientConfig) -> np.ndarray:
        """
        Creates a periodic burst envelope synchronized to the rotation frequency.

        Each burst is shaped by a Hanning window (smooth rise-and-fall).
        Pattern repeats: |<-- active_cycles burst -->|<-- silent_cycles silence -->|

        Returns:
            Float32 envelope array of shape (n_samples,), values in [0, 1].
        """
        samples_per_cycle = self.fs / self.rpm_hz
        burst_samples = max(2, int(round(config.active_cycles * samples_per_cycle)))
        silence_samples = max(0, int(round(config.silent_cycles * samples_per_cycle)))
        period_samples = burst_samples + silence_samples

        envelope = np.zeros(self.n_samples, dtype=np.float32)
        hann = np.hanning(burst_samples).astype(np.float32)

        pos = 0
        while pos < self.n_samples:
            end = min(pos + burst_samples, self.n_samples)
            envelope[pos:end] = hann[: end - pos]
            pos += period_samples

        return envelope

    # ── Fault Signal Generators ───────────────────────────────────────────────

    def generate_unbalance(
        self,
        amplitude: float,
        phase: float = 0.0,
        transient: "TransientConfig | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Unbalance: synchronous 1X → circular orbit.

        Physical model:
            Mass unbalance (mu at eccentricity e) creates centrifugal forcing:
                F_x = F₀·cos(ωt + phase)
                F_y = F₀·sin(ωt + phase)

            Jeffcott steady-state displacement:
                X(t) = F₀·H(ω)·cos(ωt + phase − φ)
                Y(t) = F₀·H(ω)·sin(ωt + phase − φ)

            → Circular orbit of radius A = F₀·H(ω) in X-Y plane.
              Phase lag φ is determined by damping (not hard-coded).

        Args:
            amplitude: Calibrated forcing amplitude F₀ (real ADC units).
                       Derive via compute_jeffcott_forcing() in main.py.
            phase:     Initial phase offset (rad).
            transient: Optional intermittent burst envelope.

        Returns:
            (x_signal, y_signal): matched X-Y channel pair, shape (n_samples,).
        """
        H = self._magnification(self.omega)
        phi = self._phase_lag(self.omega)
        A = amplitude * H

        x = A * np.cos(self.omega * self.t + phase - phi)
        y = A * np.sin(self.omega * self.t + phase - phi)

        if transient is not None:
            env = self.generate_transient_envelope(transient)
            x = x * env
            y = y * env

        return x.astype(np.float32), y.astype(np.float32)

    def generate_misalignment(
        self,
        amplitude: float,
        phase: float = 0.0,
        residual_1x_ratio: float = 0.25,
        transient: "TransientConfig | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Misalignment: dominant 2X + residual 1X → asymmetric figure-8 orbit.

        Physical model (with directional stiffness):
            Angular shaft misalignment creates a periodic angular stiffness variation
            at 2X frequency.  Anisotropic bearing stiffness (kappa = ωny/ωnx ≠ 1)
            gives X and Y axes different natural frequencies and therefore different
            transfer functions H(r) and φ(r) at the same excitation frequency.

            This breaks the orbit symmetry: instead of a geometrically symmetric
            figure-8, the orbit becomes a tilted or elongated banana shape, matching
            real RCP misalignment signatures.

            Dominant 2X:
                X(t) = Ax2·cos(2ωt + phase − φx2)   [uses ωnx]
                Y(t) = Ay2·sin(2ωt + phase − φy2)   [uses ωny ≠ ωnx when kappa≠1]

            Residual 1X coupling (typically 15–35% of 2X amplitude):
                X(t) += Ax1·cos(ωt + phase − φx1)
                Y(t) += Ay1·sin(ωt + phase − φy1)

            With kappa=1 (isotropic) the original symmetric figure-8 is recovered.

        Args:
            amplitude:         Primary forcing amplitude at 2X.
            phase:             Phase offset of 2X component (rad).
            residual_1x_ratio: Ratio of 1X residual amplitude to 2X amplitude.
                               Default 0.25 (typical range 0.15–0.35).
            transient:         Optional intermittent burst envelope.

        Returns:
            (x_signal, y_signal): matched X-Y channel pair.
        """
        omega_2x = 2.0 * self.omega

        # ── Dominant 2X: separate H and φ in X vs Y (directional stiffness) ──
        Hx2   = self._magnification(omega_2x)
        phix2 = self._phase_lag(omega_2x)
        Hy2   = self._mag_y(omega_2x)       # differs from Hx2 when kappa ≠ 1
        phiy2 = self._phi_y(omega_2x)

        Ax2 = amplitude * Hx2
        Ay2 = amplitude * Hy2

        # ── Residual 1X coupling: defined as output amplitude ratio ────────────
        # residual_1x_ratio = fraction of the OBSERVABLE 2X output amplitude.
        # Using forcing-ratio (× Hx1/Hy1) would amplify residual near Y resonance
        # when kappa < 1 (ωny close to ω), making 1X dominate over 2X in Y channels.
        # Output-ratio definition is physically correct: real misalignment shows
        # 15–35% of the 2X displacement amplitude as residual 1X.
        phix1 = self._phase_lag(self.omega)
        phiy1 = self._phi_y(self.omega)

        Ax1 = Ax2 * residual_1x_ratio
        Ay1 = Ay2 * residual_1x_ratio

        x = (
            Ax2 * np.cos(omega_2x * self.t + phase - phix2)
            + Ax1 * np.cos(self.omega * self.t + phase - phix1)
        )
        y = (
            Ay2 * np.sin(omega_2x * self.t + phase - phiy2)
            + Ay1 * np.sin(self.omega * self.t + phase - phiy1)
        )

        if transient is not None:
            env = self.generate_transient_envelope(transient)
            x = x * env
            y = y * env

        return x.astype(np.float32), y.astype(np.float32)

    def generate_oil_whip(
        self,
        amplitude: float,
        phase: float = 0.0,
        freq_ratio: float = 0.45,
        transient: "TransientConfig | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oil Whip: sub-synchronous forward whirl with self-excitation and lock-in.

        Physical model (enhanced):
            Real oil whip exhibits two characteristics absent in a pure steady-state
            Jeffcott model:

            1. Lock-in (주파수 고착):
               Oil *whirl* tracks the running speed as ≈ freq_ratio × Ω.  When the
               threshold condition is met, the sub-synchronous frequency abruptly
               "snaps" to a fixed value ω_locked = freq_ratio × Ω and no longer
               tracks speed changes.  The snap is modelled as a linear frequency
               chirp from (0.88 × freq_ratio × Ω) → ω_locked over the first
               `oil_whip_lockin_cycles` sub-synchronous cycles, after which phase
               accumulates at constant ω_locked.

            2. Self-excitation / amplitude growth (자려진동):
               The oil-film provides net positive energy to the rotor.  Amplitude
               builds up exponentially:

                   A(t) = A_max × (1 − e^(−t / τ))

               where τ = oil_whip_growth_tau.  τ=0 reverts to instant steady-state.

            Full time-domain model:
                φ(t) = ∫ ω_inst(t') dt' + phase₀   [chirp → constant]
                X(t) = A_max · (1−e^{−t/τ}) · cos(φ(t) − φ_w)
                Y(t) = A_max · (1−e^{−t/τ}) · sin(φ(t) − φ_w)

        Args:
            amplitude:  Sub-synchronous forcing amplitude F₀ (calibrated by main.py).
            phase:      Initial phase offset (rad).
            freq_ratio: Locked whirl-to-running-speed ratio (0.43–0.48). Default 0.45.
            transient:  Optional intermittent burst envelope.

        Returns:
            (x_signal, y_signal): matched X-Y channel pair.
        """
        omega_locked = self.omega * freq_ratio

        # ── Phase accumulation: chirp during lock-in acquisition ──────────────
        if self.oil_whip_lockin_cycles > 0.0 and omega_locked > 0.0:
            # How many seconds the chirp lasts
            T_cycle_locked = (2.0 * np.pi) / omega_locked   # one sub-sync period
            T_lockin = self.oil_whip_lockin_cycles * T_cycle_locked
            n_lockin = min(int(T_lockin * self.fs), self.n_samples)

            omega_inst = np.empty(self.n_samples, dtype=np.float64)
            if n_lockin > 0:
                # Frequency ramps from 88 % of locked value up to locked value
                omega_inst[:n_lockin] = np.linspace(
                    omega_locked * 0.88, omega_locked, n_lockin
                )
            omega_inst[n_lockin:] = omega_locked

            # Integrate instantaneous frequency → unwrapped phase
            phase_arr = np.cumsum(omega_inst) / self.fs + phase
        else:
            # Already locked: constant sub-synchronous frequency
            phase_arr = omega_locked * self.t + phase

        # ── Jeffcott transfer function at locked frequency ────────────────────
        H_w   = self._magnification(omega_locked)
        phi_w = self._phase_lag(omega_locked)
        A_w   = amplitude * H_w

        # ── Self-excitation growth envelope ───────────────────────────────────
        if self.oil_whip_growth_tau > 0.0:
            growth = 1.0 - np.exp(-self.t / self.oil_whip_growth_tau)
        else:
            growth = np.ones(self.n_samples, dtype=np.float64)

        x = A_w * growth * np.cos(phase_arr - phi_w)
        y = A_w * growth * np.sin(phase_arr - phi_w)

        if transient is not None:
            env = self.generate_transient_envelope(transient)
            x = x * env
            y = y * env

        return x.astype(np.float32), y.astype(np.float32)
