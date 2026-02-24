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
        self.omega_n = self.omega / float(p.freq_ratio)  # first critical speed [rad/s]

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

    # ── Transient Envelope ────────────────────────────────────────────────────

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
        """
        Misalignment: dominant 2X + residual 1X → figure-8 (banana) orbit.

        Physical model:
            Angular shaft misalignment creates periodic angular stiffness variation
            at 2X frequency. The shaft response combines:
              · Dominant 2X: from the periodic stiffness excitation
              · Residual 1X: coupling artifact (typically 15–35% of 2X amplitude)

            Jeffcott response at each harmonic:
                X(t) = A₂·cos(2ωt + phase − φ₂) + A₁·cos(ωt + phase − φ₁)
                Y(t) = A₂·sin(2ωt + phase − φ₂) + A₁·sin(ωt + phase − φ₁)

            → figure-8 orbit. The 2X Y-component oscillates twice per revolution,
              while the 1X residual sweeps once, creating the characteristic
              banana/figure-8 shape in the orbit plot.

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

        # Dominant 2X component
        H2 = self._magnification(omega_2x)
        phi2 = self._phase_lag(omega_2x)
        A2 = amplitude * H2

        # Residual 1X coupling component
        H1 = self._magnification(self.omega)
        phi1 = self._phase_lag(self.omega)
        A1 = amplitude * residual_1x_ratio * H1

        x = A2 * np.cos(omega_2x * self.t + phase - phi2) + A1 * np.cos(
            self.omega * self.t + phase - phi1
        )
        y = A2 * np.sin(omega_2x * self.t + phase - phi2) + A1 * np.sin(
            self.omega * self.t + phase - phi1
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
        """
        Oil Whip: sub-synchronous forward whirl at ≈0.45X → precessing ellipse.

        Physical model:
            Journal bearing oil film instability causes forward whirl at
            ω_whip ≈ 0.45 × ω_run (approximately half running speed).

            The Jeffcott magnification H(ω_whip) can become very large when
            ω_whip approaches ω_n (resonance lock-in), which is physically
            consistent with the severity of real oil whip events.

            X(t) = F₀·H(ω_w)·cos(ω_w·t + phase − φ_w)
            Y(t) = F₀·H(ω_w)·sin(ω_w·t + phase − φ_w)

            → The orbit precesses at ω_whip, creating an elliptical shape
              distinct from the 1X synchronous ellipse in the orbit plot.

        Args:
            amplitude:  Sub-synchronous forcing amplitude F₀.
            phase:      Initial phase offset (rad).
            freq_ratio: Whirl-to-running-speed ratio (0.43–0.48). Default 0.45.
            transient:  Optional intermittent burst envelope.

        Returns:
            (x_signal, y_signal): matched X-Y channel pair.
        """
        omega_whip = self.omega * freq_ratio

        H_w = self._magnification(omega_whip)
        phi_w = self._phase_lag(omega_whip)
        A_w = amplitude * H_w

        x = A_w * np.cos(omega_whip * self.t + phase - phi_w)
        y = A_w * np.sin(omega_whip * self.t + phase - phi_w)

        if transient is not None:
            env = self.generate_transient_envelope(transient)
            x = x * env
            y = y * env

        return x.astype(np.float32), y.astype(np.float32)
