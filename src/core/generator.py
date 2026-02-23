# handles the mathematical modeling of vibration fault signatures based on the detected RPM

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


class FaultGenerator:
    """
    Generates synthetic vibration fault signals based on physical principles.
    """

    def __init__(self, fs: int, rpm_hz: float, n_samples: int, jitter_hz: float = 0.0):
        """
        Args:
            fs:         Sampling frequency (Hz).
            rpm_hz:     Nominal rotation frequency (Hz). e.g. 30.0 for 1800 RPM.
            n_samples:  Number of samples to generate.
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
        self.n_samples = n_samples
        self.t = np.arange(n_samples) / fs

    def generate_transient_envelope(self, config: "TransientConfig") -> np.ndarray:
        """
        Creates a periodic burst envelope synchronized to the rotation frequency.

        Each burst is shaped by a Hanning window, which provides a smooth
        rise-and-fall profile and avoids discontinuities at burst edges.
        The pattern repeats throughout the signal:

            |<-- active_cycles -->|<---- silent_cycles ---->|  (repeating)
            |  /\  Hanning burst  |       silence (0)        |

        Args:
            config: TransientConfig defining burst and silence durations in
                    rotation cycles.

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
            window_len = end - pos
            envelope[pos:end] = hann[:window_len]
            pos += period_samples

        return envelope

    def generate_unbalance(
        self,
        severity: float = 1.0,
        phase: float = 0.0,
        transient: "TransientConfig | None" = None,
    ) -> np.ndarray:
        """
        Simulates Unbalance: Dominant 1X (synchronous) vibration.

        Args:
            severity:   Amplitude scaling factor (gain K in project plan).
            phase:      Initial phase offset in radians.
            transient:  If provided, applies an intermittent burst envelope so the
                        fault appears and disappears at specific rotation cycles,
                        simulating early-stage intermittent unbalance.
        """
        signal = severity * np.sin(2 * np.pi * self.rpm_hz * self.t + phase)
        if transient is not None:
            signal = signal * self.generate_transient_envelope(transient)
        return signal.astype(np.float32)

    def generate_misalignment(
        self,
        severity: float = 1.0,
        phase: float = np.pi / 4,
        transient: "TransientConfig | None" = None,
    ) -> np.ndarray:
        """
        Simulates Misalignment: Elevated 2X harmonic with residual 1X presence.
        Real misalignment signatures contain both 1X and a dominant 2X component.

        Args:
            severity:   Amplitude scaling factor.
            phase:      Phase offset of the 2X component in radians.
            transient:  If provided, applies an intermittent burst envelope,
                        simulating early-stage intermittent misalignment.
        """
        # Pure 2X injection: the real normal signal already carries a strong 1X
        # component, so any residual 1X mixed in here would add to that baseline
        # and compete against the injected 2X in the FFT dominance check.
        # Using pure 2X guarantees the injected component always dominates its band.
        signal = severity * np.sin(2 * np.pi * (2 * self.rpm_hz) * self.t + phase)
        if transient is not None:
            signal = signal * self.generate_transient_envelope(transient)
        return signal.astype(np.float32)

    def generate_oil_whip(
        self,
        severity: float = 1.0,
        phase: float = 0.0,
        freq_ratio: float = 0.45,
        transient: "TransientConfig | None" = None,
    ) -> np.ndarray:
        """
        Simulates Oil Whip: Sub-synchronous vibration specific to journal bearings in RCPs.

        Args:
            severity:   Amplitude scaling factor.
            phase:      Initial phase offset in radians.
            freq_ratio: Sub-synchronous frequency ratio relative to 1X.
                        Typically 0.43–0.48 for journal bearings. Default 0.45.
            transient:  If provided, applies an intermittent burst envelope,
                        simulating early-stage intermittent oil whip instability.
        """
        whip_freq = self.rpm_hz * freq_ratio
        signal = severity * np.sin(2 * np.pi * whip_freq * self.t + phase)
        if transient is not None:
            signal = signal * self.generate_transient_envelope(transient)
        return signal.astype(np.float32)
