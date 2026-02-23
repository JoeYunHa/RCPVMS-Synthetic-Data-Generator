# ISO 20816-7 based fault parameters

"""
Fault Configuration Module based on ISO 20816-7 and Rotary Dynamics.
Defines frequency ratios and severity gains for synthetic fault generation.
"""

from enum import Enum
from dataclasses import dataclass


class Severity(Enum):
    """
    Severity levels inspired by ISO 20816-7 Zones.
    ZONE_A: Newly commissioned (Normal)
    ZONE_B: Acceptable for long-term operation
    ZONE_C: Warning (Action required)
    ZONE_D: Critical (Danger of damage)
    """

    NORMAL = 0.5  # Zone A/B
    WARNING = 1.5  # Zone C
    CRITICAL = 3.0  # Zone D


@dataclass
class FaultParam:
    """Parameters for defining a specific fault mode."""

    freq_ratio: float  # Multiplier of the fundamental frequency (1X)
    default_gain: float  # Standard gain for signal injection
    description: str


FAULT_MODELS = {
    "unbalance": FaultParam(
        freq_ratio=1.0,
        default_gain=1.0,
        description="Mass unbalance: Dominant 1X component.",
    ),
    "misalignment": FaultParam(
        freq_ratio=2.0,
        default_gain=0.7,
        description="Shaft misalignment: Dominant 2X component.",
    ),
    "oil_whip": FaultParam(
        freq_ratio=0.45,
        default_gain=1.2,
        description="Oil Whip: Sub-synchronous instability (0.43X-0.48X).",
    ),
}

