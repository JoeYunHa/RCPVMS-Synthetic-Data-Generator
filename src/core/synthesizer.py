# integrates the generated fault into the real normal signal and packages it back into a BIN file while preserving the header

import os
import sys
from typing import Union
import numpy as np


class RCPVMSSynthesizer:
    """
    Injects fault signals into real data and reconstructs the BIN file.
    """

    def __init__(self, parser):
        self.parser = parser

    def inject_fault(
        self,
        real_signal: np.ndarray,
        fault_signal: np.ndarray,
        gain: float = 0.5,
        clip_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """
        Linear injection: V_synthetic = V_real + (gain * V_fault)

        Args:
            clip_range: Optional (min, max) tuple to clamp the result within
                        the sensor's physical measurement limits and prevent saturation.
        """
        if real_signal.shape != fault_signal.shape:
            raise ValueError("Signal dimensions do not match.")

        result = (real_signal + (gain * fault_signal)).astype(np.float32)

        if clip_range is not None:
            result = np.clip(result, clip_range[0], clip_range[1])

        return result

    def save_as_bin(
        self,
        output_path: Union[str, os.PathLike],
        header_bytes: bytes,
        channels_data: list[np.ndarray],
    ):
        """
        Saves data in non-interleaved block format: [Header][CH0 Block][CH1 Block]...
        """
        try:
            with open(output_path, "wb") as f:
                # 1. Write the preserved original header
                f.write(header_bytes)

                # 2. Append each channel block (float32, little-endian)
                for channel_data in channels_data:
                    # Convert to little-endian float32 specifically (<f4)
                    f.write(channel_data.astype("<f4").tobytes())

            print(f"[Success] Synthetic BIN saved to: {output_path}")
        except OSError as e:
            print(f"[Error] Failed to save BIN: {e}", file=sys.stderr)
            raise
