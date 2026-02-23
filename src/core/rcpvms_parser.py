"""
RCPVMS Parser - Binary Spec (R17 Verified)
Compatible with base_parser.py EventHeader structure.
"""

import math
import sys
import struct
import numpy as np
from .base_parser import BaseEventParser, EventHeader


class RCPVMSParser(BaseEventParser):
    SYSTEM_ID = 2
    SYSTEM_NAME = "RCPVMS"

    MIN_HEADER_SIZE = 0x48
    DATA_SCAN_WINDOW_FLOATS = 512
    DATA_SCAN_STEP = 256
    DATA_SCAN_LIMIT = 1024 * 1024

    def _parse_header_impl(self) -> EventHeader:
        if self.file_size < self.MIN_HEADER_SIZE:
            raise ValueError(
                f"RCPVMS file too small for header: {self.file_size} bytes"
            )

        read_size = min(self.file_size, 1024)
        with open(self.filepath, "rb") as f:
            header_chunk = f.read(read_size)

        site_id = self.decode_string(header_chunk[0x00:0x08])

        sys_id, event_ch, total_ch, event_type = struct.unpack(
            "<HHHH", header_chunk[0x08:0x10]
        )

        event_date_str = self.decode_string(header_chunk[0x10:0x28])

        alarm_result = struct.unpack("<H", header_chunk[0x28:0x2A])[0]

        file_version = self.decode_string(header_chunk[0x2C:0x30])
        if not file_version:
            file_version = "1.00"

        sampling_rate = struct.unpack("<I", header_chunk[0x30:0x34])[0]
        user_id = struct.unpack("<I", header_chunk[0x34:0x38])[0]
        event_duration_ms = struct.unpack("<I", header_chunk[0x38:0x3C])[0]

        signal_type = struct.unpack("<H", header_chunk[0x3C:0x3E])[0]

        g_per_v = struct.unpack("<f", header_chunk[0x40:0x44])[0]
        mils_per_v = struct.unpack("<f", header_chunk[0x44:0x48])[0]

        if not math.isfinite(g_per_v) or g_per_v <= 0:
            g_per_v = 1.0
        if not math.isfinite(mils_per_v) or mils_per_v <= 0:
            mils_per_v = 1.0

        data_start_offset, samples_per_channel = self._compute_data_start_offset(
            sampling_rate, event_duration_ms, total_ch
        )

        extra_fields = {
            "data_start_offset": int(data_start_offset),
            "computed_samples_per_channel": int(samples_per_channel),
            "sampling_rate_raw": int(sampling_rate),
            "event_duration_ms_raw": int(event_duration_ms),
            "signal_type": int(signal_type),
            "user_id": int(user_id),
            "g_per_v": float(g_per_v),
            "mils_per_v": float(mils_per_v),
        }

        header = EventHeader(
            site_id=site_id,
            system_id=sys_id if sys_id != 0 else self.SYSTEM_ID,
            system_name=self.SYSTEM_NAME,
            event_ch=event_ch,
            total_ch=total_ch,
            event_type=event_type,
            event_date=event_date_str,
            alarm_result=alarm_result,
            file_version=file_version,
            sampling_rate=int(sampling_rate),
            event_duration_ms=int(event_duration_ms),
            num_passes=0,
            sensitivity=float(g_per_v),
            extra_fields=extra_fields,
        )

        return header

    def _compute_data_start_offset(
        self, sampling_rate: int, duration_ms: int, total_ch: int
    ) -> tuple[int, int]:
        n_expected = 0
        if sampling_rate > 0 and duration_ms > 0:
            n_expected = int(round(sampling_rate * (duration_ms / 1000.0)))

        if total_ch > 0 and n_expected > 0:
            raw_bytes_total = total_ch * n_expected * 4
            candidate_offset = self.file_size - raw_bytes_total
            if self._validate_data_offset(candidate_offset):
                return candidate_offset, n_expected

        fallback_offset = self._scan_for_data_offset()
        if fallback_offset is None:
            fallback_offset = 0

        n_from_file = 0
        if total_ch > 0 and fallback_offset < self.file_size:
            bytes_per_ch = (self.file_size - fallback_offset) // total_ch
            n_from_file = bytes_per_ch // 4

        return fallback_offset, n_from_file

    def _validate_data_offset(self, offset: int) -> bool:
        if offset < 0 or offset >= self.file_size:
            return False
        if offset % 4 != 0:
            return False

        window_bytes = self.DATA_SCAN_WINDOW_FLOATS * 4
        if offset + window_bytes > self.file_size:
            return False

        with open(self.filepath, "rb") as f:
            f.seek(offset)
            buf = f.read(window_bytes)
        if len(buf) < 64 * 4:
            return False

        data = np.frombuffer(buf, dtype="<f4")
        if data.size < 64:
            return False

        finite_ratio = float(np.mean(np.isfinite(data)))
        return finite_ratio > 0.95

    def _scan_for_data_offset(self) -> int | None:
        window_bytes = self.DATA_SCAN_WINDOW_FLOATS * 4
        scan_end = min(self.file_size - window_bytes, self.DATA_SCAN_LIMIT)
        if scan_end <= 0:
            return None

        with open(self.filepath, "rb") as f:
            for offset in range(0, scan_end, self.DATA_SCAN_STEP):
                if offset % 4 != 0:
                    continue
                if self._looks_like_waveform(f, offset):
                    return offset

        return None

    def _looks_like_waveform(self, f, offset: int) -> bool:
        max_bytes = self.file_size - offset
        to_read = min(self.DATA_SCAN_WINDOW_FLOATS * 4, max_bytes)
        if to_read < 64 * 4:
            return False

        f.seek(offset)
        buf = f.read(to_read)
        if len(buf) < 64 * 4:
            return False

        data = np.frombuffer(buf, dtype="<f4")
        finite_mask = np.isfinite(data)
        finite_ratio = float(np.mean(finite_mask))
        if finite_ratio < 0.9:
            return False

        finite_vals = data[finite_mask]
        if finite_vals.size < 32:
            return False

        v_min = float(np.nanmin(finite_vals))
        v_max = float(np.nanmax(finite_vals))
        if not math.isfinite(v_min) or not math.isfinite(v_max):
            return False
        if (v_max - v_min) < 1e-6:
            return False

        return True

    def read_channel(self, channel_index: int) -> np.ndarray:
        if not self.header or self.header.total_ch <= 0:
            return np.array([], dtype=np.float32)

        if channel_index < 0 or channel_index >= self.header.total_ch:
            return np.array([], dtype=np.float32)

        data_start_offset = int((self.header.extra_fields or {}).get("data_start_offset", 0))
        if data_start_offset < 0 or data_start_offset >= self.file_size:
            return np.array([], dtype=np.float32)

        sampling_rate = self.header.sampling_rate
        duration_ms = self.header.event_duration_ms
        duration_sec = duration_ms / 1000.0 if duration_ms > 0 else 0.0

        n_expected = (
            int(round(sampling_rate * duration_sec))
            if sampling_rate > 0 and duration_sec > 0
            else 0
        )

        n_from_file = 0
        if self.header.total_ch > 0:
            bytes_per_ch = (self.file_size - data_start_offset) // self.header.total_ch
            n_from_file = bytes_per_ch // 4

        final_samples = n_expected
        if n_expected <= 0:
            final_samples = n_from_file
        elif n_from_file > 0 and abs(n_expected - n_from_file) > (n_from_file * 0.1):
            final_samples = n_from_file

        if final_samples <= 0:
            return np.array([], dtype=np.float32)

        data_block_size = final_samples * 4
        start_offset = data_start_offset + (channel_index * data_block_size)

        if start_offset >= self.file_size:
            return np.array([], dtype=np.float32)

        max_bytes = self.file_size - start_offset
        read_size = min(data_block_size, max_bytes)
        if read_size <= 0:
            return np.array([], dtype=np.float32)

        try:
            with open(self.filepath, "rb") as f:
                f.seek(start_offset)
                raw_bytes = f.read(read_size)

            actual_samples = len(raw_bytes) // 4
            if actual_samples <= 0:
                return np.array([], dtype=np.float32)

            raw_bytes = raw_bytes[: actual_samples * 4]
            return np.frombuffer(raw_bytes, dtype="<f4")

        except Exception as e:
            print(
                f"[RCPVMSParser] read_channel({channel_index}) error: {e}",
                file=sys.stderr,
            )
            return np.array([], dtype=np.float32)

    def read_all_channels(self) -> list[np.ndarray]:
        """
        Reads all channels in a single file open.
        Prefer this over calling read_channel() in a loop to avoid repeated I/O.
        Returns a list of float32 arrays, one per channel.
        """
        if not self.header or self.header.total_ch <= 0:
            return []

        extra = self.header.extra_fields or {}
        data_start_offset = int(extra.get("data_start_offset", 0))
        if data_start_offset < 0 or data_start_offset >= self.file_size:
            return []

        total_ch = self.header.total_ch
        sampling_rate = self.header.sampling_rate
        duration_ms = self.header.event_duration_ms
        duration_sec = duration_ms / 1000.0 if duration_ms > 0 else 0.0

        n_expected = (
            int(round(sampling_rate * duration_sec))
            if sampling_rate > 0 and duration_sec > 0
            else 0
        )
        bytes_available = self.file_size - data_start_offset
        n_from_file = (bytes_available // total_ch) // 4

        if n_expected <= 0:
            final_samples = n_from_file
        elif n_from_file > 0 and abs(n_expected - n_from_file) > n_from_file * 0.1:
            final_samples = n_from_file
        else:
            final_samples = n_expected

        if final_samples <= 0:
            return []

        try:
            with open(self.filepath, "rb") as f:
                f.seek(data_start_offset)
                raw = f.read(total_ch * final_samples * 4)

            all_data = np.frombuffer(raw, dtype="<f4")
            channels = []
            for ch in range(total_ch):
                start = ch * final_samples
                end = start + final_samples
                if end <= len(all_data):
                    channels.append(all_data[start:end].copy())
                else:
                    channels.append(np.array([], dtype=np.float32))
            return channels

        except Exception as e:
            print(f"[RCPVMSParser] read_all_channels error: {e}", file=sys.stderr)
            return []
