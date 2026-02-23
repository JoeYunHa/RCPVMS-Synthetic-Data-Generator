"""
NIMS Event Binary Parser - Base Class (Interface with Safe Logging)
"""

import os
import sys
import struct
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from abc import ABC, abstractmethod
from datetime import datetime


@dataclass
class EventHeader:
    site_id: str
    system_id: int
    system_name: str
    event_ch: int
    total_ch: int
    event_type: int
    event_date: str
    alarm_result: int
    file_version: str
    sampling_rate: int
    event_duration_ms: int
    num_passes: int = 0
    sensitivity: float = 1.0
    extra_fields: Optional[Dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.extra_fields:
            d.update(self.extra_fields)
        return d


class BaseEventParser(ABC):
    SYSTEM_ID: int = -1
    SYSTEM_NAME: str = "Unknown"

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.header: Optional[EventHeader] = None

        # 초기화 로그
        self.log(
            f"Initialized Parser for: {os.path.basename(filepath)} ({self.file_size} bytes)"
        )

        try:
            self.parse_header()
            self.log(
                f"Header Parsed Successfully: SR={self.header.sampling_rate}, Dur={self.header.event_duration_ms}ms, Chs={self.header.total_ch}"
            )
        except Exception as e:
            self.log(f"CRITICAL HEADER ERROR: {e}")
            raise

    def log(self, message: str):
        """
        [CRITICAL FIX]
        Writes logs to stderr instead of stdout.
        stdout is reserved for JSON IPC communication.
        """
        # NOTE: Log output disabled for performance on large files.
        return

    @abstractmethod
    def _parse_header_impl(self) -> EventHeader:
        pass

    def parse_header(self) -> EventHeader:
        if self.header is None:
            self.header = self._parse_header_impl()
        return self.header

    @abstractmethod
    def read_channel(self, channel_index: int) -> np.ndarray:
        pass

    def read_channel_range(
        self, channel_index: int, start_sample: int, end_sample_exclusive: int
    ) -> np.ndarray:
        """채널 데이터 범위 읽기. end_sample_exclusive는 Python 관례에 따라 exclusive."""
        full_data = self.read_channel(channel_index)
        if len(full_data) == 0:
            return full_data

        start_sample = max(0, start_sample)
        end_sample_exclusive = min(end_sample_exclusive, len(full_data))

        if start_sample >= end_sample_exclusive:
            return np.array([], dtype=np.float32)

        return full_data[start_sample:end_sample_exclusive]

    def get_file_info(self) -> dict:
        """파일 정보 반환 (main.py 호환성)"""
        header = self.parse_header()

        # 샘플 수 계산 - extra_fields 우선, 그 다음 계산 방식
        samples_per_channel = 0

        # 방법 0: extra_fields에서 computed_samples_per_channel 확인 (가장 정확)
        if header.extra_fields:
            samples_per_channel = header.extra_fields.get(
                "computed_samples_per_channel", 0
            )

        # 방법 1: 헤더의 샘플링 레이트와 지속시간으로 계산
        if (
            samples_per_channel <= 0
            and header.sampling_rate > 0
            and header.event_duration_ms > 0
        ):
            samples_per_channel = int(
                header.sampling_rate * (header.event_duration_ms / 1000.0)
            )

        # 방법 2: IVMS num_passes 사용
        if (
            samples_per_channel <= 0
            and header.num_passes > 0
            and header.sampling_rate > 0
        ):
            samples_per_channel = int(10.24 * header.sampling_rate * header.num_passes)

        # 방법 3: 파일 크기에서 역산 (fallback)
        if samples_per_channel <= 0:
            estimated_header = 512  # 일반적인 헤더 크기
            if hasattr(self, "HEADER_SIZE"):
                estimated_header = self.HEADER_SIZE
            elif hasattr(self, "GLOBAL_HEADER_SIZE"):
                estimated_header = self.GLOBAL_HEADER_SIZE

            if header.total_ch > 0:
                total_data_size = self.file_size - estimated_header
                if total_data_size > 0:
                    samples_per_channel = (total_data_size // 4) // header.total_ch

        # 채널 정보 생성 (1-based: Channel_1, Channel_2, ...)
        channels = {}
        for i in range(header.total_ch):
            channels[f"Channel_{i + 1}"] = {
                "samples": samples_per_channel,
                "sampling_rate": header.sampling_rate,
            }

        return {
            "file_format": "nims_event",
            "file_size": self.file_size,
            "system_id": header.system_id,
            "system_name": header.system_name,
            "header": header.to_dict(),
            "total_samples": samples_per_channel * header.total_ch,
            "samples_per_channel": samples_per_channel,
            "channels": channels,
        }

    @staticmethod
    def decode_string(raw_bytes: bytes, encoding="utf-8") -> str:
        try:
            return raw_bytes.split(b"\x00")[0].decode(encoding, errors="ignore").strip()
        except (UnicodeDecodeError, ValueError, IndexError, AttributeError):
            return ""
