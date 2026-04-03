"""Abstract base classes for ASR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class StreamingResult:
    text: str
    is_endpoint: bool
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class CorrectionResult:
    text: str
    language: str = ""


class StreamingEngine(ABC):
    """Interface for a streaming ASR engine that processes audio chunk by chunk."""

    @abstractmethod
    def create_stream(self) -> StreamHandle:
        ...

    @abstractmethod
    def accept_waveform(self, handle: StreamHandle, sample_rate: int, samples: np.ndarray) -> None:
        ...

    @abstractmethod
    def decode(self, handle: StreamHandle) -> StreamingResult:
        ...

    @abstractmethod
    def reset(self, handle: StreamHandle) -> None:
        ...


class StreamHandle:
    """Opaque handle for a streaming ASR session."""
    pass


class CorrectionEngine(ABC):
    """Interface for an offline ASR engine used for 2nd-pass correction."""

    @abstractmethod
    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        ...
