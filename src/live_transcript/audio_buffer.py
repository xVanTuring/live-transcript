"""Audio ring buffer for storing streaming audio and extracting segments."""

from __future__ import annotations

import numpy as np


class AudioRingBuffer:
    """Fixed-size ring buffer that stores float32 audio samples.

    Supports appending audio chunks and extracting segments by absolute
    sample position for 2nd-pass correction.
    """

    def __init__(self, max_seconds: float = 60.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.capacity = int(max_seconds * sample_rate)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0  # absolute sample count written so far
        self._start_pos = 0  # absolute position of earliest available sample

    @property
    def total_samples_written(self) -> int:
        return self._write_pos

    def append(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n == 0:
            return

        if n >= self.capacity:
            # If chunk is larger than buffer, keep only the tail
            samples = samples[-self.capacity:]
            n = self.capacity

        ring_start = self._write_pos % self.capacity
        space_to_end = self.capacity - ring_start

        if n <= space_to_end:
            self._buf[ring_start:ring_start + n] = samples
        else:
            self._buf[ring_start:] = samples[:space_to_end]
            self._buf[:n - space_to_end] = samples[space_to_end:]

        self._write_pos += n
        # Update start position if buffer has wrapped
        if self._write_pos - self._start_pos > self.capacity:
            self._start_pos = self._write_pos - self.capacity

    def extract(self, abs_start: int, abs_end: int) -> np.ndarray | None:
        """Extract audio between absolute sample positions.

        Returns None if the requested range has been overwritten.
        """
        if abs_start < self._start_pos:
            return None  # Data has been overwritten
        if abs_end > self._write_pos:
            abs_end = self._write_pos
        if abs_start >= abs_end:
            return np.array([], dtype=np.float32)

        n = abs_end - abs_start
        result = np.empty(n, dtype=np.float32)

        ring_start = abs_start % self.capacity
        space_to_end = self.capacity - ring_start

        if n <= space_to_end:
            result[:] = self._buf[ring_start:ring_start + n]
        else:
            result[:space_to_end] = self._buf[ring_start:]
            result[space_to_end:] = self._buf[:n - space_to_end]

        return result

    def seconds_to_samples(self, seconds: float) -> int:
        return int(seconds * self.sample_rate)


def pcm_s16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM 16-bit signed little-endian bytes to float32 in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0
