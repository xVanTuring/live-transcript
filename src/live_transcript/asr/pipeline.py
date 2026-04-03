"""ASR pipeline orchestrating streaming recognition and 2nd-pass correction."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ..audio_buffer import AudioRingBuffer
from ..protocol import (
    TranscriptEvent,
    make_correction,
    make_final,
    make_partial,
)
from .base import CorrectionEngine, StreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class SegmentState:
    segment_id: int
    start_sample: int
    start_time: float  # wall-clock time when segment started
    last_text: str = ""
    last_send_time: float = 0.0


@dataclass
class PipelineConfig:
    sample_rate: int = 16000
    enable_correction: bool = True
    debounce_ms: float = 100.0
    ring_buffer_seconds: float = 60.0


class ASRPipeline:
    """Orchestrates streaming ASR with optional 2nd-pass correction.

    Each WebSocket session creates one pipeline instance.
    """

    def __init__(
        self,
        streaming_engine: StreamingEngine,
        correction_engine: CorrectionEngine,
        config: PipelineConfig | None = None,
    ):
        self._streaming = streaming_engine
        self._correction = correction_engine
        self._config = config or PipelineConfig()

        self._buffer = AudioRingBuffer(
            max_seconds=self._config.ring_buffer_seconds,
            sample_rate=self._config.sample_rate,
        )
        self._stream_handle = self._streaming.create_stream()
        self._segment = SegmentState(
            segment_id=0,
            start_sample=0,
            start_time=time.monotonic(),
        )
        self._session_start = time.monotonic()

    async def feed_audio(self, samples: np.ndarray) -> list[TranscriptEvent]:
        """Feed audio samples and return any transcript events to send."""
        events: list[TranscriptEvent] = []

        self._buffer.append(samples)
        self._streaming.accept_waveform(
            self._stream_handle, self._config.sample_rate, samples
        )

        result = self._streaming.decode(self._stream_handle)

        # Calculate time offsets
        now = time.monotonic()
        seg_start_time = self._segment.start_time - self._session_start
        current_time = now - self._session_start

        # Check for text changes (partial / correction)
        if result.text and result.text != self._segment.last_text:
            elapsed_since_send = (now - self._segment.last_send_time) * 1000
            if elapsed_since_send >= self._config.debounce_ms or result.is_endpoint:
                if self._segment.last_text:
                    event = make_correction(
                        segment_id=self._segment.segment_id,
                        text=result.text,
                        previous_text=self._segment.last_text,
                        start_time=seg_start_time,
                        end_time=current_time,
                    )
                else:
                    event = make_partial(
                        segment_id=self._segment.segment_id,
                        text=result.text,
                        start_time=seg_start_time,
                        end_time=current_time,
                    )
                events.append(event)
                self._segment.last_text = result.text
                self._segment.last_send_time = now

        # Check for endpoint (sentence boundary)
        if result.is_endpoint and self._segment.last_text:
            final_event = await self._finalize_segment(seg_start_time, current_time)
            if final_event:
                events.append(final_event)

        return events

    async def _finalize_segment(
        self, seg_start_time: float, seg_end_time: float
    ) -> TranscriptEvent | None:
        """Run 2nd-pass correction on the completed segment."""
        seg = self._segment
        streaming_text = seg.last_text

        final_text = streaming_text
        language = ""

        if self._config.enable_correction:
            segment_audio = self._buffer.extract(
                seg.start_sample, self._buffer.total_samples_written
            )
            if segment_audio is not None and len(segment_audio) > 0:
                try:
                    correction = await asyncio.to_thread(
                        self._correction.transcribe,
                        segment_audio,
                        self._config.sample_rate,
                    )
                    if correction.text:
                        final_text = correction.text
                        language = correction.language
                except Exception:
                    logger.exception("2nd-pass correction failed, using streaming result")

        event = make_final(
            segment_id=seg.segment_id,
            text=final_text,
            start_time=seg_start_time,
            end_time=seg_end_time,
            language=language,
        )

        # Reset for next segment
        self._streaming.reset(self._stream_handle)
        self._segment = SegmentState(
            segment_id=seg.segment_id + 1,
            start_sample=self._buffer.total_samples_written,
            start_time=time.monotonic(),
        )

        return event

    async def flush(self) -> TranscriptEvent | None:
        """Flush any remaining audio as a final segment (called on stop)."""
        if not self._segment.last_text:
            return None

        now = time.monotonic()
        seg_start_time = self._segment.start_time - self._session_start
        current_time = now - self._session_start
        return await self._finalize_segment(seg_start_time, current_time)

    def close(self) -> None:
        """Release resources."""
        self._stream_handle = None
