"""SenseVoice 2nd-pass correction engine wrapper."""

from __future__ import annotations

import logging

import numpy as np

from .base import CorrectionEngine, CorrectionResult

logger = logging.getLogger(__name__)


class SenseVoiceCorrectionEngine(CorrectionEngine):
    """Wraps FunASR SenseVoice-Small for high-accuracy offline transcription."""

    def __init__(self, config: dict):
        from funasr import AutoModel

        model_name = config.get("model", "FunAudioLLM/SenseVoiceSmall")
        device = config.get("device", "cpu")

        self._model = AutoModel(
            model=model_name,
            trust_remote_code=True,
            device=device,
        )
        logger.info("SenseVoice correction engine loaded: %s (device=%s)", model_name, device)

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        result = self._model.generate(
            input=samples,
            input_len=len(samples),
            cache={},
            language="auto",
            use_itn=True,
        )

        if not result:
            return CorrectionResult(text="")

        item = result[0]
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        language = item.get("language", "") if isinstance(item, dict) else ""

        return CorrectionResult(text=text.strip(), language=language)


class NullCorrectionEngine(CorrectionEngine):
    """No-op correction engine — returns empty result.

    Used when correction is disabled or SenseVoice is not available.
    """

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        return CorrectionResult(text="")
