"""2nd-pass correction engine wrappers (SenseVoice, Paraformer)."""

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
        self._language = config.get("language", "zh")

        self._model = AutoModel(
            model=model_name,
            trust_remote_code=True,
            device=device,
        )
        logger.info("SenseVoice correction engine loaded: %s (device=%s, language=%s)", model_name, device, self._language)

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        result = self._model.generate(
            input=samples,
            input_len=len(samples),
            cache={},
            language=self._language,
            use_itn=True,
        )

        if not result:
            return CorrectionResult(text="")

        item = result[0]
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        language = item.get("language", "") if isinstance(item, dict) else ""

        return CorrectionResult(text=text.strip(), language=language)


class ParaformerCorrectionEngine(CorrectionEngine):
    """Wraps FunASR Paraformer-Large-VAD-Punc for high-accuracy Chinese transcription.

    Significantly better Chinese accuracy than SenseVoice-Small, with integrated
    VAD and punctuation restoration.
    """

    def __init__(self, config: dict):
        from funasr import AutoModel

        model_name = config.get(
            "model",
            "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        device = config.get("device", "cpu")
        self._hotword = config.get("hotword", "")

        self._model = AutoModel(
            model=model_name,
            device=device,
            vad_kwargs={"max_single_segment_time": 60000},
            disable_update=True,
        )
        logger.info("Paraformer correction engine loaded: %s (device=%s)", model_name, device)

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        kwargs: dict = {
            "input": samples,
            "use_itn": True,
            "ban_unk": True,
        }
        if self._hotword:
            kwargs["hotword"] = self._hotword

        result = self._model.generate(**kwargs)

        if not result:
            return CorrectionResult(text="")

        item = result[0]
        text = item.get("text", "") if isinstance(item, dict) else str(item)

        return CorrectionResult(text=text.strip(), language="zh")


class NullCorrectionEngine(CorrectionEngine):
    """No-op correction engine — returns empty result.

    Used when correction is disabled or SenseVoice is not available.
    """

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        return CorrectionResult(text="")
