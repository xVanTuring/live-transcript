"""2nd-pass correction engine wrappers (sherpa-onnx offline, faster-whisper, FunASR Paraformer, SenseVoice)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import CorrectionEngine, CorrectionResult

logger = logging.getLogger(__name__)


class SherpaOfflineCorrectionEngine(CorrectionEngine):
    """Wraps sherpa-onnx OfflineRecognizer (Paraformer) for fast inference.

    Uses ONNX Runtime — significantly faster than PyTorch on CPU,
    and supports CUDA for GPU acceleration. No torch/funasr dependencies required.
    """

    def __init__(self, config: dict):
        import sherpa_onnx

        model_dir = Path(config.get(
            "model_dir",
            "./models/sherpa-onnx-paraformer-zh-2024-03-09",
        ))
        num_threads = config.get("num_threads", 2)
        provider = config.get("device", "cpu")

        # CUDA: prefer fp32 for accuracy; CPU: prefer int8 for speed
        if provider == "cuda":
            model_file = model_dir / "model.onnx"
            if not model_file.exists():
                model_file = model_dir / "model.int8.onnx"
        else:
            model_file = model_dir / "model.int8.onnx"
            if not model_file.exists():
                model_file = model_dir / "model.onnx"

        tokens = model_dir / "tokens.txt"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Offline paraformer model not found at {model_dir}. "
                "Run 'python scripts/download_models.py' to download."
            )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(model_file),
            tokens=str(tokens),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
        )
        logger.info(
            "sherpa-onnx offline Paraformer loaded: %s (provider=%s, threads=%d)",
            model_file.name, provider, num_threads,
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        return CorrectionResult(text=text, language="zh")


class WhisperCorrectionEngine(CorrectionEngine):
    """Wraps faster-whisper for high-accuracy 2nd-pass correction.

    Best accuracy for mixed Chinese/English content. Requires GPU for
    low-latency inference (RTX 4090: ~50-100ms for 3-8s audio).
    """

    def __init__(self, config: dict):
        from faster_whisper import WhisperModel

        model_size = config.get("model", "large-v3")
        device = config.get("device", "cuda")
        compute_type = config.get("compute_type", "float16")
        self._language = config.get("language", "zh")
        self._initial_prompt = config.get(
            "initial_prompt",
            "以下是普通话语音识别，包含中英文内容。",
        )
        self._beam_size = config.get("beam_size", 5)

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            "faster-whisper correction engine loaded: %s (device=%s, compute_type=%s)",
            model_size, device, compute_type,
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> CorrectionResult:
        segments, info = self._model.transcribe(
            samples,
            language=self._language,
            beam_size=self._beam_size,
            initial_prompt=self._initial_prompt,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        text = "".join(seg.text for seg in segments).strip()
        language = info.language or self._language

        return CorrectionResult(text=text, language=language)


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
