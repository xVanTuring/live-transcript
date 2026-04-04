"""sherpa-onnx streaming ASR engine wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import CorrectionEngine as _CE  # noqa: avoid name clash
from .base import StreamHandle, StreamingEngine, StreamingResult

logger = logging.getLogger(__name__)


@dataclass
class SherpaStreamHandle(StreamHandle):
    stream: object  # sherpa_onnx.OnlineStream
    sample_count: int = 0


class SherpaOnnxStreamingEngine(StreamingEngine):
    """Wraps sherpa-onnx OnlineRecognizer for real-time streaming ASR."""

    def __init__(self, config: dict):
        import sherpa_onnx

        model_dir = Path(config["model_dir"])
        num_threads = config.get("num_threads", 2)
        sample_rate = config.get("sample_rate", 16000)
        ep = config.get("endpoint", {})
        hotwords_cfg = config.get("hotwords", {})

        # Detect model type from directory contents
        tokens = str(model_dir / "tokens.txt")

        # Check for transducer model files
        encoder = model_dir / "encoder-epoch-99-avg-1.onnx"
        decoder = model_dir / "decoder-epoch-99-avg-1.onnx"
        joiner = model_dir / "joiner-epoch-99-avg-1.onnx"

        if not encoder.exists():
            encoder = model_dir / "encoder-epoch-99-avg-1.int8.onnx"
            decoder = model_dir / "decoder-epoch-99-avg-1.int8.onnx"
            joiner = model_dir / "joiner-epoch-99-avg-1.int8.onnx"

        # Hotwords require modified_beam_search
        self._hotwords_enabled = hotwords_cfg.get("enabled", False)
        decoding_method = "modified_beam_search" if self._hotwords_enabled else "greedy_search"

        recognizer_kwargs = dict(
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=ep.get("rule1_min_trailing_silence", 2.4),
            rule2_min_trailing_silence=ep.get("rule2_min_trailing_silence", 1.2),
            rule3_min_utterance_length=ep.get("rule3_min_utterance_length", 20.0),
            decoding_method=decoding_method,
        )

        if self._hotwords_enabled:
            recognizer_kwargs["max_active_paths"] = hotwords_cfg.get("max_active_paths", 4)
            hotwords_file = hotwords_cfg.get("hotwords_file", "")
            if hotwords_file and Path(hotwords_file).exists():
                recognizer_kwargs["hotwords_file"] = hotwords_file
            recognizer_kwargs["hotwords_score"] = hotwords_cfg.get("score", 1.5)
            # BPE vocab for bilingual models
            bpe_vocab = model_dir / "bpe.vocab"
            if bpe_vocab.exists():
                recognizer_kwargs["modeling_unit"] = "cjkchar+bpe"
                recognizer_kwargs["bpe_vocab"] = str(bpe_vocab)
            else:
                recognizer_kwargs["modeling_unit"] = "cjkchar"

        if encoder.exists():
            # Transducer model
            recognizer_kwargs.update(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
            )
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**recognizer_kwargs)
        else:
            # Try CTC or other model types - list files for debugging
            onnx_files = list(model_dir.glob("*.onnx"))
            raise FileNotFoundError(
                f"Cannot find expected model files in {model_dir}. "
                f"Found: {[f.name for f in onnx_files]}"
            )

        self._sample_rate = sample_rate
        logger.info(
            "sherpa-onnx streaming engine initialized from %s (decoding=%s, hotwords=%s)",
            model_dir, decoding_method, self._hotwords_enabled,
        )

    def create_stream(self, hotwords: str = "") -> SherpaStreamHandle:
        if hotwords and self._hotwords_enabled:
            stream = self._recognizer.create_stream(hotwords=hotwords)
        else:
            stream = self._recognizer.create_stream()
        return SherpaStreamHandle(stream=stream)

    def accept_waveform(self, handle: SherpaStreamHandle, sample_rate: int, samples: np.ndarray) -> None:
        handle.stream.accept_waveform(sample_rate, samples)
        handle.sample_count += len(samples)

    def decode(self, handle: SherpaStreamHandle) -> StreamingResult:
        while self._recognizer.is_ready(handle.stream):
            self._recognizer.decode_stream(handle.stream)
        result = self._recognizer.get_result(handle.stream)
        is_endpoint = self._recognizer.is_endpoint(handle.stream)
        return StreamingResult(
            text=result.strip() if isinstance(result, str) else result.text.strip(),
            is_endpoint=is_endpoint,
        )

    def reset(self, handle: SherpaStreamHandle) -> None:
        self._recognizer.reset(handle.stream)
        handle.sample_count = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
