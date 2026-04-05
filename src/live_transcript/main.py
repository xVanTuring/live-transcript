"""Entry point for the live-transcript server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file %s not found, using defaults", path)
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_engines(config: dict):
    """Initialize ASR engines based on configuration."""
    from .asr.correction_engine import NullCorrectionEngine

    # Streaming engine (sherpa-onnx)
    streaming_config = config.get("streaming_engine", {})
    model_dir = Path(streaming_config.get("model_dir", ""))

    if model_dir.exists():
        from .asr.streaming_engine import SherpaOnnxStreamingEngine
        streaming_engine = SherpaOnnxStreamingEngine(streaming_config)
    else:
        logger.error(
            "Streaming model not found at %s. "
            "Run 'python scripts/download_models.py' first.",
            model_dir,
        )
        raise SystemExit(1)

    # Correction engine — optional
    correction_config = config.get("correction_engine", {})
    provider = correction_config.get("provider", "sensevoice")
    try:
        if provider == "sherpa-offline":
            from .asr.correction_engine import SherpaOfflineCorrectionEngine
            correction_engine = SherpaOfflineCorrectionEngine(correction_config)
        elif provider == "whisper":
            from .asr.correction_engine import WhisperCorrectionEngine
            correction_engine = WhisperCorrectionEngine(correction_config)
        elif provider == "paraformer":
            from .asr.correction_engine import ParaformerCorrectionEngine
            correction_engine = ParaformerCorrectionEngine(correction_config)
        elif provider == "sensevoice":
            from .asr.correction_engine import SenseVoiceCorrectionEngine
            correction_engine = SenseVoiceCorrectionEngine(correction_config)
        else:
            raise ValueError(f"Unknown correction engine provider: {provider}")
    except Exception as e:
        logger.warning(
            "Correction engine '%s' not available (%s). "
            "Running without 2nd-pass correction.",
            provider, e,
        )
        correction_engine = NullCorrectionEngine()

    return streaming_engine, correction_engine


def main():
    parser = argparse.ArgumentParser(description="Live Transcript Server")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)

    logger.info("Initializing ASR engines...")
    streaming_engine, correction_engine = create_engines(config)

    from .server import app, configure

    configure(streaming_engine, correction_engine, config)

    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8765)

    import uvicorn

    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
