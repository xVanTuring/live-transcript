#!/usr/bin/env python3
"""Download required ASR models for live-transcript."""

import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

SHERPA_MODEL = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
SHERPA_URL = (
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{SHERPA_MODEL}.tar.bz2"
)

SHERPA_OFFLINE_MODEL = "sherpa-onnx-paraformer-zh-2024-03-09"
SHERPA_OFFLINE_URL = (
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{SHERPA_OFFLINE_MODEL}.tar.bz2"
)


def download_sherpa_model():
    dest = MODELS_DIR / SHERPA_MODEL
    if dest.exists():
        print(f"[OK] sherpa-onnx model already exists: {dest}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODELS_DIR / f"{SHERPA_MODEL}.tar.bz2"

    print(f"Downloading sherpa-onnx bilingual zh-en model...")
    print(f"  URL: {SHERPA_URL}")
    subprocess.run(
        ["curl", "-L", "-o", str(archive), SHERPA_URL],
        check=True,
    )

    print("Extracting...")
    subprocess.run(
        ["tar", "xjf", str(archive), "-C", str(MODELS_DIR)],
        check=True,
    )
    archive.unlink()
    print(f"[OK] Model extracted to {dest}")


def download_sherpa_offline_model():
    """Download sherpa-onnx offline Paraformer model for 2nd-pass correction."""
    dest = MODELS_DIR / SHERPA_OFFLINE_MODEL
    if dest.exists():
        print(f"[OK] sherpa-onnx offline model already exists: {dest}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODELS_DIR / f"{SHERPA_OFFLINE_MODEL}.tar.bz2"

    print("Downloading sherpa-onnx offline Paraformer zh model...")
    print(f"  URL: {SHERPA_OFFLINE_URL}")
    subprocess.run(
        ["curl", "-L", "-o", str(archive), SHERPA_OFFLINE_URL],
        check=True,
    )

    print("Extracting...")
    subprocess.run(
        ["tar", "xjf", str(archive), "-C", str(MODELS_DIR)],
        check=True,
    )
    archive.unlink()
    print(f"[OK] Model extracted to {dest}")


def download_sensevoice_model():
    """SenseVoice model is auto-downloaded by FunASR on first use.

    This function just verifies funasr is installed.
    """
    try:
        import funasr  # noqa: F401
        print("[OK] funasr is installed. SenseVoice model will be downloaded on first use.")
    except ImportError:
        print("[WARN] funasr not installed. Install with: pip install funasr")
        print("       SenseVoice 2nd-pass correction will be disabled without it.")


def main():
    print("=" * 60)
    print("Live Transcript - Model Downloader")
    print("=" * 60)
    print()

    download_sherpa_model()
    print()
    download_sherpa_offline_model()
    print()
    download_sensevoice_model()

    print()
    print("Done! You can now start the server:")
    print("  python -m live_transcript.main")


if __name__ == "__main__":
    main()
