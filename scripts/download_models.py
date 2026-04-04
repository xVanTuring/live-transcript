#!/usr/bin/env python3
"""Download required ASR models for live-transcript."""

import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS = [
    {
        "name": "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        "desc": "sherpa-onnx streaming bilingual zh-en",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
               "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
    },
    {
        "name": "sherpa-onnx-paraformer-zh-2024-03-09",
        "desc": "sherpa-onnx offline Paraformer zh",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
               "sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2",
    },
]


def download_model(name: str, desc: str, url: str):
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"[OK] {desc} already exists: {dest}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing archive (already downloaded or manually placed)
    archive = None
    for ext in (".tar.bz2", ".tar.gz", ".tgz", ".tar.xz", ".zip"):
        candidate = MODELS_DIR / f"{name}{ext}"
        if candidate.exists():
            archive = candidate
            print(f"[OK] Found existing archive: {archive.name}")
            break

    if archive is None:
        archive = MODELS_DIR / f"{name}.tar.bz2"
        print(f"Downloading {desc}...")
        print(f"  URL: {url}")
        subprocess.run(
            ["curl", "-L", "-o", str(archive), url],
            check=True,
        )

    # Extract
    print(f"Extracting {archive.name}...")
    suffix = archive.name.split(".", 1)[1] if "." in archive.name else ""

    if suffix in ("tar.bz2", "tbz2"):
        subprocess.run(["tar", "xjf", str(archive), "-C", str(MODELS_DIR)], check=True)
    elif suffix in ("tar.gz", "tgz"):
        subprocess.run(["tar", "xzf", str(archive), "-C", str(MODELS_DIR)], check=True)
    elif suffix == "tar.xz":
        subprocess.run(["tar", "xJf", str(archive), "-C", str(MODELS_DIR)], check=True)
    elif suffix == "zip":
        subprocess.run(["unzip", "-q", str(archive), "-d", str(MODELS_DIR)], check=True)
    else:
        print(f"  WARNING: Unknown archive format: {archive.name}, skipping extraction")
        return

    archive.unlink()
    print(f"[OK] Extracted to {dest}")


def check_sensevoice():
    """SenseVoice model is auto-downloaded by FunASR on first use."""
    try:
        import funasr  # noqa: F401
        print("[OK] funasr is installed. SenseVoice model will be downloaded on first use.")
    except ImportError:
        print("[INFO] funasr not installed (optional). Install with: pip install -e '.[funasr]'")


def main():
    print("=" * 60)
    print("Live Transcript - Model Downloader")
    print("=" * 60)
    print()

    for model in MODELS:
        download_model(**model)
        print()

    check_sensevoice()

    print()
    print("Done! You can now start the server:")
    print("  python -m live_transcript")


if __name__ == "__main__":
    main()
