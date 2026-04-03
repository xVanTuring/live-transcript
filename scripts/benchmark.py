#!/usr/bin/env python3
"""ASR benchmark tool — measure accuracy (CER/WER) and speed across engines.

Usage:
    # Run with default config (uses config.yaml correction engine)
    python scripts/benchmark.py testdata/

    # Compare two engine configs
    python scripts/benchmark.py testdata/ -c config_sensevoice.yaml config_paraformer.yaml

    # Only run correction engine (skip streaming)
    python scripts/benchmark.py testdata/ --correction-only

Test data layout:
    testdata/
    ├── sample1.wav      # or .mp3
    ├── sample1.txt      # reference transcript (plain text, UTF-8)
    ├── sample2.mp3
    └── sample2.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load wav or mp3 file and return float32 samples at target sample rate."""
    import wave

    suffix = path.suffix.lower()

    if suffix == ".wav":
        return _load_wav(path, target_sr)
    elif suffix == ".mp3":
        return _load_mp3(path, target_sr)
    else:
        raise ValueError(f"Unsupported audio format: {suffix}")


def _load_wav(path: Path, target_sr: int) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, f"Expected 16-bit WAV, got {wf.getsampwidth()*8}-bit"
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0]

    if sample_rate != target_sr:
        samples = _resample(samples, sample_rate, target_sr)

    return samples


def _load_mp3(path: Path, target_sr: int) -> np.ndarray:
    try:
        from pydub import AudioSegment
    except ImportError:
        print("ERROR: pydub is required for mp3 support. Install with: pip install pydub", file=sys.stderr)
        sys.exit(1)

    audio = AudioSegment.from_mp3(str(path))
    audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    duration = len(samples) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    cer: float = 0.0
    wer: float = 0.0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    ref_length: int = 0  # reference character count
    processing_ms: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0  # real-time factor: processing_time / audio_duration


def compute_metrics(reference: str, hypothesis: str) -> Metrics:
    """Compute CER and WER. Uses character-level evaluation for CER (Chinese-friendly)."""
    import jiwer

    ref_clean = _normalize(reference)
    hyp_clean = _normalize(hypothesis)

    if not ref_clean:
        return Metrics()

    # CER — character-level (works for Chinese)
    char_out = jiwer.process_characters(ref_clean, hyp_clean)

    # WER — word-level (space-separated tokens)
    wer_val = jiwer.wer(ref_clean, hyp_clean) if ref_clean.strip() else 0.0

    return Metrics(
        cer=char_out.cer,
        wer=wer_val,
        substitutions=char_out.substitutions,
        deletions=char_out.deletions,
        insertions=char_out.insertions,
        hits=char_out.hits,
        ref_length=char_out.hits + char_out.substitutions + char_out.deletions,
    )


def _normalize(text: str) -> str:
    """Normalize text for comparison: strip whitespace, lowercase."""
    import re
    text = text.strip()
    # Remove common SenseVoice/Paraformer tags like <|zh|>, <|en|>, etc.
    text = re.sub(r"<\|[^|]*\|>", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Test case discovery
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    audio_path: Path
    reference: str
    name: str


def discover_test_cases(data_dir: Path) -> list[TestCase]:
    """Find audio + txt pairs in the data directory."""
    cases = []
    for txt_path in sorted(data_dir.glob("*.txt")):
        stem = txt_path.stem
        audio_path = None
        for ext in (".wav", ".mp3"):
            candidate = data_dir / f"{stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        if audio_path is None:
            print(f"WARNING: No audio file found for {txt_path.name}, skipping", file=sys.stderr)
            continue
        reference = txt_path.read_text(encoding="utf-8").strip()
        if not reference:
            print(f"WARNING: Empty reference text in {txt_path.name}, skipping", file=sys.stderr)
            continue
        cases.append(TestCase(audio_path=audio_path, reference=reference, name=stem))
    return cases


# ---------------------------------------------------------------------------
# Engine runners
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    name: str
    hypothesis: str
    metrics: Metrics


def run_correction_engine(engine, samples: np.ndarray, sample_rate: int, case: TestCase) -> RunResult:
    """Run a single correction engine on one test case."""
    audio_duration = len(samples) / sample_rate

    t0 = time.monotonic()
    result = engine.transcribe(samples, sample_rate)
    elapsed_ms = (time.monotonic() - t0) * 1000

    metrics = compute_metrics(case.reference, result.text)
    metrics.processing_ms = round(elapsed_ms, 2)
    metrics.audio_duration_s = round(audio_duration, 2)
    metrics.rtf = round(elapsed_ms / 1000 / audio_duration, 4) if audio_duration > 0 else 0

    return RunResult(name=case.name, hypothesis=result.text, metrics=metrics)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(config_name: str, results: list[RunResult]):
    """Print a formatted results table."""
    print(f"\n{'='*80}")
    print(f" Config: {config_name}")
    print(f"{'='*80}")
    print(f"{'Name':<20} {'CER':>7} {'WER':>7} {'Time(ms)':>10} {'RTF':>7} {'Audio(s)':>9}  Hypothesis")
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*9}  {'-'*20}")

    total_cer_num = 0
    total_cer_den = 0
    total_ms = 0.0
    total_audio = 0.0

    for r in results:
        m = r.metrics
        print(
            f"{r.name:<20} {m.cer:>6.1%} {m.wer:>6.1%} {m.processing_ms:>10.1f} "
            f"{m.rtf:>7.4f} {m.audio_duration_s:>9.2f}  {r.hypothesis[:50]}"
        )
        total_cer_num += m.substitutions + m.deletions + m.insertions
        total_cer_den += m.ref_length
        total_ms += m.processing_ms
        total_audio += m.audio_duration_s

    # Summary
    avg_cer = total_cer_num / total_cer_den if total_cer_den > 0 else 0
    avg_rtf = (total_ms / 1000) / total_audio if total_audio > 0 else 0
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*9}")
    print(
        f"{'TOTAL':<20} {avg_cer:>6.1%} {'':>7} {total_ms:>10.1f} "
        f"{avg_rtf:>7.4f} {total_audio:>9.2f}"
    )


def save_results_json(output_path: Path, all_results: dict[str, list[RunResult]]):
    """Save detailed results as JSON for later comparison."""
    data = {}
    for config_name, results in all_results.items():
        data[config_name] = [
            {
                "name": r.name,
                "hypothesis": r.hypothesis,
                "cer": r.metrics.cer,
                "wer": r.metrics.wer,
                "substitutions": r.metrics.substitutions,
                "deletions": r.metrics.deletions,
                "insertions": r.metrics.insertions,
                "hits": r.metrics.hits,
                "ref_length": r.metrics.ref_length,
                "processing_ms": r.metrics.processing_ms,
                "audio_duration_s": r.metrics.audio_duration_s,
                "rtf": r.metrics.rtf,
            }
            for r in results
        ]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed results saved to {output_path}")


# ---------------------------------------------------------------------------
# Engine creation from config
# ---------------------------------------------------------------------------

def create_correction_engine(config: dict):
    """Create a correction engine from config dict (same logic as main.py)."""
    from live_transcript.asr.correction_engine import (
        NullCorrectionEngine,
        ParaformerCorrectionEngine,
        SenseVoiceCorrectionEngine,
    )

    correction_config = config.get("correction_engine", {})
    provider = correction_config.get("provider", "sensevoice")

    if provider == "paraformer":
        return ParaformerCorrectionEngine(correction_config)
    elif provider == "sensevoice":
        return SenseVoiceCorrectionEngine(correction_config)
    else:
        raise ValueError(f"Unknown correction engine provider: {provider}")


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASR benchmark — measure accuracy and speed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing audio + txt test pairs")
    parser.add_argument(
        "-c", "--configs", nargs="+", default=["config.yaml"],
        help="One or more config YAML files to compare (default: config.yaml)",
    )
    parser.add_argument(
        "--correction-only", action="store_true", default=True,
        help="Only benchmark the correction engine (default)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Save detailed results to JSON file",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Target sample rate (default: 16000)",
    )
    args = parser.parse_args()

    # Discover test cases
    cases = discover_test_cases(args.data_dir)
    if not cases:
        print(f"ERROR: No test cases found in {args.data_dir}", file=sys.stderr)
        print("Expected: <name>.wav/.mp3 + <name>.txt pairs", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(cases)} test case(s) in {args.data_dir}")

    # Pre-load all audio
    print("Loading audio files...")
    audio_cache: dict[str, np.ndarray] = {}
    for case in cases:
        audio_cache[case.name] = load_audio(case.audio_path, args.sample_rate)
        duration = len(audio_cache[case.name]) / args.sample_rate
        print(f"  {case.audio_path.name}: {duration:.1f}s")

    # Run each config
    all_results: dict[str, list[RunResult]] = {}

    for config_path in args.configs:
        config_name = Path(config_path).stem
        print(f"\nLoading engine: {config_path} ...")
        config = load_config(config_path)
        engine = create_correction_engine(config)

        results = []
        for case in cases:
            samples = audio_cache[case.name]
            r = run_correction_engine(engine, samples, args.sample_rate, case)
            results.append(r)

        all_results[config_name] = results
        print_results_table(config_name, results)

    # Comparison summary if multiple configs
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(" Comparison Summary")
        print(f"{'='*80}")
        print(f"{'Config':<25} {'Avg CER':>9} {'Avg RTF':>9} {'Total Time':>12}")
        print(f"{'-'*25} {'-'*9} {'-'*9} {'-'*12}")
        for config_name, results in all_results.items():
            total_err = sum(r.metrics.substitutions + r.metrics.deletions + r.metrics.insertions for r in results)
            total_ref = sum(r.metrics.ref_length for r in results)
            total_ms = sum(r.metrics.processing_ms for r in results)
            total_audio = sum(r.metrics.audio_duration_s for r in results)
            avg_cer = total_err / total_ref if total_ref > 0 else 0
            avg_rtf = (total_ms / 1000) / total_audio if total_audio > 0 else 0
            print(f"{config_name:<25} {avg_cer:>8.1%} {avg_rtf:>9.4f} {total_ms:>10.1f}ms")

    # Save JSON
    if args.output:
        save_results_json(args.output, all_results)


if __name__ == "__main__":
    main()
