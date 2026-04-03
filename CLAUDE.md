# Live Transcript

Real-time speech recognition service. Audio via WebSocket, supports Chinese + English.

## Architecture

Dual-engine ASR pipeline:
- **sherpa-onnx**: streaming recognition (low-latency partial results, built-in VAD/endpoint detection)
- **FunASR SenseVoice-Small**: optional 2nd-pass correction at sentence boundaries (higher accuracy, punctuation)

```
Client → WebSocket (binary PCM s16le 16kHz mono + JSON control) → FastAPI server
Server → ASR Pipeline (sherpa-onnx streaming → SenseVoice 2nd-pass) → JSON events back
```

## Project Structure

```
src/live_transcript/
├── main.py                 # Entry point, engine init, uvicorn launch
├── server.py               # FastAPI WebSocket server
├── protocol.py             # Message types (partial/correction/final/error)
├── audio_buffer.py         # Ring buffer + PCM s16le→float32 conversion
└── asr/
    ├── base.py             # Abstract engine interfaces
    ├── streaming_engine.py # sherpa-onnx wrapper
    ├── correction_engine.py# SenseVoice wrapper + NullCorrectionEngine fallback
    └── pipeline.py         # Orchestrates streaming + VAD + 2nd-pass correction
```

## Commands

```bash
# Install
pip install -e ".[dev]"

# Download models (sherpa-onnx bilingual zh-en)
python scripts/download_models.py

# Run server
python -m live_transcript
python -m live_transcript -c config.yaml

# Test clients
python client/py_client.py --mic
python client/py_client.py --file test.wav
```

## Config

Runtime config in `config.yaml`. Key settings:
- `server.port`: WebSocket port (default 8765)
- `streaming_engine.model_dir`: path to sherpa-onnx model
- `streaming_engine.endpoint.rule2_min_trailing_silence`: sentence boundary silence threshold (seconds)
- `correction_engine.model`: SenseVoice model name
- `protocol.debounce_partial_ms`: min interval between partial updates (default 30)

## WebSocket Protocol

Client sends: `{"type":"start","config":{...}}` → binary PCM chunks → `{"type":"stop"}`

Server sends: `partial` / `correction` / `final` events with `segment_id`, `text`, timing fields (`processing_ms`, `correction_ms`).

## Key Design Decisions

- `pipeline.feed_audio()` runs sherpa-onnx decode inline (fast enough at ~1ms/chunk); SenseVoice 2nd-pass runs in `asyncio.to_thread()` to avoid blocking
- One `OnlineRecognizer` shared across sessions; each session gets its own `OnlineStream`
- Audio ring buffer (60s) stores raw samples for segment extraction during 2nd-pass
- Must call `recognizer.is_ready()` before `decode_stream()` to avoid native crash on small chunks

## Companion App

`AudioCap/` contains a macOS SwiftUI app that captures system audio output and streams it to this server via WebSocket. It handles 48kHz→16kHz resampling client-side.
