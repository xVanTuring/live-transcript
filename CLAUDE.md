# Live Transcript

Real-time speech recognition service. Audio via WebSocket, supports Chinese + English.

## Architecture

Dual-engine ASR pipeline:
- **sherpa-onnx**: streaming recognition (low-latency partial results, built-in VAD/endpoint detection)
- **FunASR**: optional 2nd-pass correction at sentence boundaries (higher accuracy, punctuation). Supports two providers:
  - `paraformer` (default): Paraformer-Large-VAD-Punc — best Chinese accuracy, integrated punctuation
  - `sensevoice`: SenseVoice-Small — lighter, multilingual
- **Dynamic hotwords** (optional): extracts keywords from corrected text via jieba, feeds back to streaming engine to boost subsequent recognition

```
Client → WebSocket (binary PCM s16le 16kHz mono + JSON control) → FastAPI server
Server → ASR Pipeline (sherpa-onnx streaming → 2nd-pass correction → hotword feedback loop) → JSON events back
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
    ├── streaming_engine.py # sherpa-onnx wrapper (supports hotwords + modified_beam_search)
    ├── correction_engine.py# Paraformer / SenseVoice / Null correction engines
    ├── hotword_manager.py  # Dynamic hotword extraction (jieba segmentation + sliding window)
    └── pipeline.py         # Orchestrates streaming + VAD + 2nd-pass correction + hotword feedback
scripts/
├── download_models.py      # Download sherpa-onnx models
└── benchmark.py            # ASR accuracy/speed benchmark tool
testdata/                   # Audio + reference text pairs for benchmarking
```

## Commands

```bash
# Install
pip install -e ".[dev]"

# Install benchmark dependencies (jiwer, pydub)
pip install -e ".[benchmark]"

# Install FunASR engines (optional, only if using paraformer/sensevoice provider)
pip install -e ".[funasr]"

# Download models (sherpa-onnx streaming + offline paraformer)
python scripts/download_models.py

# Run server (CPU)
python -m live_transcript
python -m live_transcript -c config.yaml

# Run server (CUDA — requires onnxruntime-gpu)
pip install onnxruntime-gpu
python -m live_transcript -c config_cuda.yaml

# Test clients
python client/py_client.py --mic
python client/py_client.py --file test.wav

# Benchmark ASR accuracy/speed
python scripts/benchmark.py testdata/
python scripts/benchmark.py testdata/ -c config_a.yaml config_b.yaml
python scripts/benchmark.py testdata/ -o results.json
```

## Config

Runtime config in `config.yaml`.

### Server
- `server.host`: bind address (default `0.0.0.0`)
- `server.port`: WebSocket port (default 8765)

### Streaming Engine (sherpa-onnx)
- `streaming_engine.model_dir`: path to sherpa-onnx model
- `streaming_engine.device`: ONNX Runtime provider — `cpu` (default) or `cuda`
- `streaming_engine.num_threads`: decode threads (default 2)
- `streaming_engine.endpoint.rule1_min_trailing_silence`: long silence threshold (default 2.4s)
- `streaming_engine.endpoint.rule2_min_trailing_silence`: sentence boundary silence threshold (default 1.2s)
- `streaming_engine.endpoint.rule3_min_utterance_length`: min utterance length (default 20s)

### Dynamic Hotwords
- `streaming_engine.hotwords.enabled`: enable dynamic hotword boosting (default false). Switches decoding to `modified_beam_search`
- `streaming_engine.hotwords.score`: hotword boost weight (default 1.5)
- `streaming_engine.hotwords.max_active_paths`: beam search paths (default 4)
- `streaming_engine.hotwords.max_words`: sliding window size for dynamic hotwords (default 50)
- `streaming_engine.hotwords.min_word_length`: minimum word length to keep, filters single characters (default 2)
- `streaming_engine.hotwords.hotwords_file`: optional path to static hotwords file (one word per line)

### Correction Engine (2nd-pass)
- `correction_engine.provider`: engine provider, one of:
  - `sherpa-offline` (default): sherpa-onnx offline Paraformer — fast ONNX Runtime inference, no torch dependency
  - `paraformer`: FunASR Paraformer-Large-VAD-Punc — best accuracy, supports hotwords, requires `pip install -e ".[funasr]"`
  - `sensevoice`: FunASR SenseVoice-Small — multilingual, requires `pip install -e ".[funasr]"`
- `correction_engine.model_dir`: model directory path (sherpa-offline)
- `correction_engine.device`: `cpu` (default) or `cuda` — all providers support GPU acceleration
- `correction_engine.num_threads`: ONNX Runtime threads (sherpa-offline, default 2)
- `correction_engine.model`: FunASR model name/path (paraformer/sensevoice)
- `correction_engine.hotword`: space-separated hotwords for FunASR Paraformer (e.g. `"机器学习 深度学习"`)
- `correction_engine.language`: language hint for SenseVoice (default `zh`, also supports `auto`)

### Audio & Protocol
- `audio.sample_rate`: target sample rate (default 16000)
- `audio.chunk_duration_ms`: audio chunk size (default 60)
- `audio.ring_buffer_seconds`: ring buffer duration for segment extraction (default 60)
- `protocol.debounce_partial_ms`: min interval between partial updates (default 30)

### Benchmark
Test data: place `<name>.wav`/`.mp3` + `<name>.txt` pairs in `testdata/`. The benchmark script reports CER (character error rate), WER, processing time, and RTF per file, with a comparison summary when multiple configs are provided.

## WebSocket Protocol

Client sends: `{"type":"start","config":{...}}` → binary PCM chunks → `{"type":"stop"}`

Server sends: `partial` / `correction` / `final` events with `segment_id`, `text`, timing fields (`processing_ms`, `correction_ms`).

## Key Design Decisions

- `pipeline.feed_audio()` runs sherpa-onnx decode inline (fast enough at ~1ms/chunk); 2nd-pass correction runs in `asyncio.to_thread()` to avoid blocking
- One `OnlineRecognizer` shared across sessions; each session gets its own `OnlineStream`
- Audio ring buffer (60s) stores raw samples for segment extraction during 2nd-pass
- Must call `recognizer.is_ready()` before `decode_stream()` to avoid native crash on small chunks
- Dynamic hotwords: after each segment's 2nd-pass correction, keywords are extracted via jieba (nouns/verbs/adjectives, filtering stop words), stored in a sliding window (LRU, max 50 words), and passed to the next `create_stream(hotwords=...)`. Stream is rebuilt (not reset) at segment boundaries when hotwords are active
- Hotwords require transducer models + `modified_beam_search`; streaming paraformer models do not support hotwords

## Companion App

`AudioCap/` contains a macOS SwiftUI app that captures system audio output and streams it to this server via WebSocket. It handles 48kHz→16kHz resampling client-side.
