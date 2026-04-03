#!/usr/bin/env python3
"""Python test client for the live-transcript WebSocket server.

Usage:
    # Stream from a WAV file:
    python py_client.py --file test.wav

    # Stream from microphone (requires sounddevice):
    python py_client.py --mic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import wave
from pathlib import Path


async def stream_wav_file(uri: str, wav_path: str, chunk_ms: int = 60):
    """Stream a WAV file to the server in chunks."""
    import websockets

    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    chunk_samples = int(sample_rate * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample

    async with websockets.connect(uri) as ws:
        # Send start
        await ws.send(json.dumps({
            "type": "start",
            "config": {
                "sample_rate": sample_rate,
                "encoding": "pcm_s16le",
                "channels": 1,
                "language": "auto",
                "enable_correction": True,
            },
        }))

        # Wait for ready
        ready = json.loads(await ws.recv())
        print(f"Server ready: {ready}")

        # Stream audio chunks
        offset = 0
        while offset < len(frames):
            chunk = frames[offset:offset + chunk_bytes]
            await ws.send(chunk)
            offset += chunk_bytes
            # Simulate real-time pace
            await asyncio.sleep(chunk_ms / 1000.0)

            # Check for incoming messages (non-blocking)
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                    event = json.loads(msg)
                    print_event(event)
            except (asyncio.TimeoutError, Exception):
                pass

        # Send stop
        await ws.send(json.dumps({"type": "stop"}))

        # Receive remaining messages
        try:
            async for msg in ws:
                event = json.loads(msg)
                print_event(event)
        except Exception:
            pass


async def stream_microphone(uri: str, sample_rate: int = 16000, chunk_ms: int = 60):
    """Stream from microphone to the server."""
    import websockets

    try:
        import sounddevice as sd
    except ImportError:
        print("Install sounddevice for microphone input: pip install sounddevice")
        sys.exit(1)

    chunk_samples = int(sample_rate * chunk_ms / 1000)

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "config": {
                "sample_rate": sample_rate,
                "encoding": "pcm_s16le",
                "channels": 1,
                "language": "auto",
                "enable_correction": True,
            },
        }))

        ready = json.loads(await ws.recv())
        print(f"Server ready: {ready}")
        print("Listening... Press Ctrl+C to stop.\n")

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[audio warning] {status}", file=sys.stderr)
            # Convert float32 to int16 PCM
            pcm = (indata[:, 0] * 32767).astype("<i2").tobytes()
            audio_queue.put_nowait(pcm)

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=audio_callback,
        )

        async def send_audio():
            with stream:
                while True:
                    chunk = await audio_queue.get()
                    await ws.send(chunk)

        async def recv_events():
            async for msg in ws:
                event = json.loads(msg)
                print_event(event)

        try:
            await asyncio.gather(send_audio(), recv_events())
        except KeyboardInterrupt:
            await ws.send(json.dumps({"type": "stop"}))
            # Drain remaining events
            try:
                async for msg in ws:
                    event = json.loads(msg)
                    print_event(event)
            except Exception:
                pass


def print_event(event: dict):
    t = event.get("type", "?")
    seg = event.get("segment_id", "")
    text = event.get("text", "")

    if t == "partial":
        print(f"  [partial  seg={seg}] {text}")
    elif t == "correction":
        prev = event.get("previous_text", "")
        print(f"  [correct  seg={seg}] {prev} → {text}")
    elif t == "final":
        lang = event.get("language", "")
        print(f"  ✓ [final   seg={seg}] {text}  ({lang})")
    elif t == "error":
        print(f"  ✗ [error] {event.get('code')}: {event.get('message')}")
    else:
        print(f"  [{t}] {event}")


def main():
    parser = argparse.ArgumentParser(description="Live Transcript Client")
    parser.add_argument("--uri", default="ws://localhost:8765/ws/transcribe")
    parser.add_argument("--file", help="WAV file to stream")
    parser.add_argument("--mic", action="store_true", help="Stream from microphone")
    parser.add_argument("--chunk-ms", type=int, default=60, help="Chunk duration in ms")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    if args.file:
        asyncio.run(stream_wav_file(args.uri, args.file, args.chunk_ms))
    elif args.mic:
        asyncio.run(stream_microphone(args.uri, args.sample_rate, args.chunk_ms))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
