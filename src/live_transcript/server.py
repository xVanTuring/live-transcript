"""FastAPI WebSocket server for real-time speech recognition."""

from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .asr.base import CorrectionEngine, StreamingEngine
from .asr.pipeline import ASRPipeline, PipelineConfig
from .audio_buffer import pcm_s16le_to_float32
from .protocol import ErrorEvent, MessageType, StartConfig, TranscriptEvent, parse_client_message

logger = logging.getLogger(__name__)

app = FastAPI(title="Live Transcript", version="0.1.0")

# These are set by main.py after engine initialization
_streaming_engine: StreamingEngine | None = None
_correction_engine: CorrectionEngine | None = None
_app_config: dict = {}


def configure(
    streaming_engine: StreamingEngine,
    correction_engine: CorrectionEngine,
    config: dict,
) -> None:
    global _streaming_engine, _correction_engine, _app_config
    _streaming_engine = streaming_engine
    _correction_engine = correction_engine
    _app_config = config


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "streaming_engine": _streaming_engine is not None,
        "correction_engine": _correction_engine is not None,
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connection accepted")

    pipeline: ASRPipeline | None = None

    try:
        # Wait for start message
        start_data = await ws.receive_text()
        msg_type, data = parse_client_message(start_data)

        if msg_type != MessageType.START:
            await ws.send_text(ErrorEvent(
                code="EXPECTED_START",
                message="First message must be a 'start' message",
            ).to_json())
            await ws.close()
            return

        client_config = StartConfig.from_dict(data.get("config", {}))
        proto_config = _app_config.get("protocol", {})
        audio_config = _app_config.get("audio", {})

        pipeline_config = PipelineConfig(
            sample_rate=client_config.sample_rate,
            enable_correction=client_config.enable_correction,
            debounce_ms=proto_config.get("debounce_partial_ms", 100),
            ring_buffer_seconds=audio_config.get("ring_buffer_seconds", 60),
        )

        pipeline = ASRPipeline(
            streaming_engine=_streaming_engine,
            correction_engine=_correction_engine,
            config=pipeline_config,
        )

        # Send ready confirmation
        await ws.send_text(json.dumps({
            "type": "ready",
            "config": {
                "sample_rate": client_config.sample_rate,
                "enable_correction": client_config.enable_correction,
            },
        }))

        logger.info(
            "Session started: sample_rate=%d correction=%s",
            client_config.sample_rate,
            client_config.enable_correction,
        )

        # Main receive loop
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                # Binary frame: audio data
                recv_ts = time.time()
                samples = pcm_s16le_to_float32(message["bytes"])
                events = await pipeline.feed_audio(samples, client_audio_ts=recv_ts)
                send_ts = time.time()
                for event in events:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(event.to_json())
                if events:
                    total_ms = (send_ts - recv_ts) * 1000
                    logger.debug(
                        "Chunk → %d events in %.1fms",
                        len(events), total_ms,
                    )

            elif "text" in message and message["text"]:
                # Text frame: control message
                msg_type, data = parse_client_message(message["text"])
                if msg_type == MessageType.STOP:
                    # Flush remaining audio
                    final = await pipeline.flush()
                    if final and ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(final.to_json())
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("Error in WebSocket session")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(ErrorEvent(
                code="INTERNAL_ERROR",
                message="An internal error occurred",
            ).to_json())
    finally:
        if pipeline:
            # Flush on unexpected disconnect
            try:
                final = await pipeline.flush()
                if final and ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(final.to_json())
            except Exception:
                pass
            pipeline.close()

        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()
        logger.info("Session ended")
