"""WebSocket message protocol definitions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    # Client -> Server
    START = "start"
    STOP = "stop"
    # Server -> Client
    PARTIAL = "partial"
    CORRECTION = "correction"
    FINAL = "final"
    ERROR = "error"
    READY = "ready"


@dataclass
class StartConfig:
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    channels: int = 1
    language: str = "auto"
    enable_correction: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StartConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TranscriptEvent:
    type: MessageType
    segment_id: int = 0
    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = ""
    previous_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        # Remove empty optional fields
        for key in ("language", "previous_text"):
            if not d[key]:
                del d[key]
        return json.dumps(d, ensure_ascii=False)


@dataclass
class ErrorEvent:
    code: str
    message: str
    type: str = "error"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_client_message(raw: str) -> tuple[MessageType, dict[str, Any]]:
    data = json.loads(raw)
    msg_type = MessageType(data["type"])
    return msg_type, data


def make_partial(segment_id: int, text: str, start_time: float, end_time: float) -> TranscriptEvent:
    return TranscriptEvent(
        type=MessageType.PARTIAL,
        segment_id=segment_id,
        text=text,
        start_time=start_time,
        end_time=end_time,
    )


def make_correction(
    segment_id: int, text: str, previous_text: str, start_time: float, end_time: float
) -> TranscriptEvent:
    return TranscriptEvent(
        type=MessageType.CORRECTION,
        segment_id=segment_id,
        text=text,
        previous_text=previous_text,
        start_time=start_time,
        end_time=end_time,
    )


def make_final(
    segment_id: int, text: str, start_time: float, end_time: float, language: str = ""
) -> TranscriptEvent:
    return TranscriptEvent(
        type=MessageType.FINAL,
        segment_id=segment_id,
        text=text,
        start_time=start_time,
        end_time=end_time,
        language=language,
    )
