from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class DecisionState(str, Enum):
    """
    Decision states for semantic buffering.
    """

    HOLD = "HOLD"
    RELEASE = "RELEASE"
    FORCE_RELEASE = "FORCE_RELEASE"


@dataclass
class ASRChunk:
    """
    Single ASR output chunk.

    Example:
    {
        "text": "Although the model is small,",
        "start_time": 0.0,
        "end_time": 1.2,
        "confidence": 0.91,
        "silence_after": 0.25
    }
    """

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    silence_after: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class BufferState:
    """
    Current state of semantic buffer.
    """

    chunks: List[ASRChunk] = field(default_factory=list)

    @property
    def start_time(self) -> float:
        if not self.chunks:
            return 0.0
        return self.chunks[0].start_time

    @property
    def end_time(self) -> float:
        if not self.chunks:
            return 0.0
        return self.chunks[-1].end_time

    @property
    def total_duration(self) -> float:
        if not self.chunks:
            return 0.0
        return self.end_time - self.start_time

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def text(self) -> str:
        return " ".join(
            chunk.text.strip()
            for chunk in self.chunks
            if chunk.text.strip()
        )