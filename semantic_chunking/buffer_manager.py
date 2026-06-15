import logging
from typing import List

from .chunk_models import ASRChunk, BufferState

logger = logging.getLogger(__name__)


class BufferManager:
    """
    Manage chunk buffering.
    """

    def __init__(self) -> None:
        self._chunks: List[ASRChunk] = []

    def append(self, chunk: ASRChunk) -> None:
        """
        Add chunk into buffer.
        """
        if not isinstance(chunk, ASRChunk):
            raise TypeError("chunk must be ASRChunk")

        self._chunks.append(chunk)

        logger.debug(
            "Chunk appended | text=%s | buffer_size=%d",
            chunk.text,
            len(self._chunks),
        )

    def clear(self) -> None:
        """
        Clear buffer.
        """
        logger.debug("Buffer cleared")
        self._chunks.clear()

    def get_chunks(self) -> List[ASRChunk]:
        return list(self._chunks)

    def get_state(self) -> BufferState:
        return BufferState(chunks=self.get_chunks())

    def get_buffered_text(self) -> str:
        return " ".join(
            chunk.text.strip()
            for chunk in self._chunks
            if chunk.text.strip()
        )

    def get_total_duration(self) -> float:
        if not self._chunks:
            return 0.0

        return (
            self._chunks[-1].end_time -
            self._chunks[0].start_time
        )

    def first_timestamp(self) -> float:
        if not self._chunks:
            return 0.0
        return self._chunks[0].start_time

    def last_timestamp(self) -> float:
        if not self._chunks:
            return 0.0
        return self._chunks[-1].end_time

    def __len__(self) -> int:
        return len(self._chunks)

    def is_empty(self) -> bool:
        return len(self._chunks) == 0