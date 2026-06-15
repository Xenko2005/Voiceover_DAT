import logging
from typing import Dict
from typing import Generator
from typing import Iterable

from .buffer_manager import BufferManager
from .chunk_models import ASRChunk
from .chunk_models import DecisionState
from .decision_engine import DecisionEngine
from .semantic_checker import SemanticChecker

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Main orchestration class.

    Input:
        Streaming ASR chunks

    Output:
        Semantic grouped chunks
    """

    def __init__(
        self,
        silence_threshold: float = 0.50,
        force_duration_threshold: float = 8.0,
        force_chunk_threshold: int = 10,
    ) -> None:

        self.buffer = BufferManager()

        self.semantic_checker = SemanticChecker()

        self.decision_engine = DecisionEngine(
            semantic_checker=self.semantic_checker,
            silence_threshold=silence_threshold,
            force_duration_threshold=force_duration_threshold,
            force_chunk_threshold=force_chunk_threshold,
        )

    def process_stream(
        self,
        chunk_stream: Iterable[ASRChunk],
    ) -> Generator[Dict, None, None]:

        for chunk in chunk_stream:

            try:
                self.buffer.append(chunk)

                decision = self.decision_engine.evaluate(
                    self.buffer
                )

                if decision in (
                    DecisionState.RELEASE,
                    DecisionState.FORCE_RELEASE,
                ):
                    yield self._release_buffer(
                        force=(
                            decision ==
                            DecisionState.FORCE_RELEASE
                        )
                    )

            except Exception as exc:
                logger.exception(
                    "Chunk processing error: %s",
                    exc,
                )

        if not self.buffer.is_empty():
            yield self._release_buffer(force=True)

    def process_chunk(
        self,
        chunk: ASRChunk,
    ) -> Dict | None:

        self.buffer.append(chunk)

        decision = self.decision_engine.evaluate(
            self.buffer
        )

        if decision in (
            DecisionState.RELEASE,
            DecisionState.FORCE_RELEASE,
        ):
            return self._release_buffer(
                force=(
                    decision ==
                    DecisionState.FORCE_RELEASE
                )
            )

        return None
    
    def flush(self) -> Dict | None:
        """
        Force release remaining buffered chunks.

        Used when stream ends.
        """

        if self.buffer.is_empty():
            return None

        return self._release_buffer(force=True)

    def _release_buffer(
        self,
        force: bool = False,
    ) -> Dict:

        text = self.buffer.get_buffered_text()

        output = {
            "text": text,
            "start_time": self.buffer.first_timestamp(),
            "end_time": self.buffer.last_timestamp(),
            "duration": self.buffer.get_total_duration(),
            "forced": force,
            "semantic_score":
                self.semantic_checker.score(text),
            "chunk_count": len(self.buffer),
        }

        logger.info(
            "Release buffer | text=%s",
            text,
        )

        self.buffer.clear()

        return output