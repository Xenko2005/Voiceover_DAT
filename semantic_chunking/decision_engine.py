import logging

from .buffer_manager import BufferManager
from .chunk_models import DecisionState
from .semantic_checker import SemanticChecker

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Core decision engine.

    Signals:
    - Silence
    - Punctuation
    - Semantic completeness
    - Buffer duration
    - Buffer size
    """

    def __init__(
        self,
        semantic_checker: SemanticChecker,
        silence_threshold: float = 0.50,
        force_duration_threshold: float = 8.0,
        force_chunk_threshold: int = 10,
    ) -> None:

        self.semantic_checker = semantic_checker
        self.silence_threshold = silence_threshold
        self.force_duration_threshold = force_duration_threshold
        self.force_chunk_threshold = force_chunk_threshold

    def evaluate(
        self,
        buffer_manager: BufferManager,
    ) -> DecisionState:

        text = buffer_manager.get_buffered_text()

        if not text:
            return DecisionState.HOLD

        duration = buffer_manager.get_total_duration()

        if (
            duration >= self.force_duration_threshold
            or len(buffer_manager) >= self.force_chunk_threshold
        ):
            logger.warning(
                "FORCE_RELEASE triggered | duration=%.2f",
                duration,
            )
            return DecisionState.FORCE_RELEASE

        chunks = buffer_manager.get_chunks()
        latest = chunks[-1]

        semantic_complete = (
            self.semantic_checker.looks_complete(text)
        )

        punctuation_release = text.endswith(
            (".", "!", "?")
        )

        silence_release = (
            latest.silence_after >= self.silence_threshold
        )

        if semantic_complete and (
            punctuation_release or silence_release
        ):
            return DecisionState.RELEASE

        return DecisionState.HOLD