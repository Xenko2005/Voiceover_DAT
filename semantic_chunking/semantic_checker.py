import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticChecker:
    """
    Rule-based semantic completeness checker.

    Future upgrade:
    - SpaCy dependency parser
    - LLM classifier
    """

    ENDING_PUNCTUATION = (".", "!", "?")

    CONNECTOR_WORDS = {
        "although",
        "because",
        "while",
        "when",
        "if",
        "since",
        "unless",
        "whereas",
        "after",
        "before",
        "until",
        "though",
    }

    COMMON_VERBS = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "perform",
        "performs",
        "performed",
        "translate",
        "translates",
        "translated",
        "run",
        "runs",
        "running",
        "work",
        "works",
        "worked",
    }

    def __init__(self) -> None:
        pass

    def looks_complete(self, text: str) -> bool:
        """
        Main semantic completeness check.
        """

        if not text or not text.strip():
            return False

        normalized = text.strip()

        if self._ends_with_terminal_punctuation(normalized):
            return True

        if self._starts_with_connector_only(normalized):
            return False

        if self._has_subject_predicate_pattern(normalized):
            return True

        return False

    def _ends_with_terminal_punctuation(self, text: str) -> bool:
        return text.endswith(self.ENDING_PUNCTUATION)

    def _starts_with_connector_only(self, text: str) -> bool:
        words = text.lower().split()

        if not words:
            return False

        first_word = re.sub(r"[^\w]", "", words[0])

        if first_word not in self.CONNECTOR_WORDS:
            return False

        if len(words) < 6:
            return True

        return False

    def _has_subject_predicate_pattern(self, text: str) -> bool:
        words = text.lower().split()

        if len(words) < 3:
            return False

        for token in words:
            token = re.sub(r"[^\w]", "", token)

            if token in self.COMMON_VERBS:
                return True

        return False

    def score(self, text: str) -> float:
        """
        Return confidence score from 0-1.
        """

        if not text:
            return 0.0

        score = 0.0

        if self._has_subject_predicate_pattern(text):
            score += 0.5

        if self._ends_with_terminal_punctuation(text):
            score += 0.3

        if len(text.split()) >= 5:
            score += 0.2

        return min(score, 1.0)