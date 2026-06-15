import re
import math
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SourceSegment:
    segment_id: int
    source_text: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class TranslationCandidate:
    text_vi: str
    sem_score: float
    dur_error: float
    flu_penalty: float
    final_score: float
    vi_units: int


class AdaptiveLengthTranslator:
    def __init__(
        self,
        tts_ceiling: float = 5.0,
        margin: int = 1,
        lambda_dur: float = 0.6,
        mu_flu: float = 0.3,
    ):
        self.tts_ceiling = tts_ceiling
        self.margin = margin
        self.lambda_dur = lambda_dur
        self.mu_flu = mu_flu

    def count_english_words(self, text: str) -> int:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        return len(words)

    def count_vietnamese_units(self, text: str) -> int:
        clean = re.sub(r"[^\w\sÀ-ỹ]", "", text, flags=re.UNICODE)
        units = [u for u in clean.split() if u.strip()]
        return len(units)

    def extract_features(self, segment: SourceSegment) -> Dict:
        text = segment.source_text
        lower = text.lower()

        wen = self.count_english_words(text)
        speech_rate = wen / max(segment.duration, 1e-6)

        has_subordinate = any(
            x in lower
            for x in ["although", "because", "while", "if", "when", "that"]
        )

        is_question = "?" in text or lower.startswith(
            ("what", "why", "how", "when", "where", "who", "do", "does", "did", "can")
        )

        has_number = bool(re.search(r"\d+", text))

        return {
            "wen": wen,
            "duration": segment.duration,
            "speech_rate": speech_rate,
            "has_subordinate": has_subordinate,
            "is_question": is_question,
            "has_number": has_number,
        }

    def estimate_dynamic_k(self, features: Dict) -> float:
        k = 1.25
        wen = features["wen"]
        speech_rate = features["speech_rate"]

        if wen <= 3:
            k += 0.20

        if features["has_subordinate"]:
            k += 0.15

        if features["is_question"]:
            k += 0.10

        if features["has_number"]:
            k -= 0.05

        if speech_rate > 3.5:
            k -= 0.10

        return max(0.9, min(k, 1.7))

    def predict_len_vi(self, segment: SourceSegment) -> int:
        features = self.extract_features(segment)
        k = self.estimate_dynamic_k(features)
        pred_len = round(features["wen"] * k)
        return max(pred_len, 1)

    def calculate_max_vi_final(self, segment: SourceSegment, pred_len_vi: int) -> int:
        physical_limit = segment.duration * self.tts_ceiling
        max_vi_final = min(pred_len_vi + self.margin, physical_limit)
        return max(1, math.floor(max_vi_final))

    def build_translation_prompt(self, segment: SourceSegment, max_vi_final: int) -> str:
        return f"""
Dịch câu sau sang tiếng Việt tự nhiên, giữ đúng ý chính.

Yêu cầu:
- Bản dịch ngắn gọn, phù hợp để lồng tiếng.
- Không thêm thông tin mới.
- Không vượt quá khoảng {max_vi_final} âm tiết tiếng Việt.
- Ưu tiên câu dễ đọc bằng TTS.

English:
{segment.source_text}

Trả về 3 bản dịch tiếng Việt khác nhau:
1.
2.
3.
""".strip()

    def mock_generate_candidates(self, segment: SourceSegment) -> List[str]:
        """
        Bản demo tạm thời.
        Sau này thay bằng LLM/Ollama/API.
        """
        text = segment.source_text.lower()

        if "this model can translate speech in real time" in text:
            return [
                "Mô hình này có thể dịch lời nói trong thời gian thực.",
                "Mô hình này dịch giọng nói thời gian thực.",
                "Mô hình này dịch tức thời.",
            ]

        return [
            f"Bản dịch đầy đủ của câu: {segment.source_text}",
            "Bản dịch tiếng Việt ngắn gọn hơn.",
            "Bản dịch rút gọn.",
        ]

    def estimate_duration_error_by_units(self, vi_units: int, max_vi_final: int) -> float:
        return abs(vi_units - max_vi_final) / max(max_vi_final, 1)

    def fluency_penalty_rule_based(self, text_vi: str) -> float:
        penalty = 0.0
        units = self.count_vietnamese_units(text_vi)
        tokens = text_vi.lower().split()

        if units <= 4:
            penalty += 0.25

        if units >= 20:
            penalty += 0.15

        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                penalty += 0.25

        risky_endings = ["vì", "nhưng", "và", "để", "rằng", "nếu", "khi"]
        if tokens and tokens[-1] in risky_endings:
            penalty += 0.30

        unnatural_patterns = [
            "dịch nói",
            "thời gian thật",
            "giọng tức",
        ]

        if any(p in text_vi.lower() for p in unnatural_patterns):
            penalty += 0.30

        return min(penalty, 1.0)

    def semantic_score_demo(self, source_text: str, candidate_vi: str) -> float:
        """
        Bản demo. Sau này thay bằng LLM judge hoặc human evaluation.
        """
        lower = candidate_vi.lower()

        if "dịch lời nói" in lower or "dịch giọng nói" in lower:
            return 0.90

        if "dịch tức thời" in lower:
            return 0.70

        return 0.75

    def score_candidate(
        self,
        source_text: str,
        candidate_text: str,
        max_vi_final: int,
    ) -> TranslationCandidate:
        vi_units = self.count_vietnamese_units(candidate_text)

        sem_score = self.semantic_score_demo(source_text, candidate_text)
        dur_error = self.estimate_duration_error_by_units(vi_units, max_vi_final)
        flu_penalty = self.fluency_penalty_rule_based(candidate_text)

        final_score = (
            sem_score
            - self.lambda_dur * dur_error
            - self.mu_flu * flu_penalty
        )

        return TranslationCandidate(
            text_vi=candidate_text,
            sem_score=sem_score,
            dur_error=dur_error,
            flu_penalty=flu_penalty,
            final_score=final_score,
            vi_units=vi_units,
        )

    def translate_and_select(self, segment: SourceSegment) -> Dict:
        pred_len_vi = self.predict_len_vi(segment)
        max_vi_final = self.calculate_max_vi_final(segment, pred_len_vi)

        candidates = self.mock_generate_candidates(segment)

        scored_candidates = [
            self.score_candidate(segment.source_text, cand, max_vi_final)
            for cand in candidates
        ]

        best = max(scored_candidates, key=lambda x: x.final_score)

        return {
            "segment_id": segment.segment_id,
            "source_text": segment.source_text,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "duration": segment.duration,
            "pred_len_vi": pred_len_vi,
            "max_vi_final": max_vi_final,
            "best_translation": best.text_vi,
            "best_score": best.final_score,
            "candidates": [
                {
                    "text_vi": c.text_vi,
                    "vi_units": c.vi_units,
                    "sem_score": c.sem_score,
                    "dur_error": c.dur_error,
                    "flu_penalty": c.flu_penalty,
                    "final_score": c.final_score,
                }
                for c in scored_candidates
            ],
            "prompt": self.build_translation_prompt(segment, max_vi_final),
        }