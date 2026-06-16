import re
import json
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class ASRChunk:
    chunk_id: int
    text: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class SemanticSegment:
    segment_id: int
    source_text: str
    start_time: float
    end_time: float
    duration: float
    source_chunk_ids: List[int]
    num_chunks: int
    release_mode: str
    release_reason: str


class SemanticChunkBuffer:
    def __init__(
        self,
        min_words: int = 4,
        max_words: int = 45,
        max_duration: float = 12.0,
        max_chunks: int = 6,
        use_ollama_judge: bool = False,
        ollama_model: str = "qwen2.5:3b",
    ):
        """
        min_words:
            Số từ tối thiểu để xem xét release nếu có dấu câu mạnh.

        max_words:
            Nếu buffer quá dài thì force release.

        max_duration:
            Nếu duration buffer quá dài thì force release.

        max_chunks:
            Nếu gom quá nhiều ASR chunks thì force release.

        use_ollama_judge:
            True = dùng Ollama kiểm tra câu đã đủ nghĩa chưa.
            False = dùng rule-based để chạy nhanh hơn.

        ollama_model:
            Model Ollama dùng nếu bật use_ollama_judge.
        """
        self.min_words = min_words
        self.max_words = max_words
        self.max_duration = max_duration
        self.max_chunks = max_chunks
        self.use_ollama_judge = use_ollama_judge
        self.ollama_model = ollama_model

        self.weak_end_words = {
            "a", "an", "the",
            "and", "or", "but", "so",
            "because", "although", "though", "while", "when", "if", "unless",
            "that", "which", "who", "whom", "whose",
            "to", "of", "for", "with", "without", "from", "in", "on", "at", "by",
            "as", "than", "then", "into", "about", "over", "under",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had",
            "do", "does", "did",
            "can", "could", "would", "should", "will", "may", "might", "must",
        }

        self.strong_punctuation = {".", "?", "!"}
        self.weak_punctuation = {",", ";", ":"}

    # =====================================================
    # Basic text utilities
    # =====================================================

    def normalize_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def merge_texts(self, chunks: List[ASRChunk]) -> str:
        parts = [self.normalize_text(c.text) for c in chunks if c.text.strip()]
        text = " ".join(parts)
        text = re.sub(r"\s+([,.?!;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def count_words(self, text: str) -> int:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        return len(words)

    def get_last_word(self, text: str) -> str:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        if not words:
            return ""
        return words[-1].lower()

    def has_strong_punctuation_end(self, text: str) -> bool:
        text = text.strip()
        return len(text) > 0 and text[-1] in self.strong_punctuation

    def has_weak_punctuation_end(self, text: str) -> bool:
        text = text.strip()
        return len(text) > 0 and text[-1] in self.weak_punctuation

    def ends_with_weak_word(self, text: str) -> bool:
        last_word = self.get_last_word(text)
        return last_word in self.weak_end_words

    def has_unclosed_quote_or_bracket(self, text: str) -> bool:
        quote_count_1 = text.count('"')
        quote_count_2 = text.count("'")
        open_paren = text.count("(")
        close_paren = text.count(")")

        if quote_count_1 % 2 != 0:
            return True

        if open_paren > close_paren:
            return True

        return False

    def starts_with_lowercase_continuation(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False

        first_char = text[0]
        return first_char.islower()

    # =====================================================
    # Ollama semantic judge
    # =====================================================

    def call_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.ollama_model

        url = "http://localhost:11434/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            }
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()

        return response.json().get("response", "").strip()

    def semantic_complete_by_ollama(self, text: str) -> bool:
        """
        Dùng Ollama để kiểm tra đoạn hiện tại đã đủ nghĩa chưa.
        Output mong muốn: YES hoặc NO.
        """
        prompt = f"""
You are a semantic boundary detector for English ASR chunks.

Decide whether the following text is semantically complete enough to be translated as one segment.

Return only YES or NO.

Guidelines:
- YES if the text expresses a complete idea or complete sentence.
- NO if it ends abruptly, depends on missing following words, or is only a fragment.
- A comma alone is usually not enough for YES.
- If the text ends with because/although/if/when/while/that/to/of/and/but, return NO.

Text:
{text}

Answer:
""".strip()

        try:
            raw = self.call_ollama(prompt)
            raw = raw.strip().upper()
            return raw.startswith("YES")
        except Exception as e:
            print("[WARN] Ollama semantic judge failed:", e)
            return self.semantic_complete_by_rules(text)

    # =====================================================
    # Rule-based semantic completeness
    # =====================================================

    def semantic_complete_by_rules(self, text: str) -> bool:
        """
        Rule-based semantic completeness.
        Không hoàn hảo, nhưng đủ tốt cho prototype.
        """
        text = self.normalize_text(text)

        if not text:
            return False

        word_count = self.count_words(text)

        if word_count < self.min_words:
            return False

        if self.has_unclosed_quote_or_bracket(text):
            return False

        if self.ends_with_weak_word(text):
            return False

        # Dấu câu mạnh thường là tín hiệu release tốt.
        if self.has_strong_punctuation_end(text):
            return True

        # Dấu phẩy/chấm phẩy thường chưa đủ, trừ khi câu đã khá dài.
        if self.has_weak_punctuation_end(text):
            return word_count >= 18 and not self.ends_with_weak_word(text)

        # Nếu không có dấu câu mạnh nhưng câu đủ dài và không kết thúc bằng từ yếu,
        # có thể release để tránh giữ buffer quá lâu.
        if word_count >= 20:
            return True

        return False

    def is_semantically_complete(self, text: str) -> bool:
        if self.use_ollama_judge:
            return self.semantic_complete_by_ollama(text)
        return self.semantic_complete_by_rules(text)

    # =====================================================
    # Buffer decision
    # =====================================================

    def should_force_release(self, buffer_chunks: List[ASRChunk]) -> Dict:
        text = self.merge_texts(buffer_chunks)
        word_count = self.count_words(text)
        duration = buffer_chunks[-1].end_time - buffer_chunks[0].start_time

        if word_count >= self.max_words:
            return {
                "force": True,
                "reason": f"word_count >= max_words ({word_count} >= {self.max_words})"
            }

        if duration >= self.max_duration:
            return {
                "force": True,
                "reason": f"duration >= max_duration ({duration:.2f}s >= {self.max_duration:.2f}s)"
            }

        if len(buffer_chunks) >= self.max_chunks:
            return {
                "force": True,
                "reason": f"num_chunks >= max_chunks ({len(buffer_chunks)} >= {self.max_chunks})"
            }

        return {
            "force": False,
            "reason": ""
        }

    def decide_release(self, buffer_chunks: List[ASRChunk]) -> Dict:
        text = self.merge_texts(buffer_chunks)

        force_info = self.should_force_release(buffer_chunks)
        if force_info["force"]:
            return {
                "release": True,
                "mode": "FORCE_RELEASE",
                "reason": force_info["reason"]
            }

        if self.is_semantically_complete(text):
            return {
                "release": True,
                "mode": "RELEASE",
                "reason": "semantic_complete"
            }

        return {
            "release": False,
            "mode": "HOLD",
            "reason": "semantic_incomplete"
        }

    # =====================================================
    # Main processing
    # =====================================================

    def create_segment(
        self,
        segment_id: int,
        buffer_chunks: List[ASRChunk],
        release_mode: str,
        release_reason: str
    ) -> SemanticSegment:
        source_text = self.merge_texts(buffer_chunks)
        start_time = buffer_chunks[0].start_time
        end_time = buffer_chunks[-1].end_time
        duration = max(0.0, end_time - start_time)
        source_chunk_ids = [c.chunk_id for c in buffer_chunks]

        return SemanticSegment(
            segment_id=segment_id,
            source_text=source_text,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            source_chunk_ids=source_chunk_ids,
            num_chunks=len(buffer_chunks),
            release_mode=release_mode,
            release_reason=release_reason,
        )

    def process_chunks(self, chunks: List[ASRChunk]) -> List[SemanticSegment]:
        """
        Xử lý tuần tự như streaming:
        đọc từng ASR chunk → đưa vào buffer → quyết định HOLD/RELEASE.
        """
        segments = []
        buffer_chunks = []
        segment_id = 1

        for chunk in chunks:
            if not chunk.text.strip():
                continue

            buffer_chunks.append(chunk)

            decision = self.decide_release(buffer_chunks)

            if decision["release"]:
                segment = self.create_segment(
                    segment_id=segment_id,
                    buffer_chunks=buffer_chunks,
                    release_mode=decision["mode"],
                    release_reason=decision["reason"],
                )

                segments.append(segment)
                segment_id += 1
                buffer_chunks = []

        # Nếu còn buffer cuối cùng thì release.
        if buffer_chunks:
            segment = self.create_segment(
                segment_id=segment_id,
                buffer_chunks=buffer_chunks,
                release_mode="FINAL_RELEASE",
                release_reason="end_of_input",
            )
            segments.append(segment)

        return segments

    # =====================================================
    # Export helpers
    # =====================================================

    def segments_to_dicts(self, segments: List[SemanticSegment]) -> List[Dict]:
        return [asdict(seg) for seg in segments]