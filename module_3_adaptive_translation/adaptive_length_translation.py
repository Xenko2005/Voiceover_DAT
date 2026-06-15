import re
import json
import math
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional


# =========================
# Data classes
# =========================

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


# =========================
# Adaptive Length Translator
# =========================

class AdaptiveLengthTranslator:
    def __init__(
        self,
        tts_ceiling: float = 5.0,
        margin: int = 1,
        lambda_dur: float = 0.6,
        mu_flu: float = 0.3,
        ollama_model: str = "qwen2.5:3b",
        use_llm_judge: bool = False,
    ):
        """
        tts_ceiling:
            Số âm tiết tiếng Việt tối đa TTS có thể đọc rõ trong 1 giây.

        margin:
            Biên an toàn cho độ dài tiếng Việt.

        lambda_dur:
            Hệ số phạt lỗi duration.

        mu_flu:
            Hệ số phạt câu thiếu tự nhiên.

        ollama_model:
            Model Ollama dùng để dịch / tạo context.

        use_llm_judge:
            True = dùng Ollama chấm SemScore.
            False = dùng heuristic demo để chạy nhanh hơn.
        """
        self.tts_ceiling = tts_ceiling
        self.margin = margin
        self.lambda_dur = lambda_dur
        self.mu_flu = mu_flu
        self.ollama_model = ollama_model
        self.use_llm_judge = use_llm_judge

    # =====================================================
    # Basic utilities
    # =====================================================

    def count_english_words(self, text: str) -> int:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        return len(words)

    def count_english_chars(self, text: str) -> int:
        return len(re.sub(r"\s+", "", text))

    def count_vietnamese_units(self, text: str) -> int:
        """
        Đếm đơn vị tiếng Việt theo khoảng trắng.
        Trong tiếng Việt, mỗi âm tiết thường được tách bằng khoảng trắng.

        Ví dụ:
        'Mô hình này dịch giọng nói thời gian thực'
        -> Mô / hình / này / dịch / giọng / nói / thời / gian / thực
        -> 9 đơn vị.
        """
        clean = re.sub(r"[^\w\sÀ-ỹ]", "", text, flags=re.UNICODE)
        units = [u for u in clean.split() if u.strip()]
        return len(units)

    # =====================================================
    # Ollama
    # =====================================================

    def call_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Gọi Ollama local.
        Cần chạy trước:
            ollama serve

        Kiểm tra model:
            ollama list
        """
        model = model or self.ollama_model

        url = "http://localhost:11434/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
            }
        }

        response = requests.post(url, json=payload, timeout=240)
        response.raise_for_status()

        return response.json().get("response", "").strip()

    def strip_code_fence(self, text: str) -> str:
        """
        Nếu LLM trả về ```json ... ```, hàm này bóc phần code fence.
        """
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        return text.strip()

    def parse_candidates_from_llm(self, raw_text: str) -> List[str]:
        """
        Parse output từ LLM thành list candidate.

        Hỗ trợ:
        1. JSON list:
           ["...", "...", "..."]

        2. Dạng đánh số:
           1. ...
           2. ...
           3. ...
        """
        raw_text = self.strip_code_fence(raw_text)

        # Cách 1: parse JSON list
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                candidates = [str(x).strip() for x in data if str(x).strip()]
                return candidates[:3]
        except Exception:
            pass

        # Cách 2: tìm JSON list nằm bên trong text
        try:
            match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    candidates = [str(x).strip() for x in data if str(x).strip()]
                    return candidates[:3]
        except Exception:
            pass

        # Cách 3: parse từng dòng
        candidates = []

        for line in raw_text.splitlines():
            line = line.strip()

            if not line:
                continue

            line = re.sub(r"^\s*[\-\*\•]\s*", "", line)
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
            line = line.strip().strip('"').strip("'").strip()

            if line:
                candidates.append(line)

        # Loại trùng
        unique = []
        for cand in candidates:
            if cand not in unique:
                unique.append(cand)

        return unique[:3]

    # =====================================================
    # Context Builder
    # =====================================================

    def build_document_context_with_ollama(
        self,
        full_transcript: str,
        model: Optional[str] = None
    ) -> str:
        """
        Đọc toàn bộ transcript và tạo Context Pack.
        Context Pack này sẽ được dùng khi dịch từng segment.
        """

        model = model or self.ollama_model

        max_chars = 9000
        transcript = full_transcript

        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "\n...[TRUNCATED]"

        prompt = f"""
Bạn là hệ thống phân tích ngữ cảnh cho bài toán dịch Anh-Việt speech-to-speech.

Hãy đọc transcript tiếng Anh dưới đây và tạo CONTEXT PACK ngắn gọn để hỗ trợ dịch từng segment.

Yêu cầu CONTEXT PACK gồm:
1. Chủ đề chính của bài.
2. Bối cảnh / tình huống.
3. Nhân vật, ngôi xưng hô, quan hệ giữa các đối tượng nếu có.
4. Văn phong nên dùng khi dịch sang tiếng Việt.
5. Các thuật ngữ, cụm từ, tên riêng quan trọng cần dịch nhất quán.
6. Những lưu ý giúp dịch các câu ngắn/câu cụt đúng ngữ cảnh.

Transcript:
{transcript}

CONTEXT PACK bằng tiếng Việt:
""".strip()

        try:
            context_pack = self.call_ollama(prompt, model=model)
            return context_pack.strip()
        except Exception as e:
            print("[WARN] Không tạo được Document Context Pack bằng Ollama:", e)
            print("[WARN] Sẽ dùng transcript rút gọn làm context.")
            return transcript[:3000]

    # =====================================================
    # Feature extraction
    # =====================================================

    def extract_features(self, segment: SourceSegment) -> Dict:
        text = segment.source_text
        lower = text.lower()

        wen = self.count_english_words(text)
        cen = self.count_english_chars(text)
        speech_rate = wen / max(segment.duration, 1e-6)

        has_subordinate = any(
            x in lower
            for x in [
                "although", "because", "while", "if", "when",
                "that", "unless", "before", "after", "since"
            ]
        )

        is_question = "?" in text or lower.startswith(
            ("what", "why", "how", "when", "where", "who",
             "do", "does", "did", "can", "could", "would", "should")
        )

        has_number = bool(re.search(r"\d+", text))

        # Tên riêng đơn giản: từ viết hoa không nằm đầu câu.
        proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", text)
        has_proper_noun = len(proper_nouns) > 1

        return {
            "wen": wen,
            "cen": cen,
            "duration": segment.duration,
            "speech_rate": speech_rate,
            "has_subordinate": has_subordinate,
            "is_question": is_question,
            "has_number": has_number,
            "has_proper_noun": has_proper_noun,
        }

    # =====================================================
    # PredLenVI and MaxVI_final
    # =====================================================

    def estimate_dynamic_k(self, features: Dict) -> float:
        """
        Rule-based k_dynamic.
        Sau này có thể thay bằng regression model.
        """
        k = 1.25

        wen = features["wen"]
        speech_rate = features["speech_rate"]

        # Câu rất ngắn thường khi dịch sang tiếng Việt dễ nở hơn
        if wen <= 3:
            k += 0.20

        # Câu có mệnh đề phụ thường cần dài hơn để rõ nghĩa
        if features["has_subordinate"]:
            k += 0.15

        # Câu hỏi thường cần thêm cụm từ hỏi trong tiếng Việt
        if features["is_question"]:
            k += 0.10

        # Có số / tên riêng thường giữ nguyên, ít nở hơn
        if features["has_number"]:
            k -= 0.05

        if features["has_proper_noun"]:
            k -= 0.05

        # Người nói quá nhanh thì phải siết độ dài
        if speech_rate > 3.5:
            k -= 0.10

        return max(0.9, min(k, 1.7))

    def predict_len_vi(self, segment: SourceSegment) -> int:
        features = self.extract_features(segment)
        k_dynamic = self.estimate_dynamic_k(features)
        pred_len = round(features["wen"] * k_dynamic)
        return max(pred_len, 1)

    def calculate_max_vi_final(self, segment: SourceSegment, pred_len_vi: int) -> int:
        physical_limit = segment.duration * self.tts_ceiling
        max_vi_final = min(pred_len_vi + self.margin, physical_limit)
        return max(1, math.floor(max_vi_final))

    # =====================================================
    # Translation prompt
    # =====================================================

    def build_context_aware_translation_prompt(
        self,
        segment: SourceSegment,
        max_vi_final: int,
        global_context: str = "",
        previous_context: str = "",
        next_context: str = "",
    ) -> str:
        """
        Prompt dịch có:
        - Global context toàn bài
        - Previous segments
        - Current segment
        - Next segments
        - Ràng buộc độ dài
        """

        max_context_chars = 4500
        if len(global_context) > max_context_chars:
            global_context = global_context[:max_context_chars] + "\n...[TRUNCATED]"

        prompt = f"""
Bạn là hệ thống dịch Anh sang Việt cho bài toán speech-to-speech / voice-over.

Nhiệm vụ:
Dịch ONLY CURRENT SEGMENT sang tiếng Việt tự nhiên, đúng ngữ cảnh toàn bài,
đồng thời phải ngắn gọn để đọc bằng TTS trong thời lượng gốc.

GLOBAL CONTEXT PACK:
{global_context if global_context else "(Không có context toàn bài)"}

PREVIOUS SEGMENTS:
{previous_context if previous_context else "(Không có)"}

CURRENT SEGMENT:
{segment.source_text}

NEXT SEGMENTS:
{next_context if next_context else "(Không có)"}

Ràng buộc dịch:
- Chỉ dịch CURRENT SEGMENT, không dịch previous/next/global context.
- Giữ đúng ý chính của CURRENT SEGMENT.
- Dùng GLOBAL CONTEXT để chọn từ ngữ phù hợp.
- Dùng PREVIOUS/NEXT SEGMENTS để xử lý đại từ, câu cụt, câu nối ý.
- Không thêm thông tin mới.
- Không giải thích.
- Không ghi chú.
- Không dùng tiếng Anh trong bản dịch, trừ thuật ngữ bắt buộc.
- Mỗi bản dịch không vượt quá khoảng {max_vi_final} âm tiết tiếng Việt.
- Sinh đúng 3 bản dịch khác nhau.
- Trả về JSON list hợp lệ, không kèm text ngoài JSON.

Ví dụ format:
["Bản dịch 1", "Bản dịch 2", "Bản dịch 3"]
""".strip()

        return prompt

    def generate_candidates_with_ollama(
        self,
        segment: SourceSegment,
        max_vi_final: int,
        global_context: str = "",
        previous_context: str = "",
        next_context: str = "",
        model: Optional[str] = None,
    ) -> List[str]:
        model = model or self.ollama_model

        prompt = self.build_context_aware_translation_prompt(
            segment=segment,
            max_vi_final=max_vi_final,
            global_context=global_context,
            previous_context=previous_context,
            next_context=next_context,
        )

        try:
            raw_response = self.call_ollama(prompt, model=model)
            candidates = self.parse_candidates_from_llm(raw_response)

            if len(candidates) == 0:
                print("[WARN] LLM không trả về candidate hợp lệ.")
                print("[RAW RESPONSE]", raw_response)
                return self.fallback_generate_candidates(segment)

            return candidates[:3]

        except Exception as e:
            print("[WARN] Lỗi khi gọi Ollama để dịch:", e)
            return self.fallback_generate_candidates(segment)

    def fallback_generate_candidates(self, segment: SourceSegment) -> List[str]:
        """
        Fallback nếu Ollama lỗi.
        Không dùng cho kết quả nghiên cứu chính.
        """
        return [
            f"[CẦN DỊCH] {segment.source_text}",
            f"[BẢN NGẮN] {segment.source_text}",
            f"[BẢN RÚT GỌN] {segment.source_text}",
        ]

    # =====================================================
    # Scoring
    # =====================================================

    def estimate_duration_error_by_units(self, vi_units: int, max_vi_final: int) -> float:
        """
        Ước lượng DurError bằng số âm tiết.
        Sau khi có TTS thật, nên thay bằng:
            DurError = abs(DurTTS - DurSrc) / DurSrc
        """
        return abs(vi_units - max_vi_final) / max(max_vi_final, 1)

    def fluency_penalty_rule_based(self, text_vi: str) -> float:
        """
        Rule-based FluPenalty:
        0 = rất tự nhiên
        1 = rất tệ
        """
        penalty = 0.0
        units = self.count_vietnamese_units(text_vi)
        tokens = text_vi.lower().split()

        if units <= 4:
            penalty += 0.20

        if units >= 22:
            penalty += 0.15

        # Lặp từ liên tục
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                penalty += 0.25

        # Kết thúc bằng từ nối nghe cụt
        risky_endings = ["vì", "nhưng", "và", "để", "rằng", "nếu", "khi", "mà"]
        if tokens and tokens[-1] in risky_endings:
            penalty += 0.30

        # Một số cụm nghe không tự nhiên
        unnatural_patterns = [
            "dịch nói",
            "thời gian thật",
            "giọng tức",
            "[cần dịch]",
            "[bản ngắn]",
            "[bản rút gọn]",
        ]

        if any(p in text_vi.lower() for p in unnatural_patterns):
            penalty += 0.40

        return min(penalty, 1.0)

    def semantic_score_heuristic(self, source_text: str, candidate_vi: str) -> float:
        """
        Heuristic đơn giản khi chưa dùng LLM judge.
        Vì không hiểu nghĩa sâu, hàm này chỉ là bản chạy nhanh.
        """
        candidate_lower = candidate_vi.lower()

        if "[cần dịch]" in candidate_lower or "[bản" in candidate_lower:
            return 0.30

        vi_units = self.count_vietnamese_units(candidate_vi)

        if vi_units <= 3:
            return 0.45

        if vi_units <= 6:
            return 0.70

        return 0.82

    def semantic_score_with_ollama(
        self,
        source_text: str,
        candidate_vi: str,
        global_context: str = "",
        model: Optional[str] = None,
    ) -> float:
        """
        Dùng Ollama để chấm mức giữ nghĩa.
        Chạy chậm hơn vì mỗi candidate cần gọi LLM.
        """
        model = model or self.ollama_model

        max_context_chars = 2500
        if len(global_context) > max_context_chars:
            global_context = global_context[:max_context_chars] + "\n...[TRUNCATED]"

        prompt = f"""
Bạn là bộ đánh giá chất lượng dịch Anh-Việt trong hệ thống speech-to-speech.

Hãy chấm điểm mức độ giữ nghĩa của bản dịch tiếng Việt so với câu tiếng Anh.
Có thể dùng context toàn bài để hiểu ngữ cảnh.

Chỉ trả về một số từ 0 đến 1. Không giải thích.

Quy ước:
1.0 = giữ nghĩa rất tốt
0.8 = giữ ý chính, mất rất ít chi tiết
0.5 = chỉ đúng một phần
0.0 = sai nghĩa

GLOBAL CONTEXT:
{global_context if global_context else "(Không có)"}

English:
{source_text}

Vietnamese:
{candidate_vi}

Score:
""".strip()

        try:
            raw = self.call_ollama(prompt, model=model)
            match = re.search(r"(\d+(\.\d+)?)", raw)
            if match:
                score = float(match.group(1))
                return max(0.0, min(score, 1.0))
        except Exception as e:
            print("[WARN] Lỗi semantic_score_with_ollama:", e)

        return self.semantic_score_heuristic(source_text, candidate_vi)

    def score_candidate(
        self,
        source_text: str,
        candidate_text: str,
        max_vi_final: int,
        global_context: str = "",
    ) -> TranslationCandidate:
        vi_units = self.count_vietnamese_units(candidate_text)

        if self.use_llm_judge:
            sem_score = self.semantic_score_with_ollama(
                source_text=source_text,
                candidate_vi=candidate_text,
                global_context=global_context,
                model=self.ollama_model,
            )
        else:
            sem_score = self.semantic_score_heuristic(source_text, candidate_text)

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

    # =====================================================
    # Main API
    # =====================================================

    def translate_and_select(
        self,
        segment: SourceSegment,
        global_context: str = "",
        previous_context: str = "",
        next_context: str = "",
    ) -> Dict:
        pred_len_vi = self.predict_len_vi(segment)
        max_vi_final = self.calculate_max_vi_final(segment, pred_len_vi)

        candidates = self.generate_candidates_with_ollama(
            segment=segment,
            max_vi_final=max_vi_final,
            global_context=global_context,
            previous_context=previous_context,
            next_context=next_context,
            model=self.ollama_model,
        )

        scored_candidates = [
            self.score_candidate(
                source_text=segment.source_text,
                candidate_text=cand,
                max_vi_final=max_vi_final,
                global_context=global_context,
            )
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
            "context_used": {
                "previous_context": previous_context,
                "next_context": next_context,
                "global_context_preview": global_context[:800],
            },
        }