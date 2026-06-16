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
# Main translator
# =========================

class AdaptiveLengthTranslator:
    def __init__(
        self,
        tts_ceiling: float = 5.0,
        margin: int = 1,
        lambda_dur: float = 0.6,
        mu_flu: float = 0.3,
        ollama_model: str = "qwen3.5:2b",
        use_llm_judge: bool = False,
    ):
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

    def count_vietnamese_units(self, text: str) -> int:
        """
        Đếm đơn vị tiếng Việt theo khoảng trắng.
        Tiếng Việt thường tách âm tiết bằng dấu cách.
        """
        clean = re.sub(r"[^\w\sÀ-ỹ]", "", text, flags=re.UNICODE)
        units = [u for u in clean.split() if u.strip()]
        return len(units)

    # =====================================================
    # Ollama API
    # =====================================================

    def call_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Gọi Ollama local.

        Bản này xử lý riêng các model thinking như qwen3.5:2b:
        - Thêm /no_think vào prompt.
        - Gửi "think": False vào Ollama API.
        - Tăng num_predict để model còn đủ token trả lời.
        """

        model = model or self.ollama_model
        url = "http://localhost:11434/api/generate"

        safe_prompt = "/no_think\n" + prompt

        payload = {
            "model": model,
            "prompt": safe_prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 1024,
                "num_ctx": 4096,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=240)

            if response.status_code != 200:
                print("\n[OLLAMA API ERROR]")
                print("Status code:", response.status_code)
                print("URL:", url)
                print("Model:", model)
                print("Response text:")
                print(response.text)
                print()
                response.raise_for_status()

            data = response.json()

            final_response = data.get("response", "")
            thinking_response = data.get("thinking", "")

            if final_response is None:
                final_response = ""

            if thinking_response is None:
                thinking_response = ""

            final_response = final_response.strip()
            thinking_response = thinking_response.strip()

            if final_response:
                return final_response

            print("\n[OLLAMA EMPTY RESPONSE]")
            print("Model:", model)
            print("Done:", data.get("done"))
            print("Done reason:", data.get("done_reason"))
            print("Prompt eval count:", data.get("prompt_eval_count"))
            print("Eval count:", data.get("eval_count"))
            print("Has thinking:", bool(thinking_response))
            print("Thinking length:", len(thinking_response))
            print("Raw keys:", list(data.keys()))
            print()

            return ""

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Không kết nối được Ollama. Hãy mở CMD khác và chạy: ollama serve"
            )

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama timeout. Prompt có thể quá dài hoặc model chạy quá chậm."
            )

        except Exception as e:
            raise RuntimeError(f"Lỗi khi gọi Ollama: {e}")

    def strip_code_fence(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        return text.strip()

    def parse_candidates_from_llm(self, raw_text: str) -> List[str]:
        """
        Parse output từ LLM thành list candidate.

        Hỗ trợ:
        - JSON list
        - JSON object
        - Dạng đánh số 1. 2. 3.
        """
        raw_text = self.strip_code_fence(raw_text)

        try:
            data = json.loads(raw_text)

            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()][:3]

            if isinstance(data, dict):
                for key in ["translations", "candidates", "results", "translation"]:
                    if key in data:
                        value = data[key]

                        if isinstance(value, list):
                            return [str(x).strip() for x in value if str(x).strip()][:3]

                        if isinstance(value, str):
                            return [value.strip()]
        except Exception:
            pass

        try:
            match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()][:3]
        except Exception:
            pass

        candidates = []

        for line in raw_text.splitlines():
            line = line.strip()

            if not line:
                continue

            line = re.sub(r"^\s*[\-\*\•]\s*", "", line)
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)

            line = re.sub(
                r"^\s*(bản dịch|translation|vietnamese|candidate)\s*\d*\s*[:：]\s*",
                "",
                line,
                flags=re.I,
            )

            line = line.strip().strip('"').strip("'").strip()

            if line:
                candidates.append(line)

        unique = []

        for cand in candidates:
            if cand not in unique:
                unique.append(cand)

        return unique[:3]

    # =====================================================
    # Vietnamese / English filtering
    # =====================================================

    def has_vietnamese_signal(self, text: str) -> bool:
        lower = text.lower()

        has_diacritics = re.search(
            r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
            lower,
        )

        if has_diacritics:
            return True

        common_vi_words = {
            "toi", "tôi", "ban", "bạn", "minh", "mình",
            "la", "là", "cua", "của", "va", "và",
            "khong", "không", "co", "có", "mot", "một",
            "nay", "này", "do", "đó", "vi", "vì",
            "sao", "som", "sớm", "qua", "quá", "luc", "lúc",
            "chuong", "chuông", "bao", "báo",
            "thuc", "thức", "buoi", "buổi", "sang", "sáng",
            "ra", "ngoai", "ngoài", "cua", "cửa",
            "rang", "rằng", "nen", "nên", "can", "cần",
        }

        words = re.findall(r"\w+", lower, flags=re.UNICODE)
        hit = sum(1 for w in words if w in common_vi_words)

        return hit >= 2

    def looks_like_english(self, text: str) -> bool:
        english_words = re.findall(r"[A-Za-z]+", text.lower())

        if len(english_words) == 0:
            return False

        common_english = {
            "the", "you", "your", "and", "or", "but", "because",
            "when", "while", "why", "what", "how", "did", "do",
            "does", "is", "are", "was", "were", "set", "morning",
            "alarm", "yourself", "early", "as", "mutter", "blare",
            "blares", "rushing", "front", "door", "reach", "keys",
            "realize", "they", "not", "there", "think", "need",
            "haircut", "neighbor", "frustrated", "shout",
        }

        count_common = sum(1 for w in english_words if w in common_english)

        if count_common >= 3 and not self.has_vietnamese_signal(text):
            return True

        return False

    def filter_valid_vietnamese_candidates(self, candidates: List[str]) -> List[str]:
        valid = []

        for cand in candidates:
            cand = cand.strip()

            if not cand:
                continue

            lower = cand.lower()

            if lower.startswith("[error]"):
                continue

            if "[cần dịch]" in lower or "[bản ngắn]" in lower or "[bản rút gọn]" in lower:
                continue

            if self.looks_like_english(cand):
                continue

            valid.append(cand)

        unique = []

        for cand in valid:
            if cand not in unique:
                unique.append(cand)

        return unique[:3]

    # =====================================================
    # Context Pack Builder
    # =====================================================

    def build_document_context_pack(
        self,
        full_transcript: str,
        model: Optional[str] = None,
        max_transcript_chars: int = 9000,
    ) -> str:
        """
        Ollama đọc full transcript và tạo Document Context Pack.
        Context Pack được tạo 1 lần cho toàn bài.
        """
        model = model or self.ollama_model

        transcript = full_transcript

        if len(transcript) > max_transcript_chars:
            transcript = transcript[:max_transcript_chars] + "\n...[TRUNCATED]"

        prompt = f"""
Bạn là hệ thống phân tích ngữ cảnh cho bài toán dịch Anh-Việt speech-to-speech.

Hãy đọc transcript tiếng Anh dưới đây và tạo CONTEXT PACK ngắn gọn để hỗ trợ dịch từng segment.

Yêu cầu CONTEXT PACK:
1. Chủ đề chính của bài.
2. Bối cảnh / tình huống.
3. Nhân vật, ngôi xưng hô, quan hệ giữa các đối tượng nếu có.
4. Văn phong nên dùng khi dịch sang tiếng Việt.
5. Các thuật ngữ, cụm từ, tên riêng quan trọng cần dịch nhất quán.
6. Những lưu ý giúp dịch các câu ngắn hoặc câu cụt đúng ngữ cảnh.

Transcript:
{transcript}

Chỉ trả về CONTEXT PACK bằng tiếng Việt. Không giải thích thêm.
""".strip()

        try:
            context_pack = self.call_ollama(prompt, model=model).strip()

            if not context_pack:
                print("[WARN] Context Pack rỗng. Dùng transcript rút gọn làm context thay thế.")
                return transcript[:3000]

            return context_pack

        except Exception as e:
            print("[WARN] Không tạo được Context Pack bằng Ollama:", e)
            print("[WARN] Dùng transcript rút gọn làm context thay thế.")
            return transcript[:3000]

    # =====================================================
    # Feature extraction
    # =====================================================

    def extract_features(self, segment: SourceSegment) -> Dict:
        text = segment.source_text
        lower = text.lower()

        wen = self.count_english_words(text)
        speech_rate = wen / max(segment.duration, 1e-6)

        has_subordinate = any(
            x in lower
            for x in [
                "although", "because", "while", "if", "when",
                "that", "unless", "before", "after", "since"
            ]
        )

        is_question = "?" in text or lower.startswith(
            (
                "what", "why", "how", "when", "where", "who",
                "do", "does", "did", "can", "could", "would", "should"
            )
        )

        has_number = bool(re.search(r"\d+", text))

        proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", text)
        has_proper_noun = len(proper_nouns) > 1

        return {
            "wen": wen,
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

        if features["has_proper_noun"]:
            k -= 0.05

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
    # Translation with Context Pack
    # =====================================================

    def build_context_pack_translation_prompt(
        self,
        segment: SourceSegment,
        max_vi_final: int,
        context_pack: str = "",
    ) -> str:
        """
        Prompt dịch dùng:
        - Document Context Pack
        - Current segment
        - Length constraint
        """

        prompt = f"""
Bạn là hệ thống dịch Anh sang Việt cho speech-to-speech / voice-over.

CONTEXT PACK:
{context_pack if context_pack else "(Không có context pack)"}

CURRENT SEGMENT:
{segment.source_text}

Hãy dịch CURRENT SEGMENT sang tiếng Việt.

Yêu cầu bắt buộc:
- Chỉ dịch CURRENT SEGMENT.
- Không dịch lại CONTEXT PACK.
- Không giữ nguyên câu tiếng Anh.
- Không trả về tiếng Anh.
- Không giải thích.
- Không ghi chú.
- Không thêm thông tin mới.
- Giữ văn phong phù hợp với CONTEXT PACK.
- Mỗi câu dịch không vượt quá khoảng {max_vi_final} âm tiết tiếng Việt.
- Trả về đúng 3 phương án dịch tiếng Việt.

Trả về đúng format sau:
1. câu tiếng Việt thứ nhất
2. câu tiếng Việt thứ hai
3. câu tiếng Việt thứ ba
""".strip()

        return prompt

    def generate_candidates_with_ollama(
        self,
        segment: SourceSegment,
        max_vi_final: int,
        context_pack: str = "",
        model: Optional[str] = None,
    ) -> List[str]:
        """
        Sinh 3 bản dịch ứng viên bằng Ollama.
        Nếu context pack làm model lỗi, thử lại với context ngắn hơn, rồi không context.
        """
        model = model or self.ollama_model

        context_versions = [
            context_pack,
            context_pack[:2500],
            "",
        ]

        last_raw_response = ""

        for context in context_versions:
            prompt = self.build_context_pack_translation_prompt(
                segment=segment,
                max_vi_final=max_vi_final,
                context_pack=context,
            )

            try:
                raw_response = self.call_ollama(prompt, model=model)
                last_raw_response = raw_response

                candidates = self.parse_candidates_from_llm(raw_response)
                candidates = self.filter_valid_vietnamese_candidates(candidates)

                if len(candidates) > 0:
                    return candidates[:3]

                print("[WARN] Ollama trả về nhưng không có candidate tiếng Việt hợp lệ.")
                print("[RAW RESPONSE]")
                print(raw_response)

            except Exception as e:
                print("[WARN] Lỗi khi gọi Ollama để dịch:", e)

        print("[ERROR] Không sinh được bản dịch tiếng Việt cho segment:")
        print(segment.source_text)
        print("[LAST RAW RESPONSE]")
        print(last_raw_response)

        return self.fallback_generate_candidates(segment)

    def fallback_generate_candidates(self, segment: SourceSegment) -> List[str]:
        """
        Fallback khi Ollama lỗi.
        Không trả lại tiếng Anh để tránh nhầm là bản dịch.
        """
        return [
            "[ERROR] Ollama không sinh được bản dịch tiếng Việt.",
            "[ERROR] Kiểm tra ollama serve, tên model, hoặc prompt quá dài.",
            "[ERROR] Không dùng kết quả này cho evaluation.",
        ]

    # =====================================================
    # Scoring
    # =====================================================

    def estimate_duration_error_by_units(self, vi_units: int, max_vi_final: int) -> float:
        return abs(vi_units - max_vi_final) / max(max_vi_final, 1)

    def fluency_penalty_rule_based(self, text_vi: str) -> float:
        penalty = 0.0

        if text_vi.strip().lower().startswith("[error]"):
            return 1.0

        units = self.count_vietnamese_units(text_vi)
        tokens = text_vi.lower().split()

        if units <= 4:
            penalty += 0.20

        if units >= 22:
            penalty += 0.15

        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                penalty += 0.25

        risky_endings = ["vì", "nhưng", "và", "để", "rằng", "nếu", "khi", "mà"]

        if tokens and tokens[-1] in risky_endings:
            penalty += 0.30

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
        candidate_lower = candidate_vi.lower()

        if candidate_lower.startswith("[error]"):
            return 0.0

        if self.looks_like_english(candidate_vi):
            return 0.1

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
        context_pack: str = "",
        model: Optional[str] = None,
    ) -> float:
        model = model or self.ollama_model

        if candidate_vi.strip().lower().startswith("[error]"):
            return 0.0

        context = context_pack[:2500]

        prompt = f"""
Bạn là bộ đánh giá chất lượng dịch Anh-Việt trong hệ thống speech-to-speech.

Hãy chấm điểm mức độ giữ nghĩa của bản dịch tiếng Việt so với câu tiếng Anh.
Có thể dùng CONTEXT PACK để hiểu ngữ cảnh toàn bài.

Chỉ trả về một số từ 0 đến 1. Không giải thích.

Quy ước:
1.0 = giữ nghĩa rất tốt
0.8 = giữ ý chính, mất rất ít chi tiết
0.5 = chỉ đúng một phần
0.0 = sai nghĩa hoặc không phải tiếng Việt

CONTEXT PACK:
{context if context else "(Không có)"}

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
        context_pack: str = "",
    ) -> TranslationCandidate:
        vi_units = self.count_vietnamese_units(candidate_text)

        if self.use_llm_judge:
            sem_score = self.semantic_score_with_ollama(
                source_text=source_text,
                candidate_vi=candidate_text,
                context_pack=context_pack,
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
        context_pack: str = "",
    ) -> Dict:
        pred_len_vi = self.predict_len_vi(segment)
        max_vi_final = self.calculate_max_vi_final(segment, pred_len_vi)

        candidates = self.generate_candidates_with_ollama(
            segment=segment,
            max_vi_final=max_vi_final,
            context_pack=context_pack,
            model=self.ollama_model,
        )

        scored_candidates = [
            self.score_candidate(
                source_text=segment.source_text,
                candidate_text=cand,
                max_vi_final=max_vi_final,
                context_pack=context_pack,
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
                "type": "document_context_pack",
                "context_pack_preview": context_pack[:800],
            },
        }