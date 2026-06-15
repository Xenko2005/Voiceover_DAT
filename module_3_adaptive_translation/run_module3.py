import os
import json
import csv
import argparse
from adaptive_length_translation import SourceSegment, AdaptiveLengthTranslator


# =====================================================
# Safe extract helpers
# =====================================================

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def extract_text(item):
    """
    Hỗ trợ nhiều tên field khác nhau.

    File module 1 hiện dùng:
        transcribed_text
    """
    return (
        item.get("transcribed_text")
        or item.get("text")
        or item.get("source_text")
        or item.get("transcript")
        or item.get("sentence")
        or ""
    )


def extract_start_time(item):
    return safe_float(
        item.get("start_time", item.get("start", 0.0)),
        default=0.0
    )


def extract_end_time(item, start_time):
    """
    File module 1 hiện dùng:
        stop_time
    """
    return safe_float(
        item.get("end_time", item.get("stop_time", item.get("end", start_time))),
        default=start_time
    )


def extract_duration(item, start_time, end_time):
    duration = item.get("duration", None)

    if duration is None or duration == "":
        return max(0.0, end_time - start_time)

    return safe_float(duration, default=max(0.0, end_time - start_time))


# =====================================================
# Load data
# =====================================================

def load_segments_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data == []:
        raise ValueError(
            f"File {path} đang là JSON rỗng []. Không có segment nào để xử lý."
        )

    source_file = None

    # Format:
    # {
    #   "source_file": "ted_talk.mp3",
    #   "chunks": [...]
    # }
    if isinstance(data, dict):
        source_file = data.get("source_file")

        if "chunks" in data:
            data = data["chunks"]
        elif "segments" in data:
            data = data["segments"]
        elif "results" in data:
            data = data["results"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("JSON không đúng format. Cần list hoặc dict chứa chunks/segments.")

    segments = []

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        text = extract_text(item).strip()

        start_time = extract_start_time(item)
        end_time = extract_end_time(item, start_time)
        duration = extract_duration(item, start_time, end_time)

        if not text:
            continue

        segments.append(
            SourceSegment(
                segment_id=i,
                source_text=text,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
            )
        )

    print(f"Source file: {source_file}")
    return segments


def load_segments_from_csv(path):
    segments = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            text = extract_text(row).strip()

            start_time = extract_start_time(row)
            end_time = extract_end_time(row, start_time)
            duration = extract_duration(row, start_time, end_time)

            if not text:
                continue

            segments.append(
                SourceSegment(
                    segment_id=i,
                    source_text=text,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                )
            )

    return segments


def load_segments_from_txt(path):
    """
    Fallback nếu chỉ có transcript txt.
    Vì không có timestamp thật, duration tạm = 3.0s.
    """
    segments = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for i, line in enumerate(lines, start=1):
        segments.append(
            SourceSegment(
                segment_id=i,
                source_text=line,
                start_time=(i - 1) * 3.0,
                end_time=i * 3.0,
                duration=3.0,
            )
        )

    return segments


def load_segments(path):
    lower = path.lower()

    if lower.endswith(".json"):
        return load_segments_from_json(path)

    if lower.endswith(".csv"):
        return load_segments_from_csv(path)

    if lower.endswith(".txt"):
        return load_segments_from_txt(path)

    raise ValueError("Chỉ hỗ trợ .json, .csv hoặc .txt")


def find_first_non_empty_data_file():
    candidate_dirs = [
        "output_files",
        "output_stream",
        "speech_to_text",
    ]

    supported_exts = [".json", ".csv", ".txt"]

    for folder in candidate_dirs:
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            lower = filename.lower()

            if not any(lower.endswith(ext) for ext in supported_exts):
                continue

            if os.path.getsize(path) <= 2:
                print(f"Bỏ qua file rỗng: {path}")
                continue

            return path

    return None


# =====================================================
# Context helpers
# =====================================================

def build_full_transcript(segments):
    """
    Ghép toàn bộ transcript từ ASR output.
    Dùng để tạo Document Context Pack.
    """
    lines = []

    for seg in segments:
        lines.append(
            f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s] {seg.source_text}"
        )

    return "\n".join(lines)


def build_local_context(segments, current_index, window_size=2):
    """
    Lấy vài segment trước và sau current segment.
    current_index bắt đầu từ 0.
    """
    start = max(0, current_index - window_size)
    end = min(len(segments), current_index + window_size + 1)

    previous_segments = segments[start:current_index]
    next_segments = segments[current_index + 1:end]

    prev_text = "\n".join(
        [
            f"- [{seg.start_time:.1f}s - {seg.end_time:.1f}s] {seg.source_text}"
            for seg in previous_segments
        ]
    )

    next_text = "\n".join(
        [
            f"- [{seg.start_time:.1f}s - {seg.end_time:.1f}s] {seg.source_text}"
            for seg in next_segments
        ]
    )

    return {
        "previous": prev_text,
        "next": next_text
    }


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Đường dẫn file input từ module 1 hoặc module 2."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:3b",
        help="Tên model Ollama, ví dụ qwen2.5:3b, llama3.2:3b."
    )

    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Bật Ollama judge để chấm SemScore. Chạy chậm hơn."
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Giới hạn số segment để test nhanh. Ví dụ --max-segments 5."
    )

    parser.add_argument(
        "--no-context-builder",
        action="store_true",
        help="Không gọi Ollama tạo Document Context Pack, dùng full transcript rút gọn."
    )

    args = parser.parse_args()

    if args.input:
        data_path = args.input
    else:
        data_path = find_first_non_empty_data_file()

    if data_path is None:
        print("Không tìm thấy file data hợp lệ.")
        print("Ví dụ chạy:")
        print(r"python module_3_adaptive_translation\run_module3.py --input output_files\ted_talk.json")
        return

    if not os.path.exists(data_path):
        print(f"Không tìm thấy file: {data_path}")
        return

    print(f"Đang đọc data từ: {data_path}")

    try:
        segments = load_segments(data_path)
    except Exception as e:
        print("Lỗi khi đọc data:", e)
        return

    if args.max_segments is not None:
        segments = segments[:args.max_segments]

    print(f"Số segment đọc được: {len(segments)}")

    if len(segments) == 0:
        print("Không có segment nào để xử lý.")
        return

    translator = AdaptiveLengthTranslator(
        tts_ceiling=5.0,
        margin=1,
        lambda_dur=0.6,
        mu_flu=0.3,
        ollama_model=args.model,
        use_llm_judge=args.use_llm_judge,
    )

    # Build full transcript
    full_transcript = build_full_transcript(segments)

    # Build document context pack
    if args.no_context_builder:
        print("Bỏ qua Document Context Builder. Dùng full transcript rút gọn làm context.")
        document_context = full_transcript[:3000]
    else:
        print("\nĐang tạo Document Context Pack bằng Ollama...")
        document_context = translator.build_document_context_with_ollama(
            full_transcript=full_transcript,
            model=args.model,
        )

    print("\n========== DOCUMENT CONTEXT PACK ==========")
    print(document_context[:2000])
    print("===========================================\n")

    results = []

    for idx, segment in enumerate(segments):
        local_context = build_local_context(segments, idx, window_size=2)

        result = translator.translate_and_select(
            segment=segment,
            global_context=document_context,
            previous_context=local_context["previous"],
            next_context=local_context["next"],
        )

        results.append(result)

        print("\n" + "=" * 80)
        print("SEGMENT:", result["segment_id"])
        print("SOURCE:", result["source_text"])
        print("START:", result["start_time"])
        print("END:", result["end_time"])
        print("DURATION:", result["duration"])
        print("PredLenVI:", result["pred_len_vi"])
        print("MaxVI_final:", result["max_vi_final"])
        print("BEST:", result["best_translation"])
        print("SCORE:", round(result["best_score"], 3))

        print("\nCandidates:")
        for c in result["candidates"]:
            print(
                f"- {c['text_vi']} | "
                f"units={c['vi_units']} | "
                f"Sem={round(c['sem_score'], 3)} | "
                f"DurErr={round(c['dur_error'], 3)} | "
                f"Flu={round(c['flu_penalty'], 3)} | "
                f"Score={round(c['final_score'], 3)}"
            )

    output_dir = os.path.join("module_3_adaptive_translation", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "module3_translation_results.json")

    final_output = {
        "input_file": data_path,
        "ollama_model": args.model,
        "use_llm_judge": args.use_llm_judge,
        "document_context": document_context,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\nĐã lưu kết quả tại:", output_path)


if __name__ == "__main__":
    main()