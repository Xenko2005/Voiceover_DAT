import os
import json
import csv
import argparse
import sys
from adaptive_length_translation import SourceSegment, AdaptiveLengthTranslator


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
    return (
        item.get("transcribed_text")
        or item.get("source_text")
        or item.get("text")
        or item.get("transcript")
        or item.get("sentence")
        or ""
    )


def extract_start_time(item):
    return safe_float(
        item.get("start_time", item.get("start", 0.0)),
        default=0.0,
    )


def extract_end_time(item, start_time):
    return safe_float(
        item.get("end_time", item.get("stop_time", item.get("end", start_time))),
        default=start_time,
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
        raise ValueError(f"File {path} đang là JSON rỗng [].")

    source_file = None

    if isinstance(data, dict):
        source_file = data.get("source_file")

        if "segments" in data:
            data = data["segments"]
        elif "chunks" in data:
            data = data["chunks"]
        elif "results" in data:
            data = data["results"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError(
            "JSON không đúng format. Cần list hoặc dict chứa chunks/segments/results."
        )

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

        segment_id = item.get("segment_id", i)

        segments.append(
            SourceSegment(
                segment_id=int(segment_id),
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

            segment_id = row.get("segment_id", i)

            segments.append(
                SourceSegment(
                    segment_id=int(segment_id),
                    source_text=text,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                )
            )

    return segments


def load_segments_from_txt(path):
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


# =====================================================
# Context helpers
# =====================================================

def build_full_transcript(segments):
    lines = []

    for seg in segments:
        lines.append(
            f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s] {seg.source_text}"
        )

    return "\n".join(lines)


# =====================================================
# Batch helpers
# =====================================================

def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def save_intermediate_output(
    output_path,
    data_path,
    model,
    use_llm_judge,
    context_pack,
    results,
    num_total_segments,
    num_processed_segments,
    current_batch,
    total_batches,
):
    """
    Lưu tạm sau mỗi batch để nếu chạy bị lỗi giữa chừng
    thì vẫn còn kết quả đã xử lý.
    """
    final_output = {
        "input_file": data_path,
        "ollama_model": model,
        "use_llm_judge": use_llm_judge,
        "context_mode": "document_context_pack",
        "current_batch": current_batch,
        "total_batches": total_batches,
        "num_total_segments": num_total_segments,
        "num_processed_segments": num_processed_segments,
        "document_context_pack": context_pack,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
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
        "--batch-size",
        type=int,
        default=5,
        help="Số segment xử lý trong mỗi batch. Mặc định 5."
    )

    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=9000,
        help="Giới hạn độ dài full transcript đưa vào Context Pack."
    )

    parser.add_argument(
        "--optimizer-config",
        type=str,
        default=None,
        help="Optional JSON produced by train_phost_module_optimizers.py.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Không tìm thấy file input: {args.input}")
        return

    print(f"Đang đọc data từ: {args.input}")

    try:
        all_segments = load_segments(args.input)
    except Exception as e:
        print("Lỗi khi đọc data:", e)
        return

    print(f"Tổng số segment đọc được: {len(all_segments)}")

    if len(all_segments) == 0:
        print("Không có segment nào để xử lý.")
        return

    if args.max_segments is not None:
        segments_to_process = all_segments[:args.max_segments]
    else:
        segments_to_process = all_segments

    print(f"Số segment sẽ xử lý: {len(segments_to_process)}")

    optimizer_config = None
    if args.optimizer_config:
        with open(args.optimizer_config, "r", encoding="utf-8") as f:
            optimizer_config = json.load(f)
        print("Đã nạp optimizer config:", args.optimizer_config)

    translator = AdaptiveLengthTranslator(
        tts_ceiling=5.0,
        margin=1,
        lambda_dur=0.6,
        mu_flu=0.3,
        ollama_model=args.model,
        use_llm_judge=args.use_llm_judge,
        optimizer_config=optimizer_config,
    )

    # Build full transcript từ toàn bộ bài
    full_transcript = build_full_transcript(all_segments)

    print("\nĐang tạo Document Context Pack bằng Ollama...")
    context_pack = translator.build_document_context_pack(
        full_transcript=full_transcript,
        model=args.model,
        max_transcript_chars=args.max_transcript_chars,
    )

    print("\n========== DOCUMENT CONTEXT PACK ==========")
    print(context_pack[:2000])
    print("===========================================\n")

    output_dir = os.path.join("module_3_adaptive_translation", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "module3_translation_results.json")

    results = []

    batches = list(make_batches(segments_to_process, args.batch_size))
    total_batches = len(batches)

    for batch_idx, batch in enumerate(batches, start=1):
        print("\n" + "=" * 80)
        print(f"Đang xử lý batch {batch_idx}/{total_batches}")
        print("Segments trong batch:", [seg.segment_id for seg in batch])
        print("=" * 80)

        for segment in batch:
            result = translator.translate_and_select(
                segment=segment,
                context_pack=context_pack,
            )

            results.append(result)

            print("\n" + "-" * 80)
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

        # Lưu tạm sau mỗi batch
        save_intermediate_output(
            output_path=output_path,
            data_path=args.input,
            model=args.model,
            use_llm_judge=args.use_llm_judge,
            context_pack=context_pack,
            results=results,
            num_total_segments=len(all_segments),
            num_processed_segments=len(results),
            current_batch=batch_idx,
            total_batches=total_batches,
        )

        print(f"\nĐã lưu tạm sau batch {batch_idx}/{total_batches}: {output_path}")

    print("\nHoàn tất Module 3.")
    print("Đã lưu kết quả tại:", output_path)


if __name__ == "__main__":
    main()
