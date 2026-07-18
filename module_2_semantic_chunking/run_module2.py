import os
import json
import argparse
import sys
from typing import List, Dict, Any

from semantic_chunk_buffer import ASRChunk, SemanticChunkBuffer


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


def extract_text(item: Dict[str, Any]) -> str:
    """
    Hỗ trợ format từ module 1:
        transcribed_text

    Và các format khác:
        text, source_text, transcript, sentence
    """
    return (
        item.get("transcribed_text")
        or item.get("text")
        or item.get("source_text")
        or item.get("transcript")
        or item.get("sentence")
        or ""
    )


def extract_start_time(item: Dict[str, Any]) -> float:
    return safe_float(
        item.get("start_time", item.get("start", 0.0)),
        default=0.0
    )


def extract_end_time(item: Dict[str, Any], start_time: float) -> float:
    """
    Module 1 hiện dùng:
        stop_time
    """
    return safe_float(
        item.get("end_time", item.get("stop_time", item.get("end", start_time))),
        default=start_time
    )


def extract_duration(item: Dict[str, Any], start_time: float, end_time: float) -> float:
    duration = item.get("duration", None)

    if duration is None or duration == "":
        return max(0.0, end_time - start_time)

    return safe_float(duration, default=max(0.0, end_time - start_time))


# =====================================================
# Load Module 1 output
# =====================================================

def load_asr_chunks_from_json(path: str) -> Dict:
    """
    Đọc file JSON từ Module 1.

    Hỗ trợ format:
    {
      "source_file": "ted_talk.mp3",
      "chunks": [
        {
          "transcribed_text": "...",
          "start_time": 7.2,
          "stop_time": 13.1,
          "duration": 5.9
        }
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data == []:
        raise ValueError(f"File {path} đang rỗng [].")

    source_file = None

    if isinstance(data, dict):
        source_file = data.get("source_file")

        if "chunks" in data:
            raw_chunks = data["chunks"]
        elif "segments" in data:
            raw_chunks = data["segments"]
        elif "results" in data:
            raw_chunks = data["results"]
        else:
            raw_chunks = [data]

    elif isinstance(data, list):
        raw_chunks = data

    else:
        raise ValueError("JSON không hợp lệ. Cần dict hoặc list.")

    chunks: List[ASRChunk] = []

    for i, item in enumerate(raw_chunks, start=1):
        if not isinstance(item, dict):
            continue

        text = extract_text(item).strip()

        if not text:
            continue

        start_time = extract_start_time(item)
        end_time = extract_end_time(item, start_time)
        duration = extract_duration(item, start_time, end_time)

        chunks.append(
            ASRChunk(
                chunk_id=i,
                text=text,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
            )
        )

    return {
        "source_file": source_file,
        "chunks": chunks,
        "raw_num_items": len(raw_chunks),
    }


# =====================================================
# Save output
# =====================================================

def save_module2_output(
    output_path: str,
    input_file: str,
    source_file: str,
    chunks: List[ASRChunk],
    segments,
    buffer_config: Dict,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "module": "module_2_semantic_chunk_buffering",
        "input_file": input_file,
        "source_file": source_file,
        "num_input_chunks": len(chunks),
        "num_output_segments": len(segments),
        "buffer_config": buffer_config,
        "segments": [
            {
                "segment_id": seg.segment_id,
                "source_text": seg.source_text,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "source_chunk_ids": seg.source_chunk_ids,
                "num_chunks": seg.num_chunks,
                "release_mode": seg.release_mode,
                "release_reason": seg.release_reason,
            }
            for seg in segments
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn file JSON output từ Module 1."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file output cho Module 2."
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Số từ tối thiểu để xét release."
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=45,
        help="Nếu buffer vượt số từ này thì force release."
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=12.0,
        help="Nếu buffer vượt duration này thì force release."
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=6,
        help="Nếu buffer gom quá nhiều chunks thì force release."
    )

    parser.add_argument(
        "--use-ollama-judge",
        action="store_true",
        help="Dùng Ollama kiểm tra semantic completeness. Chạy chậm hơn."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:3b",
        help="Tên model Ollama nếu dùng --use-ollama-judge."
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Không tìm thấy file input: {args.input}")
        return

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(
            "module_2_semantic_chunking",
            "outputs",
            f"{base_name}_semantic_segments.json"
        )

    print(f"Đang đọc Module 1 output từ: {args.input}")

    try:
        loaded = load_asr_chunks_from_json(args.input)
    except Exception as e:
        print("Lỗi đọc input:", e)
        return

    chunks = loaded["chunks"]
    source_file = loaded["source_file"]

    print(f"Source file: {source_file}")
    print(f"Số raw items: {loaded['raw_num_items']}")
    print(f"Số ASR chunks hợp lệ: {len(chunks)}")

    if len(chunks) == 0:
        print("Không có ASR chunk nào để xử lý.")
        return

    buffer_config = {
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_duration": args.max_duration,
        "max_chunks": args.max_chunks,
        "use_ollama_judge": args.use_ollama_judge,
        "ollama_model": args.model,
    }

    buffer = SemanticChunkBuffer(
        min_words=args.min_words,
        max_words=args.max_words,
        max_duration=args.max_duration,
        max_chunks=args.max_chunks,
        use_ollama_judge=args.use_ollama_judge,
        ollama_model=args.model,
    )

    print("\nĐang chạy Semantic Chunk Buffering...")
    segments = buffer.process_chunks(chunks)

    print(f"Số semantic segments tạo ra: {len(segments)}")

    print("\n========== PREVIEW OUTPUT ==========")

    for seg in segments[:10]:
        print("\n" + "-" * 80)
        print(f"SEGMENT {seg.segment_id}")
        print(f"Time: {seg.start_time:.2f}s → {seg.end_time:.2f}s")
        print(f"Duration: {seg.duration:.2f}s")
        print(f"Chunks: {seg.source_chunk_ids}")
        print(f"Mode: {seg.release_mode}")
        print(f"Reason: {seg.release_reason}")
        print(f"Text: {seg.source_text}")

    if len(segments) > 10:
        print(f"\n... còn {len(segments) - 10} segments nữa")

    output = save_module2_output(
        output_path=args.output,
        input_file=args.input,
        source_file=source_file,
        chunks=chunks,
        segments=segments,
        buffer_config=buffer_config,
    )

    print("\nĐã lưu output Module 2 tại:")
    print(args.output)

    print("\nOutput này có thể đưa vào Module 3 bằng lệnh:")
    print(
        f'python module_3_adaptive_translation\\run_module3.py '
        f'--input "{args.output}" --model qwen2.5:3b --max-segments 3'
    )


if __name__ == "__main__":
    main()
