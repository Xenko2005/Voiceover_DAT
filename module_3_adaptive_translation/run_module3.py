import os
import json
import csv
from adaptive_length_translation import SourceSegment, AdaptiveLengthTranslator


def load_segments_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []

    if isinstance(data, dict):
        if "segments" in data:
            data = data["segments"]
        elif "chunks" in data:
            data = data["chunks"]
        else:
            data = [data]

    for i, item in enumerate(data, start=1):
        text = (
            item.get("text")
            or item.get("source_text")
            or item.get("transcript")
            or item.get("sentence")
            or ""
        )

        start = float(item.get("start_time", item.get("start", 0.0)))
        end = float(item.get("end_time", item.get("end", start)))
        duration = float(item.get("duration", max(0.0, end - start)))

        if text.strip():
            segments.append(
                SourceSegment(
                    segment_id=i,
                    source_text=text.strip(),
                    start_time=start,
                    end_time=end,
                    duration=duration,
                )
            )

    return segments


def load_segments_from_csv(path):
    segments = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            text = (
                row.get("text")
                or row.get("source_text")
                or row.get("transcript")
                or row.get("sentence")
                or ""
            )

            start = float(row.get("start_time", row.get("start", 0.0)) or 0.0)
            end = float(row.get("end_time", row.get("end", start)) or start)
            duration = float(row.get("duration", max(0.0, end - start)) or max(0.0, end - start))

            if text.strip():
                segments.append(
                    SourceSegment(
                        segment_id=i,
                        source_text=text.strip(),
                        start_time=start,
                        end_time=end,
                        duration=duration,
                    )
                )

    return segments


def find_first_data_file():
    candidate_dirs = [
        "output_files",
        "output_stream",
        "speech_to_text",
    ]

    for folder in candidate_dirs:
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            lower = filename.lower()

            if lower.endswith(".json") or lower.endswith(".csv"):
                return os.path.join(folder, filename)

    return None


def main():
    data_path = find_first_data_file()

    if data_path is None:
        print("Không tìm thấy file .json hoặc .csv trong output_files/output_stream/speech_to_text.")
        print("Bạn hãy kiểm tra tên file output của module 1 rồi truyền path thủ công.")
        return

    print(f"Đang đọc data từ: {data_path}")

    if data_path.lower().endswith(".json"):
        segments = load_segments_from_json(data_path)
    elif data_path.lower().endswith(".csv"):
        segments = load_segments_from_csv(data_path)
    else:
        raise ValueError("Chỉ hỗ trợ JSON hoặc CSV ở bản này.")

    print(f"Số segment đọc được: {len(segments)}")

    translator = AdaptiveLengthTranslator(
        tts_ceiling=5.0,
        margin=1,
        lambda_dur=0.6,
        mu_flu=0.3,
    )

    results = []

    for segment in segments:
        result = translator.translate_and_select(segment)
        results.append(result)

        print("\n" + "=" * 80)
        print("SEGMENT:", result["segment_id"])
        print("SOURCE:", result["source_text"])
        print("DURATION:", result["duration"])
        print("PredLenVI:", result["pred_len_vi"])
        print("MaxVI_final:", result["max_vi_final"])
        print("BEST:", result["best_translation"])
        print("SCORE:", round(result["best_score"], 3))

    output_dir = os.path.join("module_3_adaptive_translation", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "module3_translation_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nĐã lưu kết quả tại:", output_path)


if __name__ == "__main__":
    main()