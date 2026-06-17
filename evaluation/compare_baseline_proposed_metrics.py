import os
import re
import csv
import json
import argparse
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional


# =====================================================
# Basic metric helpers
# =====================================================

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_mean(values: List[float]) -> float:
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return mean(values)


def safe_median(values: List[float]) -> float:
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return median(values)


def count_vietnamese_units(text: str) -> int:
    text = str(text or "")
    clean = re.sub(r"[^\w\sÀ-ỹ]", "", text, flags=re.UNICODE)
    units = [u for u in clean.split() if u.strip()]
    return len(units)


def duration_error_ratio(tts_duration: float, source_duration: float) -> float:
    if source_duration <= 0:
        return 0.0
    return abs(tts_duration - source_duration) / source_duration


def tts_source_ratio(tts_duration: float, source_duration: float) -> float:
    if source_duration <= 0:
        return 0.0
    return tts_duration / source_duration


def fluency_penalty_rule_based(text_vi: str) -> float:
    """
    Rule-based fluency penalty đơn giản:
    0 = ổn
    1 = tệ

    Không cần LLM judge để evaluation chạy nhanh.
    """
    text_vi = str(text_vi or "").strip()
    lower = text_vi.lower()

    if not text_vi:
        return 1.0

    if lower.startswith("[error]"):
        return 1.0

    penalty = 0.0
    units = count_vietnamese_units(text_vi)
    tokens = lower.split()

    if units <= 4:
        penalty += 0.20

    if units >= 25:
        penalty += 0.15

    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1]:
            penalty += 0.25

    risky_endings = {"vì", "nhưng", "và", "để", "rằng", "nếu", "khi", "mà"}
    if tokens and tokens[-1] in risky_endings:
        penalty += 0.30

    unnatural_patterns = [
        "[cần dịch]",
        "[bản ngắn]",
        "[bản rút gọn]",
        "translation failed",
        "empty translation",
    ]

    if any(p in lower for p in unnatural_patterns):
        penalty += 0.50

    return min(penalty, 1.0)


# =====================================================
# Load JSON
# =====================================================

def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_module3_map(path: Optional[str]) -> Dict[int, Dict]:
    if not path or not os.path.exists(path):
        return {}

    data = load_json(path)
    results = data.get("results", [])

    result_map = {}

    for item in results:
        try:
            segment_id = int(item.get("segment_id"))
            result_map[segment_id] = item
        except Exception:
            continue

    return result_map


# =====================================================
# Normalize baseline / proposed items
# =====================================================

def normalize_baseline_items(baseline_json: Dict) -> List[Dict]:
    """
    Input:
        baseline_pipeline/outputs/baseline_results.json

    Mỗi item baseline có:
        chunk_id
        source_duration
        tts_duration
        baseline_translation
        vi_units
        duration_error_ratio
    """
    rows = []

    for item in baseline_json.get("results", []):
        source_duration = safe_float(item.get("source_duration"))
        tts_duration = safe_float(item.get("tts_duration"))
        translation = item.get("baseline_translation", "")

        vi_units = item.get("vi_units")
        if vi_units is None:
            vi_units = count_vietnamese_units(translation)

        dur_err = item.get("duration_error_ratio")
        if dur_err is None:
            dur_err = duration_error_ratio(tts_duration, source_duration)

        rows.append(
            {
                "system": "baseline",
                "id": item.get("chunk_id"),
                "source_text": item.get("source_text", ""),
                "translation": translation,
                "source_duration": source_duration,

                # Baseline không có duration alignment,
                # nên raw_duration = final_duration = tts_duration.
                "raw_tts_duration": tts_duration,
                "final_tts_duration": tts_duration,

                "duration_error_ratio": safe_float(dur_err),
                "tts_source_ratio": tts_source_ratio(tts_duration, source_duration),
                "vi_units": int(vi_units),
                "fluency_penalty": fluency_penalty_rule_based(translation),

                # Baseline không chỉnh speed.
                "speed_factor": 1.0,
                "abs_speed_change": 0.0,

                "overrun": 1 if tts_duration > source_duration else 0,
                "underrun": 1 if tts_duration < source_duration else 0,
            }
        )

    return rows


def normalize_proposed_items(module4_json: Dict, module3_map: Dict[int, Dict]) -> List[Dict]:
    """
    Input:
        module_4_tts_alignment/outputs/module4_tts_results.json
        module_3_adaptive_translation/outputs/module3_translation_results.json

    Mỗi item proposed có:
        segment_id
        raw_tts_duration
        final_tts_duration
        speed_factor
        duration_error_ratio
        best_translation
    """
    rows = []

    for item in module4_json.get("results", []):
        segment_id = item.get("segment_id")
        try:
            segment_id_int = int(segment_id)
        except Exception:
            segment_id_int = segment_id

        module3_item = module3_map.get(segment_id_int, {})

        source_duration = safe_float(item.get("source_duration"))
        raw_tts_duration = safe_float(item.get("raw_tts_duration"))

        final_tts_duration = safe_float(
            item.get("final_tts_duration"),
            default=raw_tts_duration,
        )

        translation = (
            item.get("best_translation")
            or module3_item.get("best_translation")
            or ""
        )

        dur_err = item.get("duration_error_ratio")
        if dur_err is None:
            dur_err = duration_error_ratio(final_tts_duration, source_duration)

        speed_factor = safe_float(item.get("speed_factor"), default=1.0)

        rows.append(
            {
                "system": "proposed",
                "id": segment_id,
                "source_text": item.get("source_text", ""),
                "translation": translation,
                "source_duration": source_duration,

                # raw_tts_duration = TTS thật trước khi align
                # final_tts_duration = sau khi speed alignment
                "raw_tts_duration": raw_tts_duration,
                "final_tts_duration": final_tts_duration,

                "duration_error_ratio": safe_float(dur_err),
                "tts_source_ratio": tts_source_ratio(final_tts_duration, source_duration),
                "vi_units": count_vietnamese_units(translation),
                "fluency_penalty": fluency_penalty_rule_based(translation),

                "speed_factor": speed_factor,
                "abs_speed_change": abs(speed_factor - 1.0),

                "overrun": 1 if final_tts_duration > source_duration else 0,
                "underrun": 1 if final_tts_duration < source_duration else 0,

                # Metric riêng từ Module 3 nếu có
                "module3_pred_len_vi": module3_item.get("pred_len_vi"),
                "module3_max_vi_final": module3_item.get("max_vi_final"),
                "module3_best_score": module3_item.get("best_score"),
            }
        )

    return rows


# =====================================================
# Summary
# =====================================================

def summarize_rows(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "num_items": 0,
            "avg_duration_error_ratio": 0.0,
            "median_duration_error_ratio": 0.0,
            "max_duration_error_ratio": 0.0,
            "avg_vi_units": 0.0,
            "avg_raw_tts_duration": 0.0,
            "avg_final_tts_duration": 0.0,
            "avg_tts_source_ratio": 0.0,
            "avg_fluency_penalty": 0.0,
            "overrun_rate": 0.0,
            "underrun_rate": 0.0,
            "avg_speed_factor": 0.0,
            "avg_abs_speed_change": 0.0,
        }

    dur_errors = [safe_float(r.get("duration_error_ratio")) for r in rows]
    vi_units = [safe_float(r.get("vi_units")) for r in rows]
    raw_durs = [safe_float(r.get("raw_tts_duration")) for r in rows]
    final_durs = [safe_float(r.get("final_tts_duration")) for r in rows]
    ratios = [safe_float(r.get("tts_source_ratio")) for r in rows]
    flu = [safe_float(r.get("fluency_penalty")) for r in rows]
    overrun = [safe_float(r.get("overrun")) for r in rows]
    underrun = [safe_float(r.get("underrun")) for r in rows]
    speed = [safe_float(r.get("speed_factor"), 1.0) for r in rows]
    abs_speed = [safe_float(r.get("abs_speed_change")) for r in rows]

    return {
        "num_items": len(rows),

        "avg_duration_error_ratio": round(safe_mean(dur_errors), 4),
        "median_duration_error_ratio": round(safe_median(dur_errors), 4),
        "max_duration_error_ratio": round(max(dur_errors) if dur_errors else 0.0, 4),

        "avg_vi_units": round(safe_mean(vi_units), 4),

        "avg_raw_tts_duration": round(safe_mean(raw_durs), 4),
        "avg_final_tts_duration": round(safe_mean(final_durs), 4),
        "avg_tts_source_ratio": round(safe_mean(ratios), 4),

        "avg_fluency_penalty": round(safe_mean(flu), 4),

        "overrun_rate": round(safe_mean(overrun), 4),
        "underrun_rate": round(safe_mean(underrun), 4),

        "avg_speed_factor": round(safe_mean(speed), 4),
        "avg_abs_speed_change": round(safe_mean(abs_speed), 4),
    }


def build_comparison_summary(baseline_summary: Dict, proposed_summary: Dict) -> Dict:
    baseline_der = baseline_summary.get("avg_duration_error_ratio", 0.0)
    proposed_der = proposed_summary.get("avg_duration_error_ratio", 0.0)

    duration_error_reduction = baseline_der - proposed_der

    if baseline_der > 0:
        duration_error_reduction_percent = duration_error_reduction / baseline_der * 100
    else:
        duration_error_reduction_percent = 0.0

    baseline_flu = baseline_summary.get("avg_fluency_penalty", 0.0)
    proposed_flu = proposed_summary.get("avg_fluency_penalty", 0.0)

    return {
        "duration_error_reduction": round(duration_error_reduction, 4),
        "duration_error_reduction_percent": round(duration_error_reduction_percent, 2),

        "fluency_penalty_delta": round(baseline_flu - proposed_flu, 4),

        "baseline_avg_duration_error_ratio": baseline_der,
        "proposed_avg_duration_error_ratio": proposed_der,

        "baseline_overrun_rate": baseline_summary.get("overrun_rate", 0.0),
        "proposed_overrun_rate": proposed_summary.get("overrun_rate", 0.0),

        "baseline_avg_vi_units": baseline_summary.get("avg_vi_units", 0.0),
        "proposed_avg_vi_units": proposed_summary.get("avg_vi_units", 0.0),

        "interpretation": {
            "duration_error_reduction": "Dương nghĩa là proposed giảm lỗi duration so với baseline.",
            "duration_error_reduction_percent": "Phần trăm giảm lỗi duration trung bình.",
            "fluency_penalty_delta": "Dương nghĩa là proposed có fluency penalty thấp hơn baseline.",
            "overrun_rate": "Tỷ lệ đoạn TTS dài hơn source duration.",
            "avg_abs_speed_change": "Chỉ có ý nghĩa với proposed vì Module 4 có duration alignment.",
        },
    }


# =====================================================
# Save helpers
# =====================================================

def write_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return

    fieldnames = sorted(set().union(*(row.keys() for row in rows)))

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline_pipeline/outputs/baseline_results.json",
        help="Output JSON của baseline pipeline.",
    )

    parser.add_argument(
        "--proposed-m4",
        type=str,
        default="module_4_tts_alignment/outputs/module4_tts_results.json",
        help="Output JSON của Module 4 proposed.",
    )

    parser.add_argument(
        "--proposed-m3",
        type=str,
        default="module_3_adaptive_translation/outputs/module3_translation_results.json",
        help="Output JSON của Module 3 proposed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/outputs",
        help="Thư mục lưu evaluation output.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        print("Không tìm thấy baseline file:", args.baseline)
        return

    if not os.path.exists(args.proposed_m4):
        print("Không tìm thấy proposed Module 4 file:", args.proposed_m4)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_json = load_json(args.baseline)
    module4_json = load_json(args.proposed_m4)
    module3_map = load_module3_map(args.proposed_m3)

    baseline_rows = normalize_baseline_items(baseline_json)
    proposed_rows = normalize_proposed_items(module4_json, module3_map)

    baseline_summary = summarize_rows(baseline_rows)
    proposed_summary = summarize_rows(proposed_rows)
    comparison_summary = build_comparison_summary(
        baseline_summary=baseline_summary,
        proposed_summary=proposed_summary,
    )

    combined_rows = baseline_rows + proposed_rows

    final_output = {
        "baseline_file": args.baseline,
        "proposed_module4_file": args.proposed_m4,
        "proposed_module3_file": args.proposed_m3,
        "metrics": {
            "duration_error_ratio": "|TTS_duration - Source_duration| / Source_duration",
            "tts_source_ratio": "TTS_duration / Source_duration",
            "overrun_rate": "Tỷ lệ đoạn có TTS_duration > Source_duration",
            "underrun_rate": "Tỷ lệ đoạn có TTS_duration < Source_duration",
            "fluency_penalty": "Rule-based penalty, càng thấp càng tốt",
            "avg_abs_speed_change": "Trung bình |speed_factor - 1|, phản ánh mức can thiệp tốc độ của Module 4",
        },
        "baseline_summary": baseline_summary,
        "proposed_summary": proposed_summary,
        "comparison_summary": comparison_summary,
    }

    write_json(
        str(Path(args.output_dir) / "pipeline_metrics_summary.json"),
        final_output,
    )

    write_csv(
        str(Path(args.output_dir) / "baseline_metrics.csv"),
        baseline_rows,
    )

    write_csv(
        str(Path(args.output_dir) / "proposed_metrics.csv"),
        proposed_rows,
    )

    write_csv(
        str(Path(args.output_dir) / "combined_metrics.csv"),
        combined_rows,
    )

    print("\n" + "=" * 80)
    print("BASELINE VS PROPOSED METRICS")
    print("=" * 80)

    print("\nBaseline summary:")
    print(json.dumps(baseline_summary, ensure_ascii=False, indent=2))

    print("\nProposed summary:")
    print(json.dumps(proposed_summary, ensure_ascii=False, indent=2))

    print("\nComparison summary:")
    print(json.dumps(comparison_summary, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(Path(args.output_dir) / "pipeline_metrics_summary.json")
    print(Path(args.output_dir) / "baseline_metrics.csv")
    print(Path(args.output_dir) / "proposed_metrics.csv")
    print(Path(args.output_dir) / "combined_metrics.csv")


if __name__ == "__main__":
    main()