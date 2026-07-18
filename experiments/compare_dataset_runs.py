from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def duration_error_ratio(tts_duration: float, source_duration: float) -> float:
    if source_duration <= 0:
        return 0.0
    return abs(tts_duration - source_duration) / source_duration


def normalize_baseline(sample_id: str, path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    rows: list[dict[str, Any]] = []

    for item in data.get("results", []):
        source_duration = safe_float(item.get("source_duration"))
        tts_duration = safe_float(item.get("tts_duration"))
        dur_error = item.get("duration_error_ratio")
        if dur_error is None:
            dur_error = duration_error_ratio(tts_duration, source_duration)

        rows.append(
            {
                "sample_id": sample_id,
                "system": "baseline",
                "id": item.get("chunk_id"),
                "source_text": item.get("source_text", ""),
                "translation": item.get("baseline_translation", ""),
                "source_duration": source_duration,
                "raw_tts_duration": tts_duration,
                "final_tts_duration": tts_duration,
                "duration_error_ratio": safe_float(dur_error),
                "speed_factor": 1.0,
                "overrun": int(tts_duration > source_duration),
                "output_json": str(path),
            }
        )

    return rows


def normalize_proposed(sample_id: str, path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    rows: list[dict[str, Any]] = []

    for item in data.get("results", []):
        source_duration = safe_float(item.get("source_duration"))
        raw_duration = safe_float(item.get("raw_tts_duration"))
        final_duration = safe_float(item.get("final_tts_duration"), raw_duration)
        dur_error = item.get("duration_error_ratio")
        if dur_error is None:
            dur_error = duration_error_ratio(final_duration, source_duration)

        rows.append(
            {
                "sample_id": sample_id,
                "system": "proposed",
                "id": item.get("segment_id"),
                "source_text": item.get("source_text", ""),
                "translation": item.get("best_translation", ""),
                "source_duration": source_duration,
                "raw_tts_duration": raw_duration,
                "final_tts_duration": final_duration,
                "duration_error_ratio": safe_float(dur_error),
                "speed_factor": safe_float(item.get("speed_factor"), 1.0),
                "overrun": int(final_duration > source_duration),
                "output_json": str(path),
            }
        )

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [safe_float(row["duration_error_ratio"]) for row in rows]
    speeds = [abs(safe_float(row["speed_factor"], 1.0) - 1.0) for row in rows]
    overruns = [safe_float(row["overrun"]) for row in rows]

    return {
        "num_items": len(rows),
        "avg_duration_error_ratio": round(safe_mean(errors), 4),
        "median_duration_error_ratio": round(safe_median(errors), 4),
        "max_duration_error_ratio": round(max(errors) if errors else 0.0, 4),
        "overrun_rate": round(safe_mean(overruns), 4),
        "avg_abs_speed_change": round(safe_mean(speeds), 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline/proposed metric JSON files across dataset samples."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--proposed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/dataset_outputs"))
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    proposed_root = args.proposed_root.resolve()
    output_dir = args.output_dir.resolve()

    baseline_rows: list[dict[str, Any]] = []
    proposed_rows: list[dict[str, Any]] = []

    for baseline_file in sorted(baseline_root.rglob("baseline_results.json")):
        sample_id = baseline_file.parent.relative_to(baseline_root).as_posix()
        baseline_rows.extend(normalize_baseline(sample_id, baseline_file))

    for proposed_file in sorted(proposed_root.rglob("module4_tts_results.json")):
        if proposed_file.parent.name == "module4":
            sibling_copy = proposed_file.parent.parent / "module4_tts_results.json"
            if sibling_copy.exists():
                continue
        sample_id = proposed_file.parent.relative_to(proposed_root).as_posix()
        if sample_id.endswith("/module4"):
            sample_id = sample_id[: -len("/module4")]
        proposed_rows.extend(normalize_proposed(sample_id, proposed_file))

    baseline_summary = summarize(baseline_rows)
    proposed_summary = summarize(proposed_rows)
    baseline_error = baseline_summary["avg_duration_error_ratio"]
    proposed_error = proposed_summary["avg_duration_error_ratio"]
    reduction = baseline_error - proposed_error
    reduction_percent = (reduction / baseline_error * 100.0) if baseline_error > 0 else 0.0

    summary = {
        "baseline_root": str(baseline_root),
        "proposed_root": str(proposed_root),
        "baseline_summary": baseline_summary,
        "proposed_summary": proposed_summary,
        "comparison": {
            "duration_error_reduction": round(reduction, 4),
            "duration_error_reduction_percent": round(reduction_percent, 2),
        },
    }

    write_csv(output_dir / "baseline_dataset_metrics.csv", baseline_rows)
    write_csv(output_dir / "proposed_dataset_metrics.csv", proposed_rows)
    write_csv(output_dir / "combined_dataset_metrics.csv", baseline_rows + proposed_rows)
    write_json(output_dir / "dataset_metrics_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
