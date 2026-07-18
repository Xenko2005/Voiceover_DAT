from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_vi_units(text: str) -> int:
    clean = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return len([unit for unit in clean.split() if unit.strip()])


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


def read_module1_chunks(module1_root: Path, sample_id: str) -> dict[int, dict[str, Any]]:
    path = module1_root / f"{sample_id}.json"
    if not path.exists():
        return {}
    data = load_json(path)
    return {int(chunk["chunk_id"]): chunk for chunk in data.get("chunks", [])}


def reference_for_chunk_ids(
    module1_chunks: dict[int, dict[str, Any]],
    chunk_ids: list[int],
) -> str:
    refs = []
    for chunk_id in chunk_ids:
        chunk = module1_chunks.get(int(chunk_id), {})
        ref = chunk.get("reference_vi") or chunk.get("vietnamese_reference") or ""
        if ref:
            refs.append(str(ref).strip())
    return " ".join(refs).strip()


def translation_metrics(
    sample_id: str,
    system: str,
    source_text: str,
    translation: str,
    reference_vi: str,
    duration: float,
    output_json: Path,
    extra: dict[str, Any] | None = None,
    tts_ceiling: float = 6.5,
) -> dict[str, Any]:
    translation_units = count_vi_units(translation)
    valid_translation = bool(translation.strip()) and not translation.strip().startswith("[ERROR]")
    reference_units = max(1, count_vi_units(reference_vi))
    budget_units = max(1.0, duration * tts_ceiling)
    ref_abs_error = abs(translation_units - reference_units)
    budget_over_units = max(0.0, translation_units - budget_units)

    row = {
        "sample_id": sample_id,
        "system": system,
        "source_text": source_text,
        "translation": translation,
        "reference_vi": reference_vi,
        "duration": round(duration, 3),
        "translation_units": translation_units,
        "valid_translation": int(valid_translation),
        "reference_units": reference_units,
        "duration_budget_units": round(budget_units, 3),
        "ref_length_abs_error": ref_abs_error,
        "ref_length_abs_error_ratio": ref_abs_error / reference_units,
        "budget_over_units": round(budget_over_units, 3),
        "budget_overrun": int(budget_over_units > 0),
        "output_json": str(output_json),
    }
    if extra:
        row.update(extra)
    return row


def collect_baseline_rows(
    baseline_root: Path,
    module1_root: Path,
    tts_ceiling: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(baseline_root.rglob("baseline_translations.json")):
        sample_id = path.parent.relative_to(baseline_root).as_posix()
        if sample_id.startswith("_"):
            continue
        data = load_json(path)
        module1_chunks = read_module1_chunks(module1_root, sample_id)
        for item in data.get("results", []):
            chunk_id = int(item.get("chunk_id", 0))
            reference_vi = reference_for_chunk_ids(module1_chunks, [chunk_id])
            rows.append(
                translation_metrics(
                    sample_id=sample_id,
                    system="baseline",
                    source_text=str(item.get("source_text", "")),
                    translation=str(item.get("baseline_translation", "")),
                    reference_vi=reference_vi,
                    duration=safe_float(item.get("source_duration")),
                    output_json=path,
                    tts_ceiling=tts_ceiling,
                )
            )
    return rows


def collect_proposed_rows(
    proposed_root: Path,
    module1_root: Path,
    tts_ceiling: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(proposed_root.rglob("module3_translation_results.json")):
        sample_id = path.parent.relative_to(proposed_root).as_posix()
        if sample_id.startswith("_"):
            continue
        data = load_json(path)
        module1_chunks = read_module1_chunks(module1_root, sample_id)

        module2_path = path.parent / "module2_semantic_segments.json"
        source_chunk_ids_by_segment: dict[int, list[int]] = {}
        release_mode_by_segment: dict[int, str] = {}
        release_reason_by_segment: dict[int, str] = {}
        if module2_path.exists():
            module2 = load_json(module2_path)
            for segment in module2.get("segments", []):
                segment_id = int(segment.get("segment_id", 0))
                source_chunk_ids_by_segment[segment_id] = [
                    int(chunk_id) for chunk_id in segment.get("source_chunk_ids", [])
                ]
                release_mode_by_segment[segment_id] = str(segment.get("release_mode", ""))
                release_reason_by_segment[segment_id] = str(segment.get("release_reason", ""))

        for item in data.get("results", []):
            segment_id = int(item.get("segment_id", 0))
            chunk_ids = source_chunk_ids_by_segment.get(segment_id, [segment_id])
            reference_vi = reference_for_chunk_ids(module1_chunks, chunk_ids)
            translation_units = count_vi_units(str(item.get("best_translation", "")))
            max_vi_final = safe_float(item.get("max_vi_final"))
            extra = {
                "pred_len_vi": item.get("pred_len_vi"),
                "max_vi_final": item.get("max_vi_final"),
                "constraint_abs_error": abs(translation_units - max_vi_final),
                "best_score": item.get("best_score"),
                "release_mode": release_mode_by_segment.get(segment_id, ""),
                "release_reason": release_reason_by_segment.get(segment_id, ""),
                "num_source_chunks": len(chunk_ids),
                "source_chunk_ids": ",".join(str(chunk_id) for chunk_id in chunk_ids),
            }
            rows.append(
                translation_metrics(
                    sample_id=sample_id,
                    system="proposed",
                    source_text=str(item.get("source_text", "")),
                    translation=str(item.get("best_translation", "")),
                    reference_vi=reference_vi,
                    duration=safe_float(item.get("duration")),
                    output_json=path,
                    extra=extra,
                    tts_ceiling=tts_ceiling,
                )
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if int(row.get("valid_translation", 1)) == 1]
    ref_errors = [safe_float(row["ref_length_abs_error"]) for row in valid_rows]
    ref_error_ratios = [safe_float(row["ref_length_abs_error_ratio"]) for row in valid_rows]
    budget_over = [safe_float(row["budget_over_units"]) for row in valid_rows]
    overruns = [safe_float(row["budget_overrun"]) for row in valid_rows]
    durations = [safe_float(row["duration"]) for row in valid_rows]
    units = [safe_float(row["translation_units"]) for row in valid_rows]
    constraint_errors = [
        safe_float(row.get("constraint_abs_error"))
        for row in valid_rows
        if "constraint_abs_error" in row
    ]
    force_modes = [
        1.0
        for row in rows
        if str(row.get("release_mode", "")).upper() == "FORCE_RELEASE"
    ]

    return {
        "num_items": len(rows),
        "num_valid_items": len(valid_rows),
        "translation_fail_rate": round(1.0 - len(valid_rows) / max(len(rows), 1), 4),
        "total_audio_duration_seconds": round(sum(durations), 3),
        "avg_translation_units": round(safe_mean(units), 4),
        "avg_ref_length_abs_error": round(safe_mean(ref_errors), 4),
        "median_ref_length_abs_error": round(safe_median(ref_errors), 4),
        "avg_ref_length_abs_error_ratio": round(safe_mean(ref_error_ratios), 4),
        "avg_budget_over_units": round(safe_mean(budget_over), 4),
        "budget_overrun_rate": round(safe_mean(overruns), 4),
        "avg_constraint_abs_error": round(safe_mean(constraint_errors), 4)
        if constraint_errors
        else None,
        "force_release_rate": round(len(force_modes) / max(len(rows), 1), 4)
        if any("release_mode" in row for row in rows)
        else None,
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
        description="Compare baseline/proposed translation-only outputs without Edge TTS."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--proposed-root", type=Path, required=True)
    parser.add_argument("--module1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/translation_only_outputs"))
    parser.add_argument("--tts-ceiling", type=float, default=6.5)
    args = parser.parse_args()

    baseline_root = args.baseline_root.resolve()
    proposed_root = args.proposed_root.resolve()
    module1_root = args.module1_root.resolve()
    output_dir = args.output_dir.resolve()

    baseline_rows = collect_baseline_rows(
        baseline_root=baseline_root,
        module1_root=module1_root,
        tts_ceiling=args.tts_ceiling,
    )
    proposed_rows = collect_proposed_rows(
        proposed_root=proposed_root,
        module1_root=module1_root,
        tts_ceiling=args.tts_ceiling,
    )

    baseline_summary = summarize(baseline_rows)
    proposed_summary = summarize(proposed_rows)
    baseline_error = safe_float(baseline_summary["avg_ref_length_abs_error"])
    proposed_error = safe_float(proposed_summary["avg_ref_length_abs_error"])
    error_reduction = baseline_error - proposed_error
    error_reduction_percent = (
        error_reduction / baseline_error * 100.0
        if baseline_error > 0
        else 0.0
    )

    summary = {
        "baseline_root": str(baseline_root),
        "proposed_root": str(proposed_root),
        "module1_root": str(module1_root),
        "tts_ceiling": args.tts_ceiling,
        "baseline_summary": baseline_summary,
        "proposed_summary": proposed_summary,
        "comparison": {
            "ref_length_abs_error_reduction": round(error_reduction, 4),
            "ref_length_abs_error_reduction_percent": round(error_reduction_percent, 2),
            "budget_overrun_rate_delta": round(
                safe_float(baseline_summary["budget_overrun_rate"])
                - safe_float(proposed_summary["budget_overrun_rate"]),
                4,
            ),
        },
    }

    write_csv(output_dir / "baseline_translation_only_metrics.csv", baseline_rows)
    write_csv(output_dir / "proposed_translation_only_metrics.csv", proposed_rows)
    write_csv(output_dir / "combined_translation_only_metrics.csv", baseline_rows + proposed_rows)
    write_json(output_dir / "translation_only_metrics_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
