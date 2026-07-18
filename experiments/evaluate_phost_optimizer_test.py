from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "module_2_semantic_chunking"))

from semantic_chunk_buffer import ASRChunk, SemanticChunkBuffer  # noqa: E402


FEATURE_NAMES = [
    "wen",
    "char_count",
    "duration",
    "speech_rate",
    "has_subordinate",
    "is_question",
    "has_number",
    "has_proper_noun",
    "punctuation_count",
]


def natural_key(path: Path) -> list[Any]:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path))
    ]


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.rstrip("\n").strip() for line in file]


def parse_simple_phost_yaml(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        match = re.search(r"\{(.*)\}", line)
        if not match:
            continue
        item: dict[str, Any] = {}
        for part in match.group(1).split(","):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in {"duration", "offset"}:
                item[key] = float(value)
            elif key in {"rW", "uW"}:
                item[key] = int(float(value))
            else:
                item[key] = value
        if {"duration", "offset", "wav"}.issubset(item):
            items.append(item)
    return items


def count_english_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def count_units(text: str) -> int:
    clean = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return len([unit for unit in clean.split() if unit.strip()])


def extract_features(en_text: str, duration: float) -> dict[str, float]:
    lower = en_text.lower()
    wen = count_english_words(en_text)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", en_text)
    return {
        "wen": float(wen),
        "char_count": float(len(en_text)),
        "duration": float(duration),
        "speech_rate": float(wen / max(duration, 1e-6)),
        "has_subordinate": float(
            any(
                word in lower
                for word in [
                    "although",
                    "because",
                    "while",
                    "if",
                    "when",
                    "that",
                    "unless",
                    "before",
                    "after",
                    "since",
                ]
            )
        ),
        "is_question": float(
            "?" in en_text
            or lower.startswith(
                (
                    "what",
                    "why",
                    "how",
                    "when",
                    "where",
                    "who",
                    "do",
                    "does",
                    "did",
                    "can",
                    "could",
                    "would",
                    "should",
                )
            )
        ),
        "has_number": float(bool(re.search(r"\d+", en_text))),
        "has_proper_noun": float(len(proper_nouns) > 1),
        "punctuation_count": float(len(re.findall(r"[,.?!;:]", en_text))),
    }


def rule_based_pred_len(en_text: str, duration: float) -> int:
    features = extract_features(en_text, duration)
    k = 1.25
    if features["wen"] <= 3:
        k += 0.20
    if features["has_subordinate"]:
        k += 0.15
    if features["is_question"]:
        k += 0.10
    if features["has_number"]:
        k -= 0.05
    if features["has_proper_noun"]:
        k -= 0.05
    if features["speech_rate"] > 3.5:
        k -= 0.10
    return max(1, round(features["wen"] * max(0.9, min(k, 1.7))))


def learned_pred_len(config: dict[str, Any], en_text: str, duration: float) -> int:
    model = config["length_model"]
    features = extract_features(en_text, duration)
    value = float(model["intercept"])
    for index, name in enumerate(model["feature_names"]):
        raw = features[name]
        scaled = (raw - model["feature_means"][index]) / max(model["feature_scales"][index], 1e-8)
        value += model["coefficients"][index] * scaled
    return max(1, int(round(value)))


def load_phost_rows(text_root: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for yaml_path in sorted((text_root / split).glob("*/*.yaml"), key=natural_key):
        en_path = yaml_path.with_suffix(".en")
        vi_path = yaml_path.with_suffix(".vi")
        if not en_path.exists() or not vi_path.exists():
            continue
        en_lines = read_lines(en_path)
        vi_lines = read_lines(vi_path)
        yaml_items = parse_simple_phost_yaml(yaml_path)
        n = min(len(en_lines), len(vi_lines), len(yaml_items))
        for index in range(n):
            duration = float(yaml_items[index]["duration"])
            en_text = en_lines[index]
            vi_text = vi_lines[index]
            if duration <= 0 or not en_text or not vi_text:
                continue
            rows.append(
                {
                    "sample_id": yaml_path.parent.name,
                    "segment_index": index + 1,
                    "en": en_text,
                    "vi": vi_text,
                    "duration": duration,
                    "target_units": count_units(vi_text),
                }
            )
    return rows


def fragment_row(row: dict[str, Any], fragment_words: int) -> list[ASRChunk]:
    words = row["en"].split()
    duration = float(row["duration"])
    chunks: list[ASRChunk] = []
    if not words:
        return chunks
    for chunk_id, start in enumerate(range(0, len(words), fragment_words), start=1):
        part = words[start : start + fragment_words]
        start_ratio = start / len(words)
        end_ratio = min(start + len(part), len(words)) / len(words)
        chunks.append(
            ASRChunk(
                chunk_id=chunk_id,
                text=" ".join(part),
                start_time=duration * start_ratio,
                end_time=duration * end_ratio,
                duration=duration * (end_ratio - start_ratio),
            )
        )
    return chunks


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "max": round(max(values), 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(rows: list[dict[str, Any]], config: dict[str, Any], fragment_words: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = config["recommended_runtime"]
    tuned_chunking = config["semantic_chunking"]

    default_buffer = SemanticChunkBuffer(
        min_words=4,
        max_words=45,
        max_duration=12.0,
        max_chunks=6,
        use_ollama_judge=False,
    )
    tuned_buffer = SemanticChunkBuffer(
        min_words=int(tuned_chunking["min_words"]),
        max_words=int(tuned_chunking["max_words"]),
        max_duration=float(tuned_chunking["max_duration"]),
        max_chunks=int(tuned_chunking["max_chunks"]),
        use_ollama_judge=False,
    )

    detail_rows: list[dict[str, Any]] = []
    rule_abs = []
    learned_abs = []
    rule_pct = []
    learned_pct = []
    default_constraint_abs = []
    tuned_constraint_abs = []
    default_under = []
    tuned_under = []
    default_over = []
    tuned_over = []

    expected_segments = len(rows)
    default_produced = 0
    tuned_produced = 0
    default_force = 0
    tuned_force = 0

    for row in rows:
        target = max(1, int(row["target_units"]))
        rule_pred = rule_based_pred_len(row["en"], row["duration"])
        learned_pred = learned_pred_len(config, row["en"], row["duration"])

        default_max = max(1, math.floor(min(rule_pred + 1, row["duration"] * 5.0)))
        tuned_max = max(
            1,
            math.floor(
                min(
                    learned_pred + int(runtime["margin"]),
                    row["duration"] * float(runtime["tts_ceiling"]),
                )
            ),
        )

        rule_abs.append(abs(rule_pred - target))
        learned_abs.append(abs(learned_pred - target))
        rule_pct.append(abs(rule_pred - target) / target)
        learned_pct.append(abs(learned_pred - target) / target)
        default_constraint_abs.append(abs(default_max - target))
        tuned_constraint_abs.append(abs(tuned_max - target))
        default_under.append(max(0, target - default_max))
        tuned_under.append(max(0, target - tuned_max))
        default_over.append(max(0, default_max - target))
        tuned_over.append(max(0, tuned_max - target))

        chunks = fragment_row(row, fragment_words)
        default_segments = default_buffer.process_chunks(chunks)
        tuned_segments = tuned_buffer.process_chunks(chunks)
        default_produced += len(default_segments)
        tuned_produced += len(tuned_segments)
        default_force += sum(1 for seg in default_segments if seg.release_mode == "FORCE_RELEASE")
        tuned_force += sum(1 for seg in tuned_segments if seg.release_mode == "FORCE_RELEASE")

        detail_rows.append(
            {
                "sample_id": row["sample_id"],
                "segment_index": row["segment_index"],
                "duration": row["duration"],
                "target_units": target,
                "rule_pred_len": rule_pred,
                "learned_pred_len": learned_pred,
                "default_max_vi": default_max,
                "tuned_max_vi": tuned_max,
                "rule_abs_error": abs(rule_pred - target),
                "learned_abs_error": abs(learned_pred - target),
                "default_constraint_abs_error": abs(default_max - target),
                "tuned_constraint_abs_error": abs(tuned_max - target),
            }
        )

    summary = {
        "num_test_rows": len(rows),
        "length_prediction": {
            "baseline_rule_abs_error": summarize(rule_abs),
            "proposed_learned_abs_error": summarize(learned_abs),
            "baseline_rule_mape": round(mean(rule_pct), 4),
            "proposed_learned_mape": round(mean(learned_pct), 4),
            "mae_reduction_units": round(mean(rule_abs) - mean(learned_abs), 4),
            "mae_reduction_percent": round((mean(rule_abs) - mean(learned_abs)) / max(mean(rule_abs), 1e-8) * 100.0, 2),
        },
        "adaptive_length_constraint": {
            "baseline_default_constraint_abs_error": summarize(default_constraint_abs),
            "proposed_tuned_constraint_abs_error": summarize(tuned_constraint_abs),
            "baseline_under_limit_mae": round(mean(default_under), 4),
            "proposed_under_limit_mae": round(mean(tuned_under), 4),
            "baseline_over_limit_mae": round(mean(default_over), 4),
            "proposed_over_limit_mae": round(mean(tuned_over), 4),
            "constraint_mae_reduction_units": round(mean(default_constraint_abs) - mean(tuned_constraint_abs), 4),
            "constraint_mae_reduction_percent": round((mean(default_constraint_abs) - mean(tuned_constraint_abs)) / max(mean(default_constraint_abs), 1e-8) * 100.0, 2),
        },
        "semantic_chunking_simulation": {
            "fragment_words": fragment_words,
            "expected_segments": expected_segments,
            "baseline_default_produced_segments": default_produced,
            "proposed_tuned_produced_segments": tuned_produced,
            "baseline_boundary_error": round(abs(default_produced - expected_segments) / expected_segments, 4),
            "proposed_boundary_error": round(abs(tuned_produced - expected_segments) / expected_segments, 4),
            "baseline_force_release_rate": round(default_force / max(default_produced, 1), 4),
            "proposed_force_release_rate": round(tuned_force / max(tuned_produced, 1), 4),
        },
        "runtime_config_used": runtime,
        "semantic_chunking_config_used": tuned_chunking,
    }
    return summary, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PhoST module optimizer config on test split."
    )
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/phost_optimizer_test_outputs"))
    parser.add_argument("--fragment-words", type=int, default=4)
    args = parser.parse_args()

    text_root = args.text_root.resolve()
    config_path = args.optimizer_config.resolve()
    output_dir = args.output_dir.resolve()

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    rows = load_phost_rows(text_root, args.split)
    summary, detail_rows = evaluate(rows, config, args.fragment_words)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "optimizer_test_metrics_summary.json"
    detail_path = output_dir / "optimizer_test_metrics_detail.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    write_csv(detail_path, detail_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:", summary_path)
    print("Saved:", detail_path)


if __name__ == "__main__":
    main()
