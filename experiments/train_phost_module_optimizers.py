from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


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
    k = max(0.9, min(k, 1.7))
    return max(1, round(features["wen"] * k))


def load_phost_rows(text_root: Path, split: str, limit_files: int | None) -> list[dict[str, Any]]:
    split_dir = text_root / split
    yaml_files = sorted(split_dir.glob("*/*.yaml"), key=natural_key)
    if limit_files is not None:
        yaml_files = yaml_files[:limit_files]

    rows: list[dict[str, Any]] = []
    for yaml_path in yaml_files:
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
            if not en_text or not vi_text or duration <= 0:
                continue
            rows.append(
                {
                    "en": en_text,
                    "vi": vi_text,
                    "duration": duration,
                    "target_units": count_units(vi_text),
                }
            )
    return rows


def rows_to_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for row in rows:
        features = extract_features(row["en"], row["duration"])
        x.append([features[name] for name in FEATURE_NAMES])
        y.append(float(row["target_units"]))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def fit_ridge_regression(
    train_rows: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    x_train, y_train = rows_to_matrix(train_rows)
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales < 1e-8] = 1.0
    x_scaled = (x_train - means) / scales
    x_aug = np.concatenate([np.ones((x_scaled.shape[0], 1)), x_scaled], axis=1)
    regularizer = np.eye(x_aug.shape[1]) * alpha
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ y_train)

    return {
        "type": "ridge_linear_regression",
        "target": "vietnamese_units",
        "alpha": alpha,
        "feature_names": FEATURE_NAMES,
        "feature_means": [float(v) for v in means],
        "feature_scales": [float(v) for v in scales],
        "intercept": float(weights[0]),
        "coefficients": [float(v) for v in weights[1:]],
    }


def predict_with_model(model: dict[str, Any], row: dict[str, Any]) -> int:
    features = extract_features(row["en"], row["duration"])
    value = float(model["intercept"])
    for index, name in enumerate(model["feature_names"]):
        raw = features[name]
        scaled = (raw - model["feature_means"][index]) / max(model["feature_scales"][index], 1e-8)
        value += model["coefficients"][index] * scaled
    return max(1, int(round(value)))


def evaluate_length_model(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    learned_errors = []
    rule_errors = []
    learned_abs_pct = []
    rule_abs_pct = []

    for row in rows:
        target = max(1, int(row["target_units"]))
        learned = predict_with_model(model, row)
        rule = rule_based_pred_len(row["en"], row["duration"])
        learned_errors.append(abs(learned - target))
        rule_errors.append(abs(rule - target))
        learned_abs_pct.append(abs(learned - target) / target)
        rule_abs_pct.append(abs(rule - target) / target)

    return {
        "learned_mae_units": round(mean(learned_errors), 4),
        "rule_mae_units": round(mean(rule_errors), 4),
        "learned_mape": round(mean(learned_abs_pct), 4),
        "rule_mape": round(mean(rule_abs_pct), 4),
    }


def tune_runtime_constraints(model: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for tts_ceiling in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]:
        for margin in [0, 1, 2, 3, 4]:
            under_errors = []
            over_errors = []
            abs_errors = []
            for row in rows:
                target = max(1, int(row["target_units"]))
                pred = predict_with_model(model, row)
                max_final = max(1, math.floor(min(pred + margin, row["duration"] * tts_ceiling)))
                under_errors.append(max(0, target - max_final))
                over_errors.append(max(0, max_final - target))
                abs_errors.append(abs(max_final - target))

            score = mean(under_errors) + 0.5 * mean(over_errors)
            candidate = {
                "tts_ceiling": tts_ceiling,
                "margin": margin,
                "constraint_score": round(score, 4),
                "constraint_mae_units": round(mean(abs_errors), 4),
                "under_limit_mae_units": round(mean(under_errors), 4),
                "over_limit_mae_units": round(mean(over_errors), 4),
            }
            if best is None or candidate["constraint_score"] < best["constraint_score"]:
                best = candidate

    assert best is not None
    return best


def fragment_row(row: dict[str, Any], fragment_words: int) -> list[ASRChunk]:
    words = row["en"].split()
    duration = float(row["duration"])
    chunks: list[ASRChunk] = []
    if not words:
        return chunks

    chunk_id = 1
    for start in range(0, len(words), fragment_words):
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
        chunk_id += 1
    return chunks


def tune_semantic_chunking(rows: list[dict[str, Any]], fragment_words: int) -> dict[str, Any]:
    grid = []
    for min_words in [3, 4, 5]:
        for max_words in [30, 45, 60]:
            for max_duration in [8.0, 10.0, 12.0]:
                for max_chunks in [4, 6, 8]:
                    grid.append((min_words, max_words, max_duration, max_chunks))

    best: dict[str, Any] | None = None
    for min_words, max_words, max_duration, max_chunks in grid:
        buffer = SemanticChunkBuffer(
            min_words=min_words,
            max_words=max_words,
            max_duration=max_duration,
            max_chunks=max_chunks,
            use_ollama_judge=False,
        )
        expected = len(rows)
        produced = 0
        force_count = 0
        for row in rows:
            segments = buffer.process_chunks(fragment_row(row, fragment_words))
            produced += len(segments)
            force_count += sum(1 for seg in segments if seg.release_mode == "FORCE_RELEASE")

        boundary_error = abs(produced - expected) / max(expected, 1)
        force_rate = force_count / max(produced, 1)
        score = boundary_error + 0.25 * force_rate
        candidate = {
            "min_words": min_words,
            "max_words": max_words,
            "max_duration": max_duration,
            "max_chunks": max_chunks,
            "simulated_boundary_error": round(boundary_error, 4),
            "force_release_rate": round(force_rate, 4),
            "score": round(score, 4),
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate

    assert best is not None
    return best


def sample_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/tune lightweight module optimizers from PhoST train/dev text_data."
    )
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("module_3_adaptive_translation/outputs/phost_optimizer_config.json"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--train-limit-files", type=int, default=300)
    parser.add_argument("--dev-limit-files", type=int, default=None)
    parser.add_argument("--train-sample-rows", type=int, default=20000)
    parser.add_argument("--dev-sample-rows", type=int, default=3000)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fragment-words", type=int, default=4)
    args = parser.parse_args()

    text_root = args.text_root.resolve()
    if not text_root.exists():
        raise FileNotFoundError(f"text_root not found: {text_root}")

    print("Loading PhoST train rows...")
    train_rows = load_phost_rows(text_root, args.train_split, args.train_limit_files)
    print("Loading PhoST dev rows...")
    dev_rows = load_phost_rows(text_root, args.dev_split, args.dev_limit_files)

    train_rows = sample_rows(train_rows, args.train_sample_rows, args.seed)
    dev_rows = sample_rows(dev_rows, args.dev_sample_rows, args.seed)
    if not train_rows or not dev_rows:
        raise RuntimeError("No train/dev rows loaded.")

    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows:   {len(dev_rows)}")

    length_model = fit_ridge_regression(train_rows, alpha=args.ridge_alpha)
    length_metrics = evaluate_length_model(length_model, dev_rows)
    runtime_constraints = tune_runtime_constraints(length_model, dev_rows)
    chunking_config = tune_semantic_chunking(dev_rows, fragment_words=args.fragment_words)

    output = {
        "dataset": "PhoST",
        "text_root": str(text_root),
        "train_split": args.train_split,
        "dev_split": args.dev_split,
        "num_train_rows": len(train_rows),
        "num_dev_rows": len(dev_rows),
        "length_model": length_model,
        "length_model_metrics": length_metrics,
        "semantic_chunking": chunking_config,
        "recommended_runtime": {
            "tts_ceiling": runtime_constraints["tts_ceiling"],
            "margin": runtime_constraints["margin"],
            "lambda_dur": 0.6,
            "mu_flu": 0.3,
        },
        "constraint_tuning": runtime_constraints,
        "notes": [
            "This trains/tunes lightweight pipeline modules, not an end-to-end neural dubbing model.",
            "Length model predicts Vietnamese unit count from English text and source duration.",
            "Semantic chunking parameters are tuned by simulated ASR fragmentation.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "output": str(args.output),
            "length_model_metrics": length_metrics,
            "semantic_chunking": chunking_config,
            "recommended_runtime": output["recommended_runtime"],
            "constraint_tuning": runtime_constraints,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
