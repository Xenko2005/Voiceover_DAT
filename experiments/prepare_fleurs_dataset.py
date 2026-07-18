from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


LANGUAGE_LABELS = {
    "en_us": "English",
    "ja_jp": "Japanese",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download real FLEURS audio and create Module 1-style JSON files "
            "with Vietnamese references for dataset evaluation."
        )
    )
    parser.add_argument("--source-config", default="en_us", choices=["en_us", "ja_jp"])
    parser.add_argument("--target-config", default="vi_vn")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/fleurs_eval"))
    parser.add_argument("--num-items", type=int, default=30)
    parser.add_argument(
        "--group-size",
        type=int,
        default=5,
        help="Number of FLEURS utterances concatenated into one evaluation audio.",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=0.35,
        help="Silence inserted between utterances in one grouped sample.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N examples from the selected split.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/cache the selected FLEURS configs locally instead of streaming examples.",
    )
    return parser.parse_args()


def require_dependencies() -> None:
    missing: list[str] = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")
    if missing:
        joined = " ".join(missing)
        raise RuntimeError(
            "Missing dependencies. Install them first:\n"
            f"python -m pip install {joined}"
        )


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "sample"


def as_mono_float32(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    array = np.asarray(audio["array"], dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array, int(audio["sampling_rate"])


def text_from_example(example: dict[str, Any]) -> str:
    return str(
        example.get("raw_transcription")
        or example.get("transcription")
        or ""
    ).strip()


def make_grouped_sample(
    source_examples: list[dict[str, Any]],
    target_examples: list[dict[str, Any]],
    sample_id: str,
    source_language: str,
    output_audio_dir: Path,
    output_json_dir: Path,
    silence_seconds: float,
) -> dict[str, str]:
    import soundfile as sf

    audio_parts: list[np.ndarray] = []
    chunks: list[dict[str, Any]] = []
    current_time = 0.0
    sampling_rate: int | None = None

    target_refs: list[str] = []
    source_texts: list[str] = []

    for index, (source_item, target_item) in enumerate(
        zip(source_examples, target_examples),
        start=1,
    ):
        audio, sr = as_mono_float32(source_item["audio"])
        if sampling_rate is None:
            sampling_rate = sr
        if sr != sampling_rate:
            raise RuntimeError(
                f"Mixed sample rates are not supported: {sr} != {sampling_rate}"
            )

        start_time = current_time
        duration = len(audio) / sr
        stop_time = start_time + duration
        source_text = text_from_example(source_item)
        target_text = text_from_example(target_item)

        chunks.append(
            {
                "chunk_id": index,
                "transcribed_text": source_text,
                "start_time": round(start_time, 3),
                "stop_time": round(stop_time, 3),
                "duration": round(duration, 3),
                "reference_vi": target_text,
                "fleurs_source_id": source_item.get("id"),
                "fleurs_target_id": target_item.get("id"),
            }
        )
        source_texts.append(source_text)
        target_refs.append(target_text)
        audio_parts.append(audio)

        current_time = stop_time
        if index < len(source_examples) and silence_seconds > 0:
            silence = np.zeros(round(silence_seconds * sr), dtype=np.float32)
            audio_parts.append(silence)
            current_time += len(silence) / sr

    if sampling_rate is None:
        raise RuntimeError("Empty sample group.")

    grouped_audio = np.concatenate(audio_parts) if audio_parts else np.zeros(1, dtype=np.float32)
    audio_path = output_audio_dir / f"{sample_id}.wav"
    json_path = output_json_dir / f"{sample_id}.json"
    sf.write(audio_path, grouped_audio, sampling_rate)

    payload = {
        "dataset": "google/fleurs",
        "source_file": str(audio_path),
        "source_language": source_language,
        "reference_vi": " ".join(target_refs),
        "source_text": " ".join(source_texts),
        "chunks": chunks,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "id": sample_id,
        "audio_path": str(audio_path),
        "module1_json": str(json_path),
        "source_language": source_language,
        "reference_vi": payload["reference_vi"],
        "source_text": payload["source_text"],
        "duration_s": f"{len(grouped_audio) / sampling_rate:.3f}",
        "num_chunks": str(len(chunks)),
    }


def load_fleurs_with_retry(
    config_name: str,
    split: str,
    streaming: bool,
    attempts: int = 4,
):
    from datasets import load_dataset

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(
                "google/fleurs",
                config_name,
                split=split,
                streaming=streaming,
            )
        except Exception as exc:
            last_error = exc
            wait_seconds = min(20, 3 * attempt)
            print(
                f"[WARN] FLEURS load failed for {config_name} "
                f"(attempt {attempt}/{attempts}): {type(exc).__name__}: {exc}"
            )
            if attempt < attempts:
                print(f"Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
    raise RuntimeError(
        f"Cannot load google/fleurs config={config_name}, split={split}. "
        "This is usually a temporary Hugging Face connection issue. "
        "Try again later, set HF_TOKEN for higher rate limits, or use --no-streaming."
    ) from last_error


def main() -> None:
    require_dependencies()

    args = parse_args()
    if args.num_items <= 0:
        raise ValueError("--num-items must be positive.")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")

    source_language = LANGUAGE_LABELS.get(args.source_config, args.source_config)
    root = args.output_dir / f"{args.source_config}_to_{args.target_config}_{args.split}"
    audio_dir = root / "audio"
    json_dir = root / "module1_json"
    root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    streaming = not args.no_streaming
    print(
        f"Loading FLEURS source: google/fleurs {args.source_config} {args.split} "
        f"(streaming={streaming})"
    )
    source_dataset = load_fleurs_with_retry(
        args.source_config,
        args.split,
        streaming=streaming,
    )
    print(
        f"Loading FLEURS target: google/fleurs {args.target_config} {args.split} "
        f"(streaming={streaming})"
    )
    target_dataset = load_fleurs_with_retry(
        args.target_config,
        args.split,
        streaming=streaming,
    )

    total_examples = args.num_items * args.group_size
    group_count = math.ceil(total_examples / args.group_size)
    paired_examples = iter(zip(iter(source_dataset), iter(target_dataset)))
    for _ in range(args.start_index):
        try:
            next(paired_examples)
        except StopIteration:
            break

    manifest_rows: list[dict[str, str]] = []
    for group_index in range(group_count):
        source_examples = []
        target_examples = []
        for _ in range(args.group_size):
            try:
                source_item, target_item = next(paired_examples)
            except StopIteration:
                break
            source_examples.append(source_item)
            target_examples.append(target_item)
        if not source_examples:
            break
        sample_id = safe_id(
            f"fleurs_{args.source_config}_vi_{args.split}_{group_index + 1:04d}"
        )
        row = make_grouped_sample(
            source_examples=source_examples,
            target_examples=target_examples,
            sample_id=sample_id,
            source_language=source_language,
            output_audio_dir=audio_dir,
            output_json_dir=json_dir,
            silence_seconds=args.silence_seconds,
        )
        manifest_rows.append(row)
        print(
            f"[{len(manifest_rows)}/{group_count}] {row['id']} "
            f"duration={row['duration_s']}s chunks={row['num_chunks']}"
        )

    manifest_path = root / "manifest.csv"
    module1_list_path = root / "module1_json_files.txt"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "audio_path",
                "module1_json",
                "source_language",
                "reference_vi",
                "source_text",
                "duration_s",
                "num_chunks",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    module1_list_path.write_text(
        "\n".join(row["module1_json"] for row in manifest_rows),
        encoding="utf-8",
    )

    print("\nSaved FLEURS evaluation dataset:")
    print("Root:", root)
    print("Manifest:", manifest_path)
    print("Module 1 JSON folder:", json_dir)
    print("Module 1 JSON list:", module1_list_path)
    print("\nNext step examples:")
    print(
        "python module_2_semantic_chunking/run_module2.py "
        f'--input "{manifest_rows[0]["module1_json"]}"'
    )
    print(
        "python experiments/run_baseline_dataset.py "
        f'--input-dir "{json_dir}" --translation-only --limit 3'
    )


if __name__ == "__main__":
    main()
