from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def natural_key(path: Path) -> list[Any]:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path))
    ]


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.rstrip("\n").strip() for line in file]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "sample"


def parse_simple_phost_yaml(path: Path) -> list[dict[str, Any]]:
    """
    PhoST text_data YAML is a simple list of inline dictionaries, for example:
    - {duration: 2.89, offset: 0.96, rW: 7, uW: 0, wav: 39095.wav}

    We parse this directly to avoid requiring PyYAML just for dataset prep.
    """
    items: list[dict[str, Any]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("-"):
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


def find_audio_path(audio_root: Path | None, split: str, wav_name: str) -> Path | None:
    if audio_root is None or not wav_name:
        return None

    candidates = [
        audio_root / split / wav_name,
        audio_root / wav_name,
        audio_root / split / Path(wav_name).stem / wav_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = list(audio_root.rglob(wav_name))
    if matches:
        return matches[0].resolve()

    return None


def convert_one_talk(
    yaml_path: Path,
    text_root: Path,
    output_dir: Path,
    audio_root: Path | None,
    max_segments_per_file: int | None,
    remaining_segments: int | None,
) -> dict[str, Any] | None:
    en_path = yaml_path.with_suffix(".en")
    vi_path = yaml_path.with_suffix(".vi")

    if not en_path.exists() or not vi_path.exists():
        print(f"[SKIP] Missing .en or .vi next to {yaml_path}")
        return None

    english_lines = read_lines(en_path)
    vietnamese_lines = read_lines(vi_path)
    yaml_items = parse_simple_phost_yaml(yaml_path)

    n = min(len(english_lines), len(vietnamese_lines), len(yaml_items))
    if max_segments_per_file is not None:
        n = min(n, max_segments_per_file)
    if remaining_segments is not None:
        n = min(n, remaining_segments)
    if n <= 0:
        return None

    if len(english_lines) != len(vietnamese_lines) or len(english_lines) != len(yaml_items):
        print("[WARN] Length mismatch:")
        print("  yaml:", yaml_path)
        print("  en lines:", len(english_lines))
        print("  vi lines:", len(vietnamese_lines))
        print("  yaml items:", len(yaml_items))
        print("  using first:", n)

    relative = yaml_path.relative_to(text_root)
    split = relative.parts[0] if len(relative.parts) > 0 else "unknown"
    talk_id = safe_id(relative.with_suffix("").as_posix())
    output_json = output_dir / f"{talk_id}.json"

    wav_name = str(yaml_items[0].get("wav", "")) if yaml_items else ""
    audio_path = find_audio_path(audio_root, split, wav_name)

    chunks: list[dict[str, Any]] = []
    total_duration = 0.0

    for idx in range(n):
        meta = yaml_items[idx]
        offset = float(meta["offset"])
        duration = float(meta["duration"])
        end_time = offset + duration
        total_duration = max(total_duration, end_time)

        en_text = english_lines[idx].strip()
        vi_text = vietnamese_lines[idx].strip()

        chunks.append(
            {
                "chunk_id": idx + 1,
                "transcript": en_text,
                "transcribed_text": en_text,
                "source_text": en_text,
                "start_time": round(offset, 3),
                "end_time": round(end_time, 3),
                "stop_time": round(end_time, 3),
                "duration": round(duration, 3),
                "english_reference": en_text,
                "vietnamese_reference": vi_text,
                "reference_vi": vi_text,
                "wav": str(meta.get("wav", "")),
                "offset": round(offset, 3),
                "rW": meta.get("rW"),
                "uW": meta.get("uW"),
                "source_yaml": str(yaml_path.resolve()),
                "source_en": str(en_path.resolve()),
                "source_vi": str(vi_path.resolve()),
            }
        )

    document = {
        "dataset": "PhoST",
        "sample_id": talk_id,
        "split": split,
        "source_language": "English",
        "target_language": "Vietnamese",
        "source_file": wav_name,
        "audio_path": str(audio_path) if audio_path else "",
        "audio_available": audio_path is not None,
        "source_text": " ".join(english_lines[:n]),
        "reference_vi": " ".join(vietnamese_lines[:n]),
        "num_chunks": len(chunks),
        "duration": round(total_duration, 3),
        "chunks": chunks,
    }

    save_json(output_json, document)

    return {
        "sample_id": talk_id,
        "split": split,
        "output_json": str(output_json.resolve()),
        "audio_path": str(audio_path) if audio_path else "",
        "audio_available": str(audio_path is not None),
        "source_yaml": str(yaml_path.resolve()),
        "source_en": str(en_path.resolve()),
        "source_vi": str(vi_path.resolve()),
        "wav": wav_name,
        "num_chunks": len(chunks),
        "duration": round(total_duration, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PhoST text_data into Module-1-compatible JSON files."
    )
    parser.add_argument(
        "--text-root",
        type=Path,
        required=True,
        help="PhoST text_data folder, e.g. D:\\datasets\\PhoST_subset\\text_data",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="test",
        help="PhoST split to convert.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Optional PhoST audio root. If omitted, JSON is still generated from text/timestamps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/phost_module1"),
        help="Output directory for converted JSON files.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-segments-per-file", type=int, default=None)
    parser.add_argument("--max-total-segments", type=int, default=None)
    args = parser.parse_args()

    text_root = args.text_root.resolve()
    output_dir = args.output_dir.resolve()
    audio_root = args.audio_root.resolve() if args.audio_root else None

    if not text_root.exists():
        raise FileNotFoundError(f"PhoST text_root not found: {text_root}")
    if audio_root is not None and not audio_root.exists():
        raise FileNotFoundError(f"PhoST audio_root not found: {audio_root}")

    if args.split == "all":
        yaml_files = sorted(text_root.glob("*/*/*.yaml"), key=natural_key)
    else:
        yaml_files = sorted((text_root / args.split).glob("*/*.yaml"), key=natural_key)

    if args.max_files is not None:
        yaml_files = yaml_files[: args.max_files]
    if not yaml_files:
        raise RuntimeError(f"No PhoST YAML files found under: {text_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    total_segments = 0

    print("=" * 90)
    print("PhoST -> Module 1 JSON")
    print("Text root :", text_root)
    print("Audio root:", audio_root if audio_root else "(not provided)")
    print("Split     :", args.split)
    print("Files     :", len(yaml_files))
    print("Output    :", output_dir)
    print("=" * 90)

    for index, yaml_path in enumerate(yaml_files, start=1):
        remaining = None
        if args.max_total_segments is not None:
            remaining = args.max_total_segments - total_segments
            if remaining <= 0:
                break

        result = convert_one_talk(
            yaml_path=yaml_path,
            text_root=text_root,
            output_dir=output_dir,
            audio_root=audio_root,
            max_segments_per_file=args.max_segments_per_file,
            remaining_segments=remaining,
        )
        if result is None:
            continue

        manifest_rows.append(result)
        total_segments += int(result["num_chunks"])
        print(
            f"[{index}/{len(yaml_files)}] {result['sample_id']} -> "
            f"{result['num_chunks']} chunks, audio={result['audio_available']}"
        )

    manifest_path = output_dir / "manifest.csv"
    list_path = output_dir / "module1_json_files.txt"

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with list_path.open("w", encoding="utf-8") as file:
        for row in manifest_rows:
            file.write(row["output_json"] + "\n")

    print("\n" + "=" * 90)
    print("DONE")
    print("Manifest :", manifest_path)
    print("JSON list:", list_path)
    print("Num files:", len(manifest_rows))
    print("Segments :", total_segments)
    if not any(row["audio_available"] == "True" for row in manifest_rows):
        print(
            "Note: no PhoST audio found. This is still usable for translation/chunking/"
            "TTS-duration evaluation with oracle timestamps, but not for ASR evaluation."
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
