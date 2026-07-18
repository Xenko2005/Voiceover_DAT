from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


def natural_key(path: Path):
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
    return text.strip("._-")


def convert_yaml_file(
    yaml_path: Path,
    text_root: Path,
    output_root: Path,
) -> dict[str, Any] | None:
    en_path = yaml_path.with_suffix(".en")
    vi_path = yaml_path.with_suffix(".vi")

    if not en_path.exists() or not vi_path.exists():
        print(f"[SKIP] Missing .en or .vi for {yaml_path}")
        return None

    en_lines = read_lines(en_path)
    vi_lines = read_lines(vi_path)

    with yaml_path.open("r", encoding="utf-8") as file:
        meta_items = yaml.safe_load(file)

    if not isinstance(meta_items, list):
        print(f"[SKIP] YAML is not list: {yaml_path}")
        return None

    n = min(len(en_lines), len(vi_lines), len(meta_items))

    if n == 0:
        print(f"[SKIP] Empty file: {yaml_path}")
        return None

    if len(en_lines) != len(vi_lines) or len(en_lines) != len(meta_items):
        print("[WARN] Length mismatch:")
        print("  yaml:", yaml_path)
        print("  en:", len(en_lines))
        print("  vi:", len(vi_lines))
        print("  yaml items:", len(meta_items))
        print("  using first:", n)

    relative = yaml_path.relative_to(text_root)
    sample_id = safe_id(relative.with_suffix("").as_posix())

    output_json = output_root / f"{sample_id}.json"

    chunks: list[dict[str, Any]] = []

    for idx in range(n):
        meta = meta_items[idx]

        offset = float(meta["offset"])
        duration = float(meta["duration"])
        end_time = offset + duration

        en_text = en_lines[idx]
        vi_text = vi_lines[idx]

        chunks.append(
            {
                "chunk_id": idx + 1,

                "transcript": en_text,
                "start_time": round(offset, 3),
                "end_time": round(end_time, 3),
                "duration": round(duration, 3),

                "english_reference": en_text,
                "vietnamese_reference": vi_text,

                "wav": meta.get("wav"),
                "offset": round(offset, 3),
                "rW": meta.get("rW"),
                "uW": meta.get("uW"),

                "source_yaml": str(yaml_path),
                "source_en": str(en_path),
                "source_vi": str(vi_path),
            }
        )

    save_json(output_json, chunks)

    return {
        "sample_id": sample_id,
        "source_yaml": str(yaml_path),
        "source_en": str(en_path),
        "source_vi": str(vi_path),
        "output_json": str(output_json),
        "num_chunks": len(chunks),
        "wav": chunks[0].get("wav") if chunks else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PhoST text_data into Module-1-compatible JSON files."
    )

    parser.add_argument(
        "--text-root",
        type=Path,
        required=True,
        help="Path to PhoST text_data folder.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output folder for converted JSON files.",
    )

    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Only convert first N yaml files for testing.",
    )

    args = parser.parse_args()

    text_root = args.text_root.resolve()
    output_root = args.output_root.resolve()

    if not text_root.exists():
        raise FileNotFoundError(f"text_root not found: {text_root}")

    yaml_files = sorted(text_root.rglob("*.yaml"), key=natural_key)

    if args.limit_files is not None:
        yaml_files = yaml_files[: args.limit_files]

    if not yaml_files:
        raise RuntimeError(f"No YAML files found under {text_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    total_chunks = 0

    print("=" * 80)
    print("PhoST text conversion")
    print("Text root:", text_root)
    print("Output root:", output_root)
    print("YAML files:", len(yaml_files))
    print("=" * 80)

    for i, yaml_path in enumerate(yaml_files, start=1):
        result = convert_yaml_file(
            yaml_path=yaml_path,
            text_root=text_root,
            output_root=output_root,
        )

        if result is None:
            continue

        manifest_rows.append(result)
        total_chunks += int(result["num_chunks"])

        print(
            f"[{i}/{len(yaml_files)}] "
            f"{result['sample_id']} -> {result['num_chunks']} chunks"
        )

    manifest_path = output_root / "manifest.csv"

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "source_yaml",
                "source_en",
                "source_vi",
                "output_json",
                "num_chunks",
                "wav",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 80)
    print("DONE")
    print("Manifest:", manifest_path)
    print("JSON files:", len(manifest_rows))
    print("Total chunks:", total_chunks)
    print("=" * 80)


if __name__ == "__main__":
    main()