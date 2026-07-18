from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def make_sample_id(input_file: Path, input_root: Path) -> str:
    """
    Tạo ID an toàn từ đường dẫn tương đối.

    Ví dụ:
    dataset_module1/ted/audio_001.json
    -> ted__audio_001
    """
    relative = input_file.relative_to(input_root)
    name = relative.with_suffix("").as_posix()
    name = re.sub(r"[^A-Za-z0-9._-]+", "__", name)

    return name.strip("._-") or "sample"


def is_completed(output_dir: Path, translation_only: bool) -> bool:
    """
    Kiểm tra sample đã hoàn thành chưa để hỗ trợ resume.
    """
    if translation_only:
        result_file = output_dir / "baseline_translations.json"
    else:
        result_file = output_dir / "baseline_results.json"

    if not result_file.exists():
        return False

    try:
        data = load_json(result_file)
        return data.get("status") == "complete"
    except Exception:
        return False


def run_one_sample(
    baseline_script: Path,
    input_json: Path,
    output_dir: Path,
    model: str,
    voice: str,
    rate: str,
    sample_rate: int,
    translation_only: bool,
    force_translate: bool,
    force_tts: bool,
    max_chunks: int | None,
) -> int:
    command = [
        sys.executable,
        str(baseline_script),
        "--input",
        str(input_json),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--voice",
        voice,
        "--rate",
        rate,
        "--sample-rate",
        str(sample_rate),
    ]

    if translation_only:
        command.append("--translation-only")

    if force_translate:
        command.append("--force-translate")

    if force_tts:
        command.append("--force-tts")

    if max_chunks is not None:
        command.extend(["--max-chunks", str(max_chunks)])

    print("\n" + "=" * 90)
    print("INPUT :", input_json)
    print("OUTPUT:", output_dir)
    print("CMD   :", subprocess.list2cmdline(command))
    print("=" * 90)

    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline pipeline for all Module 1 JSON files."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Thư mục chứa các JSON output của Module 1.",
    )

    parser.add_argument(
        "--baseline-script",
        type=Path,
        default=Path("baseline_pipeline/run_baseline.py"),
        help="Đường dẫn tới run_baseline.py.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment_outputs/baseline"),
        help="Thư mục output chung của dataset.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Pattern tìm input JSON.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:2b",
    )

    parser.add_argument(
        "--voice",
        type=str,
        default="vi-VN-HoaiMyNeural",
    )

    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ chạy N file đầu tiên.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Bỏ qua N file đầu tiên.",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Chỉ chạy tối đa N chunks trong mỗi file.",
    )

    parser.add_argument(
        "--translation-only",
        action="store_true",
        help="Chỉ chạy dịch, chưa chạy TTS.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Chạy lại cả sample đã complete.",
    )

    parser.add_argument(
        "--force-translate",
        action="store_true",
        help="Ép dịch lại.",
    )

    parser.add_argument(
        "--force-tts",
        action="store_true",
        help="Ép tạo lại TTS.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Dừng toàn dataset khi một sample lỗi.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    baseline_script = args.baseline_script.resolve()
    output_root = args.output_root.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy input directory: {input_dir}"
        )

    if not baseline_script.exists():
        raise FileNotFoundError(
            f"Không tìm thấy run_baseline.py: {baseline_script}"
        )

    input_files = sorted(input_dir.rglob(args.pattern))
    input_files = [
        path for path in input_files
        if path.is_file()
    ]

    input_files = input_files[args.start_index:]

    if args.limit is not None:
        input_files = input_files[:args.limit]

    if not input_files:
        raise RuntimeError(
            f"Không tìm thấy file {args.pattern} trong {input_dir}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "dataset_run_summary.json"
    results: list[dict[str, Any]] = []

    print("=" * 90)
    print("BASELINE DATASET RUNNER")
    print("Input directory:", input_dir)
    print("Number of files:", len(input_files))
    print("Translation only:", args.translation_only)
    print("=" * 90)

    for index, input_json in enumerate(input_files, start=1):
        sample_id = make_sample_id(input_json, input_dir)
        sample_output = output_root / sample_id

        print(
            f"\n[{index}/{len(input_files)}] "
            f"Processing: {sample_id}"
        )

        if (
            not args.force
            and is_completed(
                sample_output,
                translation_only=args.translation_only,
            )
        ):
            print("[SKIP] Sample đã complete.")

            results.append(
                {
                    "sample_id": sample_id,
                    "input_json": str(input_json),
                    "output_dir": str(sample_output),
                    "status": "skipped_complete",
                    "runtime_seconds": 0.0,
                }
            )

            continue

        sample_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        start_time = time.perf_counter()

        try:
            return_code = run_one_sample(
                baseline_script=baseline_script,
                input_json=input_json,
                output_dir=sample_output,
                model=args.model,
                voice=args.voice,
                rate=args.rate,
                sample_rate=args.sample_rate,
                translation_only=args.translation_only,
                force_translate=args.force_translate,
                force_tts=args.force_tts,
                max_chunks=args.max_chunks,
            )

            runtime = time.perf_counter() - start_time

            status = (
                "complete"
                if return_code == 0
                else "failed"
            )

            result_item = {
                "sample_id": sample_id,
                "input_json": str(input_json),
                "output_dir": str(sample_output),
                "status": status,
                "return_code": return_code,
                "runtime_seconds": round(runtime, 3),
            }

            results.append(result_item)

            if status == "failed":
                print(
                    f"[ERROR] Sample {sample_id} lỗi, "
                    f"return code = {return_code}"
                )

                if args.stop_on_error:
                    break

        except KeyboardInterrupt:
            print("\n[STOP] Người dùng dừng chương trình.")

            save_json(
                summary_path,
                {
                    "results": results,
                    "status": "interrupted",
                },
            )

            raise

        except Exception as error:
            runtime = time.perf_counter() - start_time

            print("[ERROR]", error)

            results.append(
                {
                    "sample_id": sample_id,
                    "input_json": str(input_json),
                    "output_dir": str(sample_output),
                    "status": "failed",
                    "error": str(error),
                    "runtime_seconds": round(runtime, 3),
                }
            )

            if args.stop_on_error:
                break

        save_json(
            summary_path,
            {
                "input_dir": str(input_dir),
                "output_root": str(output_root),
                "translation_only": args.translation_only,
                "num_selected": len(input_files),
                "num_complete": sum(
                    item["status"]
                    in {"complete", "skipped_complete"}
                    for item in results
                ),
                "num_failed": sum(
                    item["status"] == "failed"
                    for item in results
                ),
                "results": results,
            },
        )

    print("\n" + "=" * 90)
    print("DATASET RUN FINISHED")
    print("Summary:", summary_path)
    print("=" * 90)


if __name__ == "__main__":
    main()
