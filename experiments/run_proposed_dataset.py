from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def make_sample_id(input_file: Path, input_root: Path) -> str:
    relative = input_file.relative_to(input_root)
    name = relative.with_suffix("").as_posix()
    name = re.sub(r"[^A-Za-z0-9._-]+", "__", name)
    return name.strip("._-") or "sample"


def run_command(command: list[str], cwd: Path) -> None:
    print("CMD:", subprocess.list2cmdline(command))
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def is_complete(sample_output: Path, run_tts: bool) -> bool:
    module3_file = sample_output / "module3_translation_results.json"
    module4_file = sample_output / "module4_tts_results.json"

    if run_tts:
        return module4_file.exists()
    return module3_file.exists()


def run_one_sample(
    project_root: Path,
    input_json: Path,
    sample_output: Path,
    model: str,
    voice: str,
    rate: str,
    sample_rate: int,
    max_segments: int | None,
    batch_size: int,
    use_ollama_judge: bool,
    run_tts: bool,
    optimizer_config_path: Path | None,
    module2_config: dict[str, Any],
) -> None:
    module2_output = sample_output / "module2_semantic_segments.json"
    module3_output_global = project_root / "module_3_adaptive_translation" / "outputs" / "module3_translation_results.json"
    module3_output_sample = sample_output / "module3_translation_results.json"
    module4_output_dir = sample_output / "module4"

    sample_output.mkdir(parents=True, exist_ok=True)

    module2_cmd = [
        sys.executable,
        str(project_root / "module_2_semantic_chunking" / "run_module2.py"),
        "--input",
        str(input_json),
        "--output",
        str(module2_output),
        "--min-words",
        str(module2_config["min_words"]),
        "--max-words",
        str(module2_config["max_words"]),
        "--max-duration",
        str(module2_config["max_duration"]),
        "--max-chunks",
        str(module2_config["max_chunks"]),
    ]
    run_command(module2_cmd, cwd=project_root)

    module3_cmd = [
        sys.executable,
        str(project_root / "module_3_adaptive_translation" / "run_module3.py"),
        "--input",
        str(module2_output),
        "--model",
        model,
        "--batch-size",
        str(batch_size),
    ]
    if max_segments is not None:
        module3_cmd.extend(["--max-segments", str(max_segments)])
    if use_ollama_judge:
        module3_cmd.append("--use-llm-judge")
    if optimizer_config_path is not None:
        module3_cmd.extend(["--optimizer-config", str(optimizer_config_path)])

    run_command(module3_cmd, cwd=project_root)

    if not module3_output_global.exists():
        raise FileNotFoundError(f"Module 3 output not found: {module3_output_global}")
    shutil.copy2(module3_output_global, module3_output_sample)

    if not run_tts:
        return

    module4_cmd = [
        sys.executable,
        str(project_root / "module_4_tts_alignment" / "run_module4.py"),
        "--input",
        str(module3_output_sample),
        "--output-dir",
        str(module4_output_dir),
        "--voice",
        voice,
        "--rate",
        rate,
        "--sample-rate",
        str(sample_rate),
    ]
    if max_segments is not None:
        module4_cmd.extend(["--max-segments", str(max_segments)])

    run_command(module4_cmd, cwd=project_root)

    module4_output_sample = sample_output / "module4_tts_results.json"
    module4_output_global = module4_output_dir / "module4_tts_results.json"
    if module4_output_global.exists():
        shutil.copy2(module4_output_global, module4_output_sample)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run proposed Module 2 -> Module 3 -> Module 4 pipeline for a dataset of Module 1 JSON files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiment_outputs/proposed"))
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--voice", default="vi-VN-HoaiMyNeural")
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--use-ollama-judge", action="store_true")
    parser.add_argument("--optimizer-config", type=Path, default=None)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--translation-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    optimizer_config_path = args.optimizer_config.resolve() if args.optimizer_config else None
    module2_config = {
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_duration": args.max_duration,
        "max_chunks": args.max_chunks,
    }

    if optimizer_config_path is not None:
        with optimizer_config_path.open("r", encoding="utf-8") as file:
            optimizer_config = json.load(file)
        tuned_chunking = optimizer_config.get("semantic_chunking", {})
        for key in ["min_words", "max_words", "max_duration", "max_chunks"]:
            if key in tuned_chunking:
                module2_config[key] = tuned_chunking[key]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(path for path in input_dir.rglob(args.pattern) if path.is_file())
    input_files = [
        path for path in input_files
        if path.name not in {"manifest.json", "manifest.csv"}
    ]
    input_files = input_files[args.start_index :]
    if args.limit is not None:
        input_files = input_files[: args.limit]
    if not input_files:
        raise RuntimeError(f"No input JSON files found in {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "dataset_run_summary.json"
    results: list[dict[str, Any]] = []
    run_tts = not args.translation_only

    print("=" * 90)
    print("PROPOSED DATASET RUNNER")
    print("Input :", input_dir)
    print("Output:", output_root)
    print("Files :", len(input_files))
    print("TTS   :", run_tts)
    print("Module2 config:", module2_config)
    if optimizer_config_path:
        print("Optimizer config:", optimizer_config_path)
    print("=" * 90)

    for index, input_json in enumerate(input_files, start=1):
        sample_id = make_sample_id(input_json, input_dir)
        sample_output = output_root / sample_id
        start_time = time.perf_counter()

        print("\n" + "=" * 90)
        print(f"[{index}/{len(input_files)}] {sample_id}")
        print("INPUT :", input_json)
        print("OUTPUT:", sample_output)
        print("=" * 90)

        if not args.force and is_complete(sample_output, run_tts=run_tts):
            print("[SKIP] Sample already complete.")
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

        try:
            run_one_sample(
                project_root=project_root,
                input_json=input_json,
                sample_output=sample_output,
                model=args.model,
                voice=args.voice,
                rate=args.rate,
                sample_rate=args.sample_rate,
                max_segments=args.max_segments,
                batch_size=args.batch_size,
                use_ollama_judge=args.use_ollama_judge,
                run_tts=run_tts,
                optimizer_config_path=optimizer_config_path,
                module2_config=module2_config,
            )
            status = "complete"
            error = ""
        except Exception as exc:
            status = "failed"
            error = str(exc)
            print("[ERROR]", error)
            if args.stop_on_error:
                raise

        runtime = time.perf_counter() - start_time
        results.append(
            {
                "sample_id": sample_id,
                "input_json": str(input_json),
                "output_dir": str(sample_output),
                "status": status,
                "error": error,
                "runtime_seconds": round(runtime, 3),
            }
        )

        save_json(
            summary_path,
            {
                "input_dir": str(input_dir),
                "output_root": str(output_root),
                "translation_only": args.translation_only,
                "num_selected": len(input_files),
                "num_complete": sum(
                    item["status"] in {"complete", "skipped_complete"}
                    for item in results
                ),
                "num_failed": sum(item["status"] == "failed" for item in results),
                "results": results,
            },
        )

    print("\n" + "=" * 90)
    print("PROPOSED DATASET RUN FINISHED")
    print("Summary:", summary_path)
    print("=" * 90)


if __name__ == "__main__":
    main()

