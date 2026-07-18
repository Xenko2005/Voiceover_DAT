from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "sample"


def natural_key(text: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.rstrip("\n").strip() for line in file]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def parse_simple_phost_yaml(path: Path) -> list[dict[str, Any]]:
    """
    PhoST YAML files are simple inline dictionaries, for example:
    - {duration: 2.89, offset: 0.96, rW: 7, uW: 0, wav: 39095.wav}
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


def discover_sample_ids(repo_id: str, split: str) -> list[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    pattern = re.compile(rf"^text_data/{re.escape(split)}/([^/]+)/\1\.yaml$")
    sample_ids = []
    for filename in files:
        match = pattern.match(filename)
        if match:
            sample_ids.append(match.group(1))
    return sorted(set(sample_ids), key=natural_key)


def download_one_sample(
    repo_id: str,
    split: str,
    sample_id: str,
    sample_dir: Path,
    sample_cache_dir: Path,
    download_audio: bool,
) -> dict[str, Path | None]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path | None] = {}
    for suffix in ["en", "vi", "yaml"]:
        remote = f"text_data/{split}/{sample_id}/{sample_id}.{suffix}"
        cached = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote,
            cache_dir=str(sample_cache_dir),
        )
        target = sample_dir / f"{sample_id}.{suffix}"
        shutil.copy2(cached, target)
        downloaded[suffix] = target

    audio_path: Path | None = None
    if download_audio:
        remote_audio = f"audio_data/{split}/wav/{sample_id}.wav"
        cached_audio = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_audio,
            cache_dir=str(sample_cache_dir),
        )
        audio_path = sample_dir / f"{sample_id}.wav"
        shutil.copy2(cached_audio, audio_path)

    downloaded["audio"] = audio_path
    return downloaded


def build_module1_json(
    sample_id: str,
    split: str,
    en_path: Path,
    vi_path: Path,
    yaml_path: Path,
    audio_path: Path | None,
    output_path: Path,
    max_segments_per_sample: int | None,
) -> dict[str, Any]:
    english_lines = read_lines(en_path)
    vietnamese_lines = read_lines(vi_path)
    yaml_items = parse_simple_phost_yaml(yaml_path)

    n = min(len(english_lines), len(vietnamese_lines), len(yaml_items))
    if max_segments_per_sample is not None:
        n = min(n, max_segments_per_sample)
    if n <= 0:
        raise RuntimeError(f"Sample {sample_id} has no usable segments.")

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
        "sample_id": safe_id(f"{split}/{sample_id}"),
        "split": split,
        "source_language": "English",
        "target_language": "Vietnamese",
        "source_file": f"{sample_id}.wav",
        "audio_path": str(audio_path.resolve()) if audio_path else "",
        "audio_available": audio_path is not None,
        "audio_is_temporary": audio_path is not None,
        "source_text": " ".join(english_lines[:n]),
        "reference_vi": " ".join(vietnamese_lines[:n]),
        "num_chunks": len(chunks),
        "duration": round(total_duration, 3),
        "chunks": chunks,
    }

    save_json(output_path, document)
    return document


def run_stream(command: list[str], cwd: Path) -> None:
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


def run_compare(
    project_root: Path,
    baseline_root: Path,
    proposed_root: Path,
    metrics_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(project_root / "experiments" / "compare_dataset_runs.py"),
        "--baseline-root",
        str(baseline_root),
        "--proposed-root",
        str(proposed_root),
        "--output-dir",
        str(metrics_dir),
    ]
    run_stream(command, cwd=project_root)


def run_translation_only_compare(
    project_root: Path,
    baseline_root: Path,
    proposed_root: Path,
    module1_root: Path,
    metrics_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(project_root / "experiments" / "compare_translation_only_runs.py"),
        "--baseline-root",
        str(baseline_root),
        "--proposed-root",
        str(proposed_root),
        "--module1-root",
        str(module1_root),
        "--output-dir",
        str(metrics_dir),
    ]
    run_stream(command, cwd=project_root)


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "sample_id",
        "status",
        "num_chunks",
        "duration",
        "baseline_output",
        "proposed_output",
        "runtime_seconds",
        "error",
    ]
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_done_samples(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            item["sample_id"]
            for item in data.get("results", [])
            if item.get("status") == "complete"
        }
    except Exception:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream PhoST test samples from Hugging Face one-by-one, run "
            "baseline and proposed pipelines, then delete each sample temp cache."
        )
    )
    parser.add_argument("--repo-id", default="vinai/PhoST")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output-root", type=Path, default=Path("experiment_outputs/hf_phost_test_streaming"))
    parser.add_argument("--temp-root", type=Path, default=Path("runtime_hf_samples"))
    parser.add_argument("--cache-root", type=Path, default=Path("runtime_hf_cache"))
    parser.add_argument("--optimizer-config", type=Path, default=Path("module_3_adaptive_translation/outputs/phost_optimizer_config_fulltrain.json"))
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--voice", default="vi-VN-HoaiMyNeural")
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--max-segments-per-sample", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--translation-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-audio-download", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    module1_root = output_root / "module1_json"
    current_input_root = output_root / "_current_module1"
    baseline_root = output_root / "baseline"
    proposed_root = output_root / "proposed"
    metrics_dir = output_root / "metrics"
    temp_root = args.temp_root.resolve()
    cache_root = args.cache_root.resolve()
    summary_path = output_root / "hf_streaming_run_summary.json"
    manifest_path = output_root / "hf_streaming_manifest.csv"

    if args.optimizer_config and not args.optimizer_config.exists():
        raise FileNotFoundError(f"Optimizer config not found: {args.optimizer_config}")

    output_root.mkdir(parents=True, exist_ok=True)
    module1_root.mkdir(parents=True, exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)
    proposed_root.mkdir(parents=True, exist_ok=True)

    if args.sample_ids:
        sample_ids = [str(item) for item in args.sample_ids]
    else:
        print("Discovering PhoST sample ids from Hugging Face...")
        sample_ids = discover_sample_ids(args.repo_id, args.split)

    sample_ids = sample_ids[args.start_index :]
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        raise RuntimeError("No samples selected.")

    completed = set() if args.force else load_done_samples(summary_path)
    results: list[dict[str, Any]] = []

    print("=" * 90)
    print("PHOST HUGGING FACE STREAMING RUN")
    print("Repo        :", args.repo_id)
    print("Split       :", args.split)
    print("Samples     :", len(sample_ids))
    print("Model       :", args.model)
    print("Output      :", output_root)
    print("Temp/cache  :", temp_root, "|", cache_root)
    print("Delete temp :", not args.keep_temp)
    print("Mode        :", "prepare-only" if args.prepare_only else ("translation-only" if args.translation_only else "full TTS"))
    print("=" * 90)

    for index, sample_id in enumerate(sample_ids, start=1):
        safe_sample = safe_id(f"{args.split}_{sample_id}")
        sample_temp = temp_root / safe_sample
        sample_cache = cache_root / safe_sample
        module1_json = module1_root / f"{safe_sample}.json"
        current_module1_json = current_input_root / f"{safe_sample}.json"
        sample_baseline_root = baseline_root / "_current"
        sample_proposed_root = proposed_root / "_current"
        final_baseline_dir = baseline_root / safe_sample
        final_proposed_dir = proposed_root / safe_sample

        if sample_id in completed and not args.force:
            print(f"[{index}/{len(sample_ids)}] {sample_id}: skipped, already complete.")
            continue

        print("\n" + "=" * 90)
        print(f"[{index}/{len(sample_ids)}] Sample {sample_id}")
        print("=" * 90)

        start_time = time.perf_counter()
        status = "failed"
        error = ""
        document: dict[str, Any] = {}

        try:
            if sample_baseline_root.exists():
                shutil.rmtree(sample_baseline_root)
            if sample_proposed_root.exists():
                shutil.rmtree(sample_proposed_root)
            if current_input_root.exists():
                shutil.rmtree(current_input_root)
            sample_baseline_root.mkdir(parents=True, exist_ok=True)
            sample_proposed_root.mkdir(parents=True, exist_ok=True)
            current_input_root.mkdir(parents=True, exist_ok=True)

            paths = download_one_sample(
                repo_id=args.repo_id,
                split=args.split,
                sample_id=sample_id,
                sample_dir=sample_temp,
                sample_cache_dir=sample_cache,
                download_audio=not args.skip_audio_download,
            )

            document = build_module1_json(
                sample_id=sample_id,
                split=args.split,
                en_path=paths["en"],  # type: ignore[arg-type]
                vi_path=paths["vi"],  # type: ignore[arg-type]
                yaml_path=paths["yaml"],  # type: ignore[arg-type]
                audio_path=paths["audio"],  # type: ignore[arg-type]
                output_path=module1_json,
                max_segments_per_sample=args.max_segments_per_sample,
            )
            shutil.copy2(module1_json, current_module1_json)

            if args.prepare_only:
                status = "prepared_only"
            else:
                run_stream(
                    [
                        sys.executable,
                        str(project_root / "experiments" / "run_baseline_dataset.py"),
                        "--input-dir",
                        str(current_input_root),
                        "--output-root",
                        str(sample_baseline_root),
                        "--model",
                        args.model,
                        "--voice",
                        args.voice,
                        "--rate",
                        args.rate,
                        "--sample-rate",
                        str(args.sample_rate),
                        "--limit",
                        "1",
                        "--force",
                        "--stop-on-error",
                        *(["--translation-only"] if args.translation_only else []),
                        *(["--max-chunks", str(args.max_chunks)] if args.max_chunks is not None else []),
                    ],
                    cwd=project_root,
                )

                proposed_command = [
                    sys.executable,
                    str(project_root / "experiments" / "run_proposed_dataset.py"),
                    "--input-dir",
                    str(current_input_root),
                    "--output-root",
                    str(sample_proposed_root),
                    "--model",
                    args.model,
                    "--voice",
                    args.voice,
                    "--rate",
                    args.rate,
                    "--sample-rate",
                    str(args.sample_rate),
                    "--batch-size",
                    str(args.batch_size),
                    "--limit",
                    "1",
                    "--force",
                    "--stop-on-error",
                ]
                if args.optimizer_config:
                    proposed_command.extend(["--optimizer-config", str(args.optimizer_config)])
                if args.translation_only:
                    proposed_command.append("--translation-only")
                if args.max_chunks is not None:
                    proposed_command.extend(["--max-segments", str(args.max_chunks)])

                run_stream(proposed_command, cwd=project_root)

                if final_baseline_dir.exists():
                    shutil.rmtree(final_baseline_dir)
                if final_proposed_dir.exists():
                    shutil.rmtree(final_proposed_dir)

                current_baseline_children = [
                    item for item in sample_baseline_root.iterdir()
                    if item.is_dir()
                ]
                current_proposed_children = [
                    item for item in sample_proposed_root.iterdir()
                    if item.is_dir()
                ]
                if not current_baseline_children or not current_proposed_children:
                    raise RuntimeError("Per-sample runner did not create expected output folders.")

                shutil.move(str(current_baseline_children[0]), str(final_baseline_dir))
                shutil.move(str(current_proposed_children[0]), str(final_proposed_dir))
                status = "complete"

        except Exception as exc:
            error = str(exc)
            status = "failed"
            print("[ERROR]", error)
            if args.stop_on_error:
                raise
        finally:
            if sample_baseline_root.exists():
                shutil.rmtree(sample_baseline_root, ignore_errors=True)
            if sample_proposed_root.exists():
                shutil.rmtree(sample_proposed_root, ignore_errors=True)
            if current_input_root.exists():
                shutil.rmtree(current_input_root, ignore_errors=True)
            if not args.keep_temp:
                if sample_temp.exists():
                    shutil.rmtree(sample_temp, ignore_errors=True)
                if sample_cache.exists():
                    shutil.rmtree(sample_cache, ignore_errors=True)

        runtime = time.perf_counter() - start_time
        result = {
            "sample_id": sample_id,
            "status": status,
            "num_chunks": document.get("num_chunks", 0),
            "duration": document.get("duration", 0.0),
            "baseline_output": str(final_baseline_dir) if final_baseline_dir.exists() else "",
            "proposed_output": str(final_proposed_dir) if final_proposed_dir.exists() else "",
            "runtime_seconds": round(runtime, 3),
            "error": error,
        }
        results.append(result)
        append_manifest_row(manifest_path, result)

        previous_results = []
        if summary_path.exists():
            try:
                previous_results = json.loads(summary_path.read_text(encoding="utf-8")).get("results", [])
            except Exception:
                previous_results = []

        filtered_previous = [
            item for item in previous_results
            if item.get("sample_id") != sample_id
        ]
        save_json(
            summary_path,
            {
                "repo_id": args.repo_id,
                "split": args.split,
                "output_root": str(output_root),
                "translation_only": args.translation_only,
                "prepare_only": args.prepare_only,
                "num_selected": len(sample_ids),
                "results": filtered_previous + results,
            },
        )

    if not args.prepare_only and args.translation_only and not args.skip_compare:
        print("\nAggregating translation-only baseline/proposed metrics...")
        run_translation_only_compare(
            project_root=project_root,
            baseline_root=baseline_root,
            proposed_root=proposed_root,
            module1_root=module1_root,
            metrics_dir=metrics_dir,
        )

    if not args.prepare_only and not args.translation_only and not args.skip_compare:
        print("\nAggregating baseline/proposed metrics...")
        run_compare(
            project_root=project_root,
            baseline_root=baseline_root,
            proposed_root=proposed_root,
            metrics_dir=metrics_dir,
        )

    print("\n" + "=" * 90)
    print("DONE")
    print("Summary :", summary_path)
    print("Manifest:", manifest_path)
    if metrics_dir.exists():
        if args.translation_only:
            print("Metrics :", metrics_dir / "translation_only_metrics_summary.json")
        else:
            print("Metrics :", metrics_dir / "dataset_metrics_summary.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
