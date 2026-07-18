import os
import re
import json
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

import requests
import edge_tts


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =====================================================
# CMD helpers
# =====================================================

def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print("\n[CMD ERROR]")
        print("Command:", " ".join(cmd))
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError("Command failed.")

    return result


def check_ffmpeg():
    run_cmd(["ffmpeg", "-version"])
    run_cmd(["ffprobe", "-version"])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# Audio helpers
# =====================================================

def get_audio_duration(audio_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]

    result = run_cmd(cmd)

    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    run_cmd(cmd)


def create_silence_wav(output_path: str, duration: float, sample_rate: int = 24000):
    if duration <= 0:
        duration = 0.01

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", f"{duration:.3f}",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    run_cmd(cmd)


def concat_wavs(wav_files: List[str], output_wav: str):
    concat_list_path = output_wav.replace(".wav", "_concat_list.txt")

    with open(concat_list_path, "w", encoding="utf-8") as f:
        for wav in wav_files:
            wav_path = Path(wav).resolve().as_posix()
            f.write(f"file '{wav_path}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:a", "pcm_s16le",
        output_wav,
    ]
    run_cmd(cmd)


def convert_wav_to_mp3(input_wav: str, output_mp3: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_wav,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        output_mp3,
    ]
    run_cmd(cmd)


# =====================================================
# Load Module 1 chunks
# =====================================================

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def extract_text(item: Dict) -> str:
    return (
        item.get("transcript")
        or item.get("transcribed_text")
        or item.get("source_text")
        or item.get("text")
        or ""
    )


def extract_start_time(item: Dict) -> float:
    return safe_float(
        item.get("start_time", item.get("start", 0.0)),
        default=0.0,
    )


def extract_end_time(item: Dict, start_time: float) -> float:
    return safe_float(
        item.get("end_time", item.get("stop_time", item.get("end", start_time))),
        default=start_time,
    )


def load_module1_chunks(input_path: str) -> List[Dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "chunks" in data:
            data = data["chunks"]
        elif "segments" in data:
            data = data["segments"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("Input Module 1 phải là list hoặc dict chứa field chunks.")

    chunks = []

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        text = extract_text(item).strip()
        start_time = extract_start_time(item)
        end_time = extract_end_time(item, start_time)

        if not text:
            continue

        duration = safe_float(
            item.get("duration", end_time - start_time),
            default=max(0.0, end_time - start_time),
        )

        chunks.append(
            {
                "chunk_id": i,
                "source_text": text,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
            }
        )

    return chunks


# =====================================================
# Ollama direct translation baseline
# =====================================================

def call_ollama(prompt: str, model: str = "qwen3.5:2b") -> str:
    url = "http://localhost:11434/api/generate"

    safe_prompt = "/no_think\n" + prompt

    payload = {
        "model": model,
        "prompt": safe_prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 128,
            "num_ctx": 1024,
        },
    }

    response = requests.post(url, json=payload, timeout=240)

    if response.status_code != 200:
        print("\n[OLLAMA API ERROR]")
        print("Status:", response.status_code)
        print("Response:", response.text)
        response.raise_for_status()

    data = response.json()
    return str(data.get("response", "")).strip()


def clean_translation_output(text: str) -> str:
    text = str(text or "").strip()

    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) > 0:
        text = lines[0]

    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text)
    text = re.sub(
        r"^\s*(Vietnamese|Translation|Bản dịch)\s*[:：]\s*",
        "",
        text,
        flags=re.I,
    )

    return text.strip().strip('"').strip("'")


def build_baseline_translation_prompt(source_text: str) -> str:
    return f"""
Bạn là hệ thống dịch Anh sang Việt.

Hãy dịch câu/đoạn tiếng Anh sau sang tiếng Việt tự nhiên.

Yêu cầu:
- Chỉ trả về một bản dịch tiếng Việt.
- Không giải thích.
- Không ghi chú.
- Không dùng context toàn bài.
- Không cần tối ưu độ dài.

English:
{source_text}

Vietnamese:
""".strip()


def translate_direct(source_text: str, model: str) -> str:
    prompt = build_baseline_translation_prompt(source_text)

    try:
        raw = call_ollama(prompt, model=model)
        cleaned = clean_translation_output(raw)

        if not cleaned:
            return "[ERROR] Empty translation."

        return cleaned

    except Exception as e:
        print("[WARN] Translation error:", e)
        return "[ERROR] Translation failed."


# =====================================================
# Translation phase
# =====================================================

def load_existing_translations(translation_file: str) -> Dict[int, Dict]:
    if not os.path.exists(translation_file):
        return {}

    try:
        data = load_json(translation_file)
        result_map = {}

        for item in data.get("results", []):
            chunk_id = int(item.get("chunk_id"))
            result_map[chunk_id] = item

        return result_map

    except Exception:
        return {}


def run_translation_phase(args, chunks: List[Dict], output_dir: Path) -> List[Dict]:
    translation_file = output_dir / "baseline_translations.json"

    existing_map = {}

    if not args.force_translate:
        existing_map = load_existing_translations(str(translation_file))

    translation_results = []

    print("\n" + "=" * 80)
    print("PHASE 1: TRANSLATE ALL CHUNKS FIRST")
    print("=" * 80)

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = int(chunk["chunk_id"])

        if chunk_id in existing_map:
            old_item = existing_map[chunk_id]
            translation = old_item.get("baseline_translation", "")

            if translation and not translation.startswith("[ERROR]"):
                translation_results.append(old_item)
                print(f"[SKIP] Chunk {chunk_id} đã có bản dịch.")
                continue

        print("\n" + "-" * 80)
        print(f"TRANSLATING CHUNK {chunk_id} ({idx}/{len(chunks)})")
        print("SOURCE:", chunk["source_text"])

        translation = translate_direct(
            source_text=chunk["source_text"],
            model=args.model,
        )

        item = {
            "chunk_id": chunk_id,
            "source_text": chunk["source_text"],
            "baseline_translation": translation,
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "source_duration": chunk["duration"],
        }

        translation_results.append(item)

        print("VI:", translation)

        partial_output = {
            "module": "baseline_translation_phase",
            "status": "partial",
            "description": "Baseline translation only: Module 1 chunks -> direct LLM translation.",
            "input_file": args.input,
            "model": args.model,
            "num_translated_chunks": len(translation_results),
            "results": translation_results,
        }

        save_json(str(translation_file), partial_output)

    final_output = {
        "module": "baseline_translation_phase",
        "status": "complete",
        "description": "Baseline translation only: Module 1 chunks -> direct LLM translation.",
        "input_file": args.input,
        "model": args.model,
        "num_translated_chunks": len(translation_results),
        "results": translation_results,
    }

    save_json(str(translation_file), final_output)

    print("\nTranslation phase done.")
    print("Saved:", translation_file)

    return translation_results


# =====================================================
# TTS baseline
# =====================================================

async def synthesize_edge_tts(
    text: str,
    output_mp3: str,
    voice: str = "vi-VN-HoaiMyNeural",
    rate: str = "+0%",
):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
    )
    await communicate.save(output_mp3)


def count_vietnamese_units(text: str) -> int:
    clean = re.sub(r"[^\w\sÀ-ỹ]", "", str(text or ""), flags=re.UNICODE)
    units = [u for u in clean.split() if u.strip()]
    return len(units)


def duration_error_ratio(tts_duration: float, source_duration: float) -> float:
    if source_duration <= 0:
        return 0.0

    return abs(tts_duration - source_duration) / source_duration


async def run_tts_phase(args, translation_results: List[Dict], output_dir: Path) -> List[Dict]:
    segment_dir = output_dir / "segments"
    silence_dir = output_dir / "silence"

    ensure_dir(str(segment_dir))
    ensure_dir(str(silence_dir))

    print("\n" + "=" * 80)
    print("PHASE 2: TTS ALL TRANSLATIONS")
    print("=" * 80)

    final_results = []
    timeline_wavs = []
    previous_end_time = 0.0

    for idx, item in enumerate(translation_results, start=1):
        chunk_id = int(item["chunk_id"])
        source_text = item.get("source_text", "")
        translation = item.get("baseline_translation", "")

        start_time = safe_float(item.get("start_time"))
        end_time = safe_float(item.get("end_time"), default=start_time)
        source_duration = safe_float(item.get("source_duration"), default=end_time - start_time)

        print("\n" + "-" * 80)
        print(f"TTS CHUNK {chunk_id} ({idx}/{len(translation_results)})")
        print("VI:", translation)

        gap = start_time - previous_end_time

        if gap > 0.05:
            silence_path = silence_dir / f"gap_before_{chunk_id:04d}.wav"
            create_silence_wav(
                output_path=str(silence_path),
                duration=gap,
                sample_rate=args.sample_rate,
            )
            timeline_wavs.append(str(silence_path))

        raw_mp3 = segment_dir / f"chunk_{chunk_id:04d}_tts.mp3"
        raw_wav = segment_dir / f"chunk_{chunk_id:04d}_tts.wav"

        if translation.startswith("[ERROR]") or not translation.strip():
            print("[WARN] Translation failed. Tạo silence thay thế.")
            create_silence_wav(
                output_path=str(raw_wav),
                duration=source_duration,
                sample_rate=args.sample_rate,
            )
            tts_duration = get_audio_duration(str(raw_wav))

        else:
            # Nếu file wav đã tồn tại và không ép TTS lại, bỏ qua để chạy nhanh hơn.
            if raw_wav.exists() and not args.force_tts:
                print("[SKIP] TTS wav đã tồn tại.")
                tts_duration = get_audio_duration(str(raw_wav))
            else:
                await synthesize_edge_tts(
                    text=translation,
                    output_mp3=str(raw_mp3),
                    voice=args.voice,
                    rate=args.rate,
                )

                convert_to_wav(
                    input_path=str(raw_mp3),
                    output_path=str(raw_wav),
                    sample_rate=args.sample_rate,
                )

                tts_duration = get_audio_duration(str(raw_wav))

        timeline_wavs.append(str(raw_wav))
        previous_end_time = max(previous_end_time, end_time)

        vi_units = count_vietnamese_units(translation)
        dur_error = duration_error_ratio(tts_duration, source_duration)

        result_item = {
            "chunk_id": chunk_id,
            "source_text": source_text,
            "baseline_translation": translation,
            "start_time": start_time,
            "end_time": end_time,
            "source_duration": source_duration,
            "vi_units": vi_units,
            "tts_duration": tts_duration,
            "duration_error_ratio": dur_error,
            "files": {
                "tts_mp3": str(raw_mp3),
                "tts_wav": str(raw_wav),
            },
        }

        final_results.append(result_item)

        print("TTS DURATION:", round(tts_duration, 3))
        print("DURATION ERROR:", round(dur_error, 3))

        partial_output = {
            "module": "baseline_stt_translate_tts",
            "status": "partial",
            "description": "Baseline: Module 1 STT chunks -> direct translation -> direct TTS, without semantic chunking, context pack, adaptive length constraint, candidate scoring, or duration alignment.",
            "input_file": args.input,
            "model": args.model,
            "tts_engine": "edge-tts",
            "voice": args.voice,
            "num_processed_chunks": len(final_results),
            "results": final_results,
        }

        save_json(str(output_dir / "baseline_results.json"), partial_output)

    dubbed_wav = output_dir / "baseline_dubbed_audio.wav"
    dubbed_mp3 = output_dir / "baseline_dubbed_audio.mp3"

    print("\n" + "=" * 80)
    print("PHASE 3: CONCAT BASELINE AUDIO")
    print("=" * 80)

    concat_wavs(timeline_wavs, str(dubbed_wav))
    convert_wav_to_mp3(str(dubbed_wav), str(dubbed_mp3))

    avg_duration_error = 0.0
    if final_results:
        avg_duration_error = sum(x["duration_error_ratio"] for x in final_results) / len(final_results)

    avg_vi_units = 0.0
    if final_results:
        avg_vi_units = sum(x["vi_units"] for x in final_results) / len(final_results)

    final_output = {
        "module": "baseline_stt_translate_tts",
        "status": "complete",
        "description": "Baseline: Module 1 STT chunks -> direct translation -> direct TTS, without semantic chunking, context pack, adaptive length constraint, candidate scoring, or duration alignment.",
        "input_file": args.input,
        "model": args.model,
        "tts_engine": "edge-tts",
        "voice": args.voice,
        "dubbed_wav": str(dubbed_wav),
        "dubbed_mp3": str(dubbed_mp3),
        "num_chunks": len(final_results),
        "summary": {
            "avg_duration_error_ratio": avg_duration_error,
            "avg_vi_units": avg_vi_units,
        },
        "results": final_results,
    }

    save_json(str(output_dir / "baseline_results.json"), final_output)

    print("\nBASELINE DONE")
    print("Translations JSON:", output_dir / "baseline_translations.json")
    print("Output JSON:", output_dir / "baseline_results.json")
    print("Dubbed WAV:", dubbed_wav)
    print("Dubbed MP3:", dubbed_mp3)
    print("Avg duration error:", round(avg_duration_error, 4))

    return final_results


# =====================================================
# Main baseline pipeline
# =====================================================

async def run_baseline(args):
    check_ffmpeg()

    chunks = load_module1_chunks(args.input)

    if args.max_chunks is not None:
        chunks = chunks[:args.max_chunks]

    output_dir = Path(args.output_dir)
    ensure_dir(str(output_dir))

    print("=" * 80)
    print("BASELINE PIPELINE")
    print("Flow: Module 1 chunks -> translate all -> TTS all -> concat")
    print("Input:", args.input)
    print("Model:", args.model)
    print("Voice:", args.voice)
    print("Chunks:", len(chunks))
    print("=" * 80)

    translation_results = run_translation_phase(
        args=args,
        chunks=chunks,
        output_dir=output_dir,
    )

    if args.translation_only:
        print("\nĐã chạy xong translation-only. Dừng trước TTS.")
        return

    await run_tts_phase(
        args=args,
        translation_results=translation_results,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="output_files/ted_talk.json",
        help="Output JSON của Module 1.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_pipeline/outputs",
        help="Thư mục lưu baseline output.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:2b",
        help="Model Ollama dùng để dịch baseline.",
    )

    parser.add_argument(
        "--voice",
        type=str,
        default="vi-VN-HoaiMyNeural",
        help="Voice edge-tts tiếng Việt.",
    )

    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
        help="Tốc độ edge-tts.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate WAV output.",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Giới hạn số chunk để test nhanh.",
    )

    parser.add_argument(
        "--translation-only",
        action="store_true",
        help="Chỉ dịch và lưu baseline_translations.json, không chạy TTS.",
    )

    parser.add_argument(
        "--force-translate",
        action="store_true",
        help="Ép dịch lại từ đầu, không dùng baseline_translations.json cũ.",
    )

    parser.add_argument(
        "--force-tts",
        action="store_true",
        help="Ép tạo TTS lại dù file wav đã tồn tại.",
    )

    args = parser.parse_args()

    asyncio.run(run_baseline(args))


if __name__ == "__main__":
    main()
