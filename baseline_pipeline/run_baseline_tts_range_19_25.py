import os
import re
import json
import argparse
import asyncio
import subprocess
import unicodedata
from pathlib import Path
from typing import List, Dict

import edge_tts


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


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


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
        "ffmpeg", "-y",
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
        "ffmpeg", "-y",
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
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:a", "pcm_s16le",
        output_wav,
    ]
    run_cmd(cmd)


def convert_wav_to_mp3(input_wav: str, output_mp3: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        output_mp3,
    ]
    run_cmd(cmd)


def count_vietnamese_units(text: str) -> int:
    clean = re.sub(r"[^\w\sÀ-ỹ]", "", str(text or ""), flags=re.UNICODE)
    units = [u for u in clean.split() if u.strip()]
    return len(units)


def duration_error_ratio(tts_duration: float, source_duration: float) -> float:
    if source_duration <= 0:
        return 0.0
    return abs(tts_duration - source_duration) / source_duration


async def synthesize_edge_tts(text: str, output_mp3: str, voice: str = "vi-VN-HoaiMyNeural", rate: str = "+0%"):
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(output_mp3)


def sanitize_tts_text(text: str) -> str:
    text = str(text or "").strip()
    if text.lower().startswith("[error]"):
        return ""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "...", "–": "-", "—": "-",
        "\n": " ", "\r": " ", "\t": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹ\s\.\,\?\!\:\;\-\'\"\(\)]", "", text, flags=re.UNICODE)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_for_tts(text: str, max_chars: int = 160) -> List[str]:
    text = sanitize_tts_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    parts = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                parts.append(current)
            if len(sent) > max_chars:
                words = sent.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) + 1 <= max_chars:
                        temp = (temp + " " + word).strip()
                    else:
                        if temp:
                            parts.append(temp)
                        temp = word
                current = temp
            else:
                current = sent
    if current:
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


async def synthesize_edge_tts_with_retry(
    text: str,
    output_mp3: str,
    voice: str = "vi-VN-HoaiMyNeural",
    rate: str = "+0%",
    timeout: int = 45,
    retries: int = 3,
) -> bool:
    text = sanitize_tts_text(text)
    if not text:
        return False
    for attempt in range(1, retries + 1):
        try:
            print(f"[TTS] Attempt {attempt}/{retries}")
            await asyncio.wait_for(
                synthesize_edge_tts(text=text, output_mp3=output_mp3, voice=voice, rate=rate),
                timeout=timeout,
            )
            if os.path.exists(output_mp3) and os.path.getsize(output_mp3) > 0:
                return True
            print("[WARN] edge-tts tạo file rỗng.")
        except asyncio.TimeoutError:
            print(f"[WARN] edge-tts timeout sau {timeout}s.")
        except Exception as e:
            print("[WARN] edge-tts lỗi:", e)
        try:
            if os.path.exists(output_mp3):
                os.remove(output_mp3)
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


async def synthesize_text_to_wav_robust(
    text: str,
    output_wav: str,
    temp_dir: str,
    voice: str,
    rate: str,
    sample_rate: int,
    timeout: int = 45,
    retries: int = 3,
    max_chars: int = 160,
) -> bool:
    os.makedirs(temp_dir, exist_ok=True)
    parts = split_text_for_tts(text, max_chars=max_chars)
    if not parts:
        return False
    part_wavs = []
    for i, part in enumerate(parts, start=1):
        part_mp3 = os.path.join(temp_dir, f"part_{i:03d}.mp3")
        part_wav = os.path.join(temp_dir, f"part_{i:03d}.wav")
        print(f"[TTS PART] {i}/{len(parts)}: {part[:100]}")
        ok = await synthesize_edge_tts_with_retry(
            text=part,
            output_mp3=part_mp3,
            voice=voice,
            rate=rate,
            timeout=timeout,
            retries=retries,
        )
        if not ok:
            print("[WARN] Một part TTS thất bại.")
            return False
        convert_to_wav(input_path=part_mp3, output_path=part_wav, sample_rate=sample_rate)
        part_wavs.append(part_wav)
    if len(part_wavs) == 1:
        convert_to_wav(input_path=part_wavs[0], output_path=output_wav, sample_rate=sample_rate)
    else:
        concat_wavs(part_wavs, output_wav)
    return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0


def load_translation_items(translations_path: str) -> List[Dict]:
    data = load_json(translations_path)
    if "results" not in data:
        raise ValueError("baseline_translations.json phải có field 'results'.")
    return data["results"]


def load_existing_results(results_path: str) -> Dict[int, Dict]:
    if not os.path.exists(results_path):
        return {}
    data = load_json(results_path)
    result_map = {}
    for item in data.get("results", []):
        try:
            result_map[int(item["chunk_id"])] = item
        except Exception:
            continue
    return result_map


def build_output_json(input_file: str, model: str, voice: str, results_map: Dict[int, Dict], complete: bool = False) -> Dict:
    results = [results_map[k] for k in sorted(results_map.keys())]
    avg_duration_error = 0.0
    if results:
        avg_duration_error = sum(safe_float(x.get("duration_error_ratio")) for x in results) / len(results)
    avg_vi_units = 0.0
    if results:
        avg_vi_units = sum(safe_float(x.get("vi_units")) for x in results) / len(results)
    return {
        "module": "baseline_stt_translate_tts",
        "status": "complete" if complete else "partial",
        "description": "Baseline: Module 1 STT chunks -> direct translation -> direct TTS, without semantic chunking, context pack, adaptive length constraint, candidate scoring, or duration alignment.",
        "input_file": input_file,
        "model": model,
        "tts_engine": "edge-tts",
        "voice": voice,
        "num_processed_chunks": len(results),
        "summary": {
            "avg_duration_error_ratio": avg_duration_error,
            "avg_vi_units": avg_vi_units,
        },
        "results": results,
    }


async def run_range(args):
    check_ffmpeg()
    output_dir = Path(args.output_dir)
    segment_dir = output_dir / "segments"
    silence_dir = output_dir / "silence"
    ensure_dir(str(output_dir))
    ensure_dir(str(segment_dir))
    ensure_dir(str(silence_dir))

    translations = load_translation_items(args.translations)
    translations = [x for x in translations if args.start_id <= int(x["chunk_id"]) <= args.end_id]
    if not translations:
        raise ValueError(f"Không tìm thấy chunk {args.start_id} đến {args.end_id} trong {args.translations}.")

    results_map = load_existing_results(args.results)

    previous_end_time = 0.0
    for cid in sorted(results_map.keys()):
        if cid < args.start_id:
            previous_end_time = max(previous_end_time, safe_float(results_map[cid].get("end_time")))

    print("=" * 80)
    print(f"TTS RANGE: chunk {args.start_id} -> {args.end_id}")
    print("Translations:", args.translations)
    print("Existing results:", args.results)
    print("Output dir:", args.output_dir)
    print("Previous end time:", previous_end_time)
    print("=" * 80)

    for idx, item in enumerate(translations, start=1):
        chunk_id = int(item["chunk_id"])
        source_text = item.get("source_text", "")
        translation = item.get("baseline_translation", "")
        start_time = safe_float(item.get("start_time"))
        end_time = safe_float(item.get("end_time"), default=start_time)
        source_duration = safe_float(item.get("source_duration"), default=max(0.0, end_time - start_time))

        print("\n" + "-" * 80)
        print(f"TTS CHUNK {chunk_id} ({idx}/{len(translations)})")
        print("SOURCE:", source_text)
        print("VI:", translation)
        print("START:", start_time)
        print("END:", end_time)
        print("SOURCE_DURATION:", source_duration)

        gap = start_time - previous_end_time
        if gap > 0.05:
            silence_path = silence_dir / f"gap_before_{chunk_id:04d}.wav"
            if Path(silence_path).exists() and not args.force_silence:
                print("[SKIP] Silence wav đã tồn tại:", silence_path)
            else:
                print(f"Creating silence gap_before_{chunk_id:04d}: {gap:.3f}s")
                create_silence_wav(output_path=str(silence_path), duration=gap, sample_rate=args.sample_rate)

        raw_mp3 = segment_dir / f"chunk_{chunk_id:04d}_tts.mp3"
        raw_wav = segment_dir / f"chunk_{chunk_id:04d}_tts.wav"
        temp_part_dir = segment_dir / f"chunk_{chunk_id:04d}_parts"

        if translation.startswith("[ERROR]") or not translation.strip():
            print("[WARN] Translation lỗi/rỗng. Tạo silence thay thế.")
            create_silence_wav(output_path=str(raw_wav), duration=source_duration, sample_rate=args.sample_rate)
            tts_duration = get_audio_duration(str(raw_wav))
        else:
            if Path(raw_wav).exists() and not args.force_tts:
                print("[SKIP] TTS wav đã tồn tại:", raw_wav)
                tts_duration = get_audio_duration(str(raw_wav))
            else:
                ok = await synthesize_text_to_wav_robust(
                    text=translation,
                    output_wav=str(raw_wav),
                    temp_dir=str(temp_part_dir),
                    voice=args.voice,
                    rate=args.rate,
                    sample_rate=args.sample_rate,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_chars=args.max_chars,
                )
                if ok:
                    tts_duration = get_audio_duration(str(raw_wav))
                else:
                    print("[WARN] TTS thất bại. Tạo silence thay thế.")
                    create_silence_wav(output_path=str(raw_wav), duration=source_duration, sample_rate=args.sample_rate)
                    tts_duration = get_audio_duration(str(raw_wav))

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
        results_map[chunk_id] = result_item
        previous_end_time = max(previous_end_time, end_time)

        print("TTS DURATION:", round(tts_duration, 3))
        print("DURATION ERROR:", round(dur_error, 3))

        partial_output = build_output_json(
            input_file=args.input_file,
            model=args.model,
            voice=args.voice,
            results_map=results_map,
            complete=False,
        )
        save_json(args.results, partial_output)
        print("[SAVED]", args.results)

    output = build_output_json(
        input_file=args.input_file,
        model=args.model,
        voice=args.voice,
        results_map=results_map,
        complete=args.mark_complete,
    )
    save_json(args.results, output)

    print("\nDone range.")
    print("Segments folder:", segment_dir)
    print("Silence folder:", silence_dir)
    print("Updated results:", args.results)

    if args.finalize:
        build_final_audio(args, results_map)


def build_final_audio(args, results_map: Dict[int, Dict]):
    output_dir = Path(args.output_dir)
    silence_dir = output_dir / "silence"
    dubbed_wav = output_dir / "baseline_dubbed_audio.wav"
    dubbed_mp3 = output_dir / "baseline_dubbed_audio.mp3"
    ordered = [results_map[k] for k in sorted(results_map.keys())]
    timeline_wavs = []
    previous_end_time = 0.0
    print("\n" + "=" * 80)
    print("FINALIZE: build baseline_dubbed_audio")
    print("=" * 80)
    for item in ordered:
        chunk_id = int(item["chunk_id"])
        start_time = safe_float(item.get("start_time"))
        end_time = safe_float(item.get("end_time"), default=start_time)
        gap = start_time - previous_end_time
        if gap > 0.05:
            silence_path = silence_dir / f"gap_before_{chunk_id:04d}.wav"
            if not silence_path.exists():
                create_silence_wav(output_path=str(silence_path), duration=gap, sample_rate=args.sample_rate)
            timeline_wavs.append(str(silence_path))
        wav = item.get("files", {}).get("tts_wav")
        if wav and Path(wav).exists():
            timeline_wavs.append(wav)
        else:
            print("[WARN] Missing wav for chunk", chunk_id, wav)
        previous_end_time = max(previous_end_time, end_time)
    concat_wavs(timeline_wavs, str(dubbed_wav))
    convert_wav_to_mp3(str(dubbed_wav), str(dubbed_mp3))
    data = load_json(args.results)
    data["status"] = "complete"
    data["dubbed_wav"] = str(dubbed_wav)
    data["dubbed_mp3"] = str(dubbed_mp3)
    save_json(args.results, data)
    print("Dubbed WAV:", dubbed_wav)
    print("Dubbed MP3:", dubbed_mp3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translations", type=str, default="baseline_pipeline/outputs/baseline_translations.json")
    parser.add_argument("--results", type=str, default="baseline_pipeline/outputs/baseline_results.json")
    parser.add_argument("--output-dir", type=str, default="baseline_pipeline/outputs")
    parser.add_argument("--input-file", type=str, default="output_files\\ted_talk.json")
    parser.add_argument("--start-id", type=int, default=19)
    parser.add_argument("--end-id", type=int, default=25)
    parser.add_argument("--model", type=str, default="qwen3.5:2b")
    parser.add_argument("--voice", type=str, default="vi-VN-HoaiMyNeural")
    parser.add_argument("--rate", type=str, default="+0%")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=160)
    parser.add_argument("--force-tts", action="store_true")
    parser.add_argument("--force-silence", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--mark-complete", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_range(args))


if __name__ == "__main__":
    main()
