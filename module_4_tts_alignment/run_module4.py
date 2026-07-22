import os
import json
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

import edge_tts


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =====================================================
# Command helpers
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
    try:
        run_cmd(["ffmpeg", "-version"])
        run_cmd(["ffprobe", "-version"])
    except Exception:
        raise RuntimeError(
            "Không tìm thấy ffmpeg/ffprobe. Hãy cài ffmpeg trước rồi mở CMD lại."
        )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =====================================================
# Audio helpers
# =====================================================

def get_audio_duration(audio_path: str) -> float:
    """
    Đo duration audio bằng ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]

    result = run_cmd(cmd)
    value = result.stdout.strip()

    try:
        return float(value)
    except Exception:
        return 0.0


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000):
    """
    Convert mp3/wav bất kỳ về wav mono.
    """
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


MAX_TTS_SPEED = 1.15
SPEED_TOLERANCE_SECONDS = 0.05


def build_atempo_filter(speed_factor: float) -> str:
    """
    ffmpeg atempo chỉ hỗ trợ mỗi filter trong khoảng 0.5 đến 2.0.
    Nếu speed_factor lớn hơn 2 hoặc nhỏ hơn 0.5 thì phải chain nhiều atempo.

    speed_factor > 1: nói nhanh hơn
    speed_factor < 1: nói chậm hơn
    """
    if speed_factor <= 0:
        speed_factor = 1.0

    factors = []

    while speed_factor > 2.0:
        factors.append(2.0)
        speed_factor /= 2.0

    while speed_factor < 0.5:
        factors.append(0.5)
        speed_factor /= 0.5

    factors.append(speed_factor)

    return ",".join([f"atempo={factor:.6f}" for factor in factors])


def adjust_speed_to_target_duration(
    input_wav: str,
    output_wav: str,
    target_duration: float,
    sample_rate: int = 24000,
) -> Dict:
    """
    Giữ giọng TTS tự nhiên bằng cách chỉ tăng tốc trong một giới hạn nhỏ.
    Phần thời lượng còn thiếu sẽ được bù bằng silence, phần dư sẽ bị trim.
    """
    raw_duration = get_audio_duration(input_wav)

    if target_duration <= 0:
        target_duration = raw_duration

    if raw_duration <= 0:
        ideal_speed_factor = 1.0
        speed_factor = 1.0
    else:
        ideal_speed_factor = raw_duration / max(target_duration, 1e-6)
        if ideal_speed_factor < 1.0:
            speed_factor = MAX_TTS_SPEED
        else:
            speed_factor = min(ideal_speed_factor, MAX_TTS_SPEED)

    temp_speed_wav = output_wav.replace(".wav", "_speed.wav")
    temp_pad_wav = output_wav.replace(".wav", "_pad.wav")

    atempo_filter = build_atempo_filter(speed_factor)

    cmd_speed = [
        "ffmpeg",
        "-y",
        "-i", input_wav,
        "-filter:a", atempo_filter,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        temp_speed_wav,
    ]
    run_cmd(cmd_speed)

    adjusted_duration = raw_duration / max(speed_factor, 1e-6) if raw_duration > 0 else target_duration
    pad_duration = max(0.0, target_duration - adjusted_duration)
    trim_duration = max(0.0, adjusted_duration - target_duration)

    if pad_duration > SPEED_TOLERANCE_SECONDS:
        create_silence_wav(temp_pad_wav, pad_duration, sample_rate=sample_rate)
        concat_wavs([temp_speed_wav, temp_pad_wav], output_wav)
    elif trim_duration > SPEED_TOLERANCE_SECONDS:
        cmd_trim = [
            "ffmpeg",
            "-y",
            "-i", temp_speed_wav,
            "-t", f"{target_duration:.3f}",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            output_wav,
        ]
        run_cmd(cmd_trim)
    else:
        cmd_exact = [
            "ffmpeg",
            "-y",
            "-i", temp_speed_wav,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            output_wav,
        ]
        run_cmd(cmd_exact)

    final_duration = get_audio_duration(output_wav)

    try:
        os.remove(temp_speed_wav)
    except Exception:
        pass

    try:
        os.remove(temp_pad_wav)
    except Exception:
        pass

    return {
        "raw_duration": raw_duration,
        "target_duration": target_duration,
        "ideal_speed_factor": ideal_speed_factor,
        "speed_factor": speed_factor,
        "adjusted_duration": adjusted_duration,
        "pad_duration": pad_duration,
        "trim_duration": trim_duration,
        "final_duration": final_duration,
        "duration_error_abs": abs(final_duration - target_duration),
        "duration_error_ratio": abs(final_duration - target_duration) / max(target_duration, 1e-6),
    }


def create_silence_wav(output_path: str, duration: float, sample_rate: int = 24000):
    """
    Tạo audio silence để giữ khoảng nghỉ giữa các segment.
    """
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
    """
    Ghép nhiều wav bằng ffmpeg concat demuxer.
    """
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
# TTS helpers
# =====================================================

async def synthesize_edge_tts(
    text: str,
    output_mp3: str,
    voice: str = "vi-VN-HoaiMyNeural",
    rate: str = "+0%",
):
    """
    Tạo TTS tiếng Việt bằng edge-tts.
    """
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
    )
    await communicate.save(output_mp3)


def clean_translation(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    # Bỏ lỗi fallback nếu có.
    if text.lower().startswith("[error]"):
        return ""

    return text


# =====================================================
# Load Module 3
# =====================================================

def load_module3_results(input_path: str) -> Dict:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data:
        raise ValueError("File input không có field 'results'. Hãy dùng output của Module 3.")

    return data


def compute_gap_after_segments(results: List[Dict]) -> List[Optional[float]]:
    gaps: List[Optional[float]] = []

    for index, result in enumerate(results):
        if index >= len(results) - 1:
            gaps.append(None)
            continue

        current_start = float(result.get("start_time", 0.0))
        current_duration = float(result.get("duration", 0.0))
        current_end = float(result.get("end_time", current_start + max(0.0, current_duration)))
        next_start = float(results[index + 1].get("start_time", 0.0))
        gaps.append(max(0.0, next_start - current_end))

    return gaps


# =====================================================
# Feedback for Module 3
# =====================================================

def build_feedback_item(result: Dict, tts_info: Dict, gap_after_segment: Optional[float]) -> Dict:
    """
    Feedback giúp Module 3 biết câu nào nên dịch ngắn/dài hơn.
    """
    source_duration = result.get("duration", 0.0)
    raw_tts_duration = tts_info.get("raw_duration", 0.0)
    speed_factor = tts_info.get("speed_factor", 1.0)

    if source_duration <= 0:
        suggestion = "unknown"
        reason = "source_duration không hợp lệ"
    elif raw_tts_duration > source_duration * 1.25:
        suggestion = "make_translation_shorter"
        reason = "TTS gốc dài hơn source duration quá nhiều, phải tăng tốc đáng kể."
    elif raw_tts_duration < source_duration * 0.75:
        suggestion = "allow_translation_longer"
        reason = "TTS gốc ngắn hơn source duration khá nhiều, có thể dịch tự nhiên hơn/dài hơn."
    else:
        suggestion = "ok"
        reason = "TTS duration tương đối phù hợp với source duration."

    return {
        "segment_id": result.get("segment_id"),
        "source_text": result.get("source_text"),
        "best_translation": result.get("best_translation"),
        "source_duration": source_duration,
        "gap_after_segment": gap_after_segment,
        "raw_tts_duration": raw_tts_duration,
        "speed_factor": speed_factor,
        "suggestion": suggestion,
        "reason": reason,
    }


# =====================================================
# Main pipeline
# =====================================================

async def run_module4(args):
    check_ffmpeg()

    data = load_module3_results(args.input)
    results = data["results"]

    if args.max_segments is not None:
        results = results[:args.max_segments]

    gap_after_segments = compute_gap_after_segments(results)

    output_dir = Path(args.output_dir)
    segments_dir = output_dir / "segments"
    silence_dir = output_dir / "silence"

    ensure_dir(str(output_dir))
    ensure_dir(str(segments_dir))
    ensure_dir(str(silence_dir))

    module4_results = []
    feedback_items = []

    timeline_wavs = []
    previous_end_time = 0.0

    print(f"Tổng số segment sẽ xử lý TTS: {len(results)}")
    print(f"Voice: {args.voice}")
    print(f"Rate: {args.rate}")
    print("=" * 80)

    for idx, result in enumerate(results, start=1):
        segment_id = int(result.get("segment_id", idx))
        gap_after_segment = gap_after_segments[idx - 1] if idx - 1 < len(gap_after_segments) else None

        source_text = result.get("source_text", "")
        translation = clean_translation(result.get("best_translation", ""))

        start_time = float(result.get("start_time", 0.0))
        end_time = float(result.get("end_time", start_time))
        source_duration = float(result.get("duration", max(0.0, end_time - start_time)))

        print("\n" + "-" * 80)
        print(f"SEGMENT {segment_id}")
        print("SOURCE:", source_text)
        print("VI:", translation)
        print("START:", start_time)
        print("END:", end_time)
        print("SOURCE DURATION:", source_duration)
        print("GAP AFTER SEGMENT:", gap_after_segment if gap_after_segment is not None else "N/A")

        # Thêm silence nếu có gap giữa 2 segment.
        gap = start_time - previous_end_time
        if gap > 0.05:
            silence_path = silence_dir / f"gap_before_{segment_id:04d}.wav"
            create_silence_wav(
                output_path=str(silence_path),
                duration=gap,
                sample_rate=args.sample_rate,
            )
            timeline_wavs.append(str(silence_path))
            print(f"Added silence gap: {gap:.3f}s")

        raw_mp3 = segments_dir / f"segment_{segment_id:04d}_raw.mp3"
        raw_wav = segments_dir / f"segment_{segment_id:04d}_raw.wav"
        final_wav = segments_dir / f"segment_{segment_id:04d}_final.wav"

        if not translation:
            print("[WARN] Translation rỗng hoặc là [ERROR]. Tạo silence thay thế.")
            create_silence_wav(
                output_path=str(final_wav),
                duration=source_duration,
                sample_rate=args.sample_rate,
            )

            tts_info = {
                "raw_duration": 0.0,
                "target_duration": source_duration,
                "ideal_speed_factor": 1.0,
                "speed_factor": 1.0,
                "adjusted_duration": 0.0,
                "pad_duration": source_duration,
                "trim_duration": 0.0,
                "final_duration": get_audio_duration(str(final_wav)),
                "duration_error_abs": 0.0,
                "duration_error_ratio": 0.0,
            }

        else:
            # 1. TTS ra mp3
            await synthesize_edge_tts(
                text=translation,
                output_mp3=str(raw_mp3),
                voice=args.voice,
                rate=args.rate,
            )

            # 2. Convert mp3 sang wav
            convert_to_wav(
                input_path=str(raw_mp3),
                output_path=str(raw_wav),
                sample_rate=args.sample_rate,
            )

            # 3. Speed adjust theo source duration
            tts_info = adjust_speed_to_target_duration(
                input_wav=str(raw_wav),
                output_wav=str(final_wav),
                target_duration=source_duration,
                sample_rate=args.sample_rate,
            )

        timeline_wavs.append(str(final_wav))
        previous_end_time = max(previous_end_time, end_time)

        feedback_item = build_feedback_item(result, tts_info, gap_after_segment)
        feedback_items.append(feedback_item)

        item = {
            "segment_id": segment_id,
            "source_text": source_text,
            "best_translation": translation,
            "start_time": start_time,
            "end_time": end_time,
            "source_duration": source_duration,
            "gap_after_segment": gap_after_segment,
            "raw_tts_duration": tts_info["raw_duration"],
            "target_duration": tts_info["target_duration"],
            "ideal_speed_factor": tts_info.get("ideal_speed_factor"),
            "speed_factor": tts_info["speed_factor"],
            "adjusted_duration": tts_info.get("adjusted_duration"),
            "pad_duration": tts_info.get("pad_duration"),
            "trim_duration": tts_info.get("trim_duration"),
            "final_tts_duration": tts_info["final_duration"],
            "duration_error_abs": tts_info["duration_error_abs"],
            "duration_error_ratio": tts_info["duration_error_ratio"],
            "feedback_suggestion": feedback_item["suggestion"],
            "files": {
                "raw_mp3": str(raw_mp3),
                "raw_wav": str(raw_wav),
                "final_wav": str(final_wav),
            },
        }

        module4_results.append(item)

        print("RAW TTS DURATION:", round(tts_info["raw_duration"], 3))
        print("IDEAL SPEED FACTOR:", round(tts_info.get("ideal_speed_factor", 1.0), 3))
        print("SPEED FACTOR:", round(tts_info["speed_factor"], 3))
        print("PAD DURATION:", round(tts_info.get("pad_duration", 0.0), 3))
        print("TRIM DURATION:", round(tts_info.get("trim_duration", 0.0), 3))
        print("FINAL TTS DURATION:", round(tts_info["final_duration"], 3))
        print("FEEDBACK:", feedback_item["suggestion"])

        # Lưu tạm sau mỗi segment để tránh mất kết quả nếu lỗi giữa chừng.
        save_outputs(
            output_dir=output_dir,
            input_file=args.input,
            voice=args.voice,
            rate=args.rate,
            module4_results=module4_results,
            feedback_items=feedback_items,
            partial=True,
        )

    dubbed_wav = output_dir / "dubbed_audio.wav"
    dubbed_mp3 = output_dir / "dubbed_audio.mp3"

    print("\n" + "=" * 80)
    print("Đang ghép toàn bộ audio...")
    concat_wavs(timeline_wavs, str(dubbed_wav))
    convert_wav_to_mp3(str(dubbed_wav), str(dubbed_mp3))

    save_outputs(
        output_dir=output_dir,
        input_file=args.input,
        voice=args.voice,
        rate=args.rate,
        module4_results=module4_results,
        feedback_items=feedback_items,
        partial=False,
        dubbed_wav=str(dubbed_wav),
        dubbed_mp3=str(dubbed_mp3),
    )

    print("\nHoàn tất Module 4.")
    print("Dubbed WAV:", dubbed_wav)
    print("Dubbed MP3:", dubbed_mp3)
    print("Kết quả JSON:", output_dir / "module4_tts_results.json")
    print("Feedback JSON:", output_dir / "feedback_for_module3.json")


def save_outputs(
    output_dir: Path,
    input_file: str,
    voice: str,
    rate: str,
    module4_results: List[Dict],
    feedback_items: List[Dict],
    partial: bool = False,
    dubbed_wav: Optional[str] = None,
    dubbed_mp3: Optional[str] = None,
):
    module4_output = {
        "module": "module_4_tts_alignment",
        "status": "partial" if partial else "complete",
        "input_file": input_file,
        "tts_engine": "edge-tts",
        "voice": voice,
        "rate": rate,
        "dubbed_wav": dubbed_wav,
        "dubbed_mp3": dubbed_mp3,
        "num_processed_segments": len(module4_results),
        "results": module4_results,
    }

    output_path = output_dir / "module4_tts_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(module4_output, f, ensure_ascii=False, indent=2)

    feedback_output = {
        "module": "module_4_feedback_for_module_3",
        "description": "Feedback duration từ TTS thật để Module 3 biết segment nào cần dịch ngắn/dài hơn.",
        "items": feedback_items,
    }

    feedback_path = output_dir / "feedback_for_module3.json"

    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback_output, f, ensure_ascii=False, indent=2)


# =====================================================
# CLI
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="module_3_adaptive_translation/outputs/module3_translation_results.json",
        help="Đường dẫn output JSON của Module 3.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="module_4_tts_alignment/outputs",
        help="Thư mục output của Module 4.",
    )

    parser.add_argument(
        "--voice",
        type=str,
        default="vi-VN-HoaiMyNeural",
        help="Voice tiếng Việt edge-tts. Ví dụ: vi-VN-HoaiMyNeural hoặc vi-VN-NamMinhNeural.",
    )

    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
        help="Tốc độ TTS ban đầu của edge-tts. Ví dụ: +0%, +10%, -10%.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate output wav.",
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Giới hạn số segment để test nhanh.",
    )

    args = parser.parse_args()

    asyncio.run(run_module4(args))


if __name__ == "__main__":
    main()
