import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from silero_vad import load_silero_vad

from speech_to_text import TranscriberConfig, WhisperTranscriber
from voice_activity_detection import VADConfig, detect_speech_chunks, load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 1 pipeline: VAD with timestamps + Whisper transcription"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="WAV/MP3 file path or a directory containing WAV/MP3 files",
    )
    parser.add_argument(
        "--stream-dir",
        type=Path,
        default=Path("output_stream"),
        help="Folder for streaming chunk outputs (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_files"),
        help="Folder for per-audio final JSON outputs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Whisper model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, for example "cuda" or "cpu"',
    )
    return parser.parse_args()


SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


def discover_audio_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Only WAV and MP3 files are supported.")
        return [input_path]

    if input_path.is_dir():
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend([p for p in input_path.glob(f"*{ext}") if p.is_file()])
        return sorted(files)

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def slice_audio_chunk(audio, sampling_rate: int, start_time: float, stop_time: float) -> np.ndarray:
    start_idx = max(0, int(start_time * sampling_rate))
    stop_idx = max(start_idx, int(stop_time * sampling_rate))
    return audio[start_idx:stop_idx].cpu().numpy()


def process_audio_file(
    audio_path: Path,
    vad_model,
    vad_config: VADConfig,
    transcriber: WhisperTranscriber,
    stream_dir: Path,
    output_dir: Path,
) -> dict:
    audio = load_audio(audio_path, sampling_rate=vad_config.sampling_rate)
    chunks = detect_speech_chunks(audio, vad_model=vad_model, config=vad_config)

    stream_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_path = stream_dir / f"{audio_path.stem}.stream.jsonl"
    output_path = output_dir / f"{audio_path.stem}.json"

    chunk_results = []
    with stream_path.open("w", encoding="utf-8") as stream_file:
        for chunk in chunks:
            chunk_waveform = slice_audio_chunk(
                audio,
                sampling_rate=vad_config.sampling_rate,
                start_time=chunk["start_time"],
                stop_time=chunk["stop_time"],
            )
            text = transcriber.transcribe_waveform_chunk(chunk_waveform)

            result_chunk = {
                "transcribed_text": text,
                "start_time": chunk["start_time"],
                "stop_time": chunk["stop_time"],
                "duration": chunk["duration"],
            }
            chunk_results.append(result_chunk)
            stream_file.write(json.dumps(result_chunk, ensure_ascii=False) + "\n")
            stream_file.flush()

    final_payload = {
        "source_file": str(audio_path.name),
        "chunks": chunk_results,
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(final_payload, output_file, ensure_ascii=False, indent=2)

    return {
        "source_file": str(audio_path),
        "stream_output": str(stream_path),
        "final_output": str(output_path),
        "chunk_count": len(chunk_results),
    }


def main() -> None:
    args = parse_args()

    audio_files = discover_audio_files(args.input_path)
    if not audio_files:
        raise ValueError("No WAV/MP3 files found in the provided input path.")

    vad_model = load_silero_vad()
    vad_config = VADConfig()

    transcriber_config = TranscriberConfig(
        model_name=args.model,
        language="en",
        device=args.device,
    )
    transcriber = WhisperTranscriber(transcriber_config)

    summaries = []
    for wav_file in audio_files:
        summary = process_audio_file(
            audio_path=wav_file,
            vad_model=vad_model,
            vad_config=vad_config,
            transcriber=transcriber,
            stream_dir=args.stream_dir,
            output_dir=args.output_dir,
        )
        summaries.append(summary)

    print(json.dumps({"processed": summaries}, indent=2))


if __name__ == "__main__":
    main()
