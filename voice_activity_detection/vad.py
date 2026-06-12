from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from silero_vad import get_speech_timestamps
from whisper.audio import load_audio as whisper_load_audio


@dataclass
class VADConfig:
    sampling_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 100


def load_audio(audio_path: Path, sampling_rate: int = 16000) -> torch.Tensor:
    if audio_path.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError(f"Only WAV and MP3 input are supported. Received: {audio_path}")
    waveform = whisper_load_audio(str(audio_path), sr=sampling_rate)
    return torch.from_numpy(waveform)


def detect_speech_chunks(
    audio: torch.Tensor, vad_model, config: VADConfig
) -> List[Dict[str, float]]:
    raw_chunks = get_speech_timestamps(
        audio,
        model=vad_model,
        threshold=config.threshold,
        sampling_rate=config.sampling_rate,
        min_speech_duration_ms=config.min_speech_duration_ms,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
        return_seconds=True,
    )

    normalized_chunks: List[Dict[str, float]] = []
    for chunk in raw_chunks:
        start = round(float(chunk["start"]), 3)
        stop = round(float(chunk["end"]), 3)
        if stop <= start:
            continue
        normalized_chunks.append(
            {
                "start_time": start,
                "stop_time": stop,
                "duration": round(stop - start, 3),
            }
        )

    return normalized_chunks
