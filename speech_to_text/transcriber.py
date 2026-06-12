from dataclasses import dataclass
from typing import Optional

import numpy as np
import whisper


@dataclass
class TranscriberConfig:
    model_name: str = "large-v3"
    language: str = "en"
    device: Optional[str] = None


class WhisperTranscriber:
    def __init__(self, config: TranscriberConfig):
        self.config = config
        self.model = whisper.load_model(config.model_name, device=config.device)

    def transcribe_waveform_chunk(self, chunk_waveform: np.ndarray) -> str:
        if chunk_waveform.size == 0:
            return ""

        use_fp16 = self.model.device.type == "cuda"
        result = self.model.transcribe(
            chunk_waveform,
            language=self.config.language,
            task="transcribe",
            fp16=use_fp16,
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
        )
        return str(result.get("text", "")).strip()
