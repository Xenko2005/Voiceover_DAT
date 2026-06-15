import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

from semantic_chunking import (
    ASRChunk,
    SemanticChunker,
)

# Hiển thị log ra terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:

    chunker = SemanticChunker(
        silence_threshold=0.5,
        force_duration_threshold=8.0,
        force_chunk_threshold=10,
    )

    chunks = [

        ASRChunk(
            text="Although the model is small,",
            start_time=0.0,
            end_time=1.2,
            confidence=0.91,
            silence_after=0.10,
        ),

        ASRChunk(
            text="it performs very well",
            start_time=1.2,
            end_time=2.5,
            confidence=0.93,
            silence_after=0.10,
        ),

        ASRChunk(
            text="on real-time translation tasks.",
            start_time=2.5,
            end_time=4.0,
            confidence=0.95,
            silence_after=0.80,
        ),
    ]

    print("\n========== INPUT CHUNKS ==========\n")

    for idx, chunk in enumerate(chunks, start=1):
        print(
            f"Chunk {idx}: "
            f"'{chunk.text}' "
            f"({chunk.start_time:.2f}s -> {chunk.end_time:.2f}s)"
        )

    print("\n========== PROCESSING ==========\n")

    for chunk in chunks:

        result = chunker.process_chunk(chunk)

        if result:

            print("\n========== RELEASE ==========\n")

            print(f"Text        : {result['text']}")
            print(f"Start Time  : {result['start_time']}")
            print(f"End Time    : {result['end_time']}")
            print(f"Duration    : {result['duration']}")
            print(f"Forced      : {result['forced']}")
            print(f"Score       : {result['semantic_score']}")
            print(f"Chunk Count : {result['chunk_count']}")

    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()