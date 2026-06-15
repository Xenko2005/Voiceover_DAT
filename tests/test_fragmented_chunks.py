import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR))

from semantic_chunking.chunk_models import ASRChunk
from semantic_chunking.chunk_processor import SemanticChunker


def build_test_chunks():

    return [

        # Example 1
        ASRChunk(
            text="Although the model is small,",
            start_time=0.0,
            end_time=1.2,
            silence_after=0.1,
        ),
        ASRChunk(
            text="it performs very well",
            start_time=1.2,
            end_time=2.5,
            silence_after=0.1,
        ),
        ASRChunk(
            text="on real-time translation tasks.",
            start_time=2.5,
            end_time=4.0,
            silence_after=0.8,
        ),

        # Example 2
        ASRChunk(
            text="When the speaker pauses,",
            start_time=5.0,
            end_time=6.0,
            silence_after=0.1,
        ),
        ASRChunk(
            text="the system should wait",
            start_time=6.0,
            end_time=7.2,
            silence_after=0.1,
        ),
        ASRChunk(
            text="for more context.",
            start_time=7.2,
            end_time=8.0,
            silence_after=0.8,
        ),

        # Example 3
        ASRChunk(
            text="Because translation quality",
            start_time=9.0,
            end_time=10.0,
            silence_after=0.1,
        ),
        ASRChunk(
            text="depends heavily",
            start_time=10.0,
            end_time=11.0,
            silence_after=0.1,
        ),
        ASRChunk(
            text="on sentence context.",
            start_time=11.0,
            end_time=12.5,
            silence_after=0.8,
        ),

        # Example 4
        ASRChunk(
            text="If we translate",
            start_time=13.0,
            end_time=14.0,
            silence_after=0.1,
        ),
        ASRChunk(
            text="every chunk immediately,",
            start_time=14.0,
            end_time=15.2,
            silence_after=0.1,
        ),
        ASRChunk(
            text="the output may be incorrect.",
            start_time=15.2,
            end_time=17.0,
            silence_after=0.8,
        ),
    ]


def print_release(result):

    print("\n" + "=" * 80)
    print("RELEASED CLUSTER")
    print("=" * 80)

    print(result["text"])
    print()

    print(f"Start Time : {result['start_time']}")
    print(f"End Time   : {result['end_time']}")
    print(f"Duration   : {result['duration']}")
    print(f"Forced     : {result['forced']}")
    print(f"Score      : {result['semantic_score']}")
    print(f"Chunk Count: {result['chunk_count']}")


def main():

    chunker = SemanticChunker()

    chunks = build_test_chunks()

    release_count = 0

    print("\n")
    print("=" * 80)
    print("FRAGMENTED CHUNK TEST")
    print("=" * 80)

    for idx, chunk in enumerate(chunks, start=1):

        print("\n" + "-" * 80)
        print(f"INPUT CHUNK {idx}")
        print("-" * 80)

        print(chunk.text)

        result = chunker.process_chunk(chunk)

        if result is None:

            print("Decision: HOLD")

        else:

            release_count += 1

            print("Decision: RELEASE")

            print_release(result)

    # flush remaining buffer

    if not chunker.buffer.is_empty():

        result = chunker._release_buffer(force=True)

        release_count += 1

        print_release(result)

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Input Chunks      : {len(chunks)}")
    print(f"Released Clusters : {release_count}")

    if release_count > 0:
        print(
            f"Compression Ratio : "
            f"{len(chunks) / release_count:.2f}"
        )


if __name__ == "__main__":
    main()