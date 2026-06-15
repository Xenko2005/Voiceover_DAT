import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json

from semantic_chunking.chunk_models import ASRChunk
from semantic_chunking.chunk_processor import SemanticChunker


def load_chunks(json_path: str):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    asr_chunks = []

    chunks = data["chunks"]

    for chunk in chunks:

        silence_after = 0.3

        asr_chunk = ASRChunk(
            text=chunk["transcribed_text"],
            start_time=chunk["start_time"],
            end_time=chunk["stop_time"],
            confidence=1.0,
            silence_after=silence_after,
        )

        asr_chunks.append(asr_chunk)

    return asr_chunks


def print_cluster(cluster: dict):

    print("\n" + "=" * 80)
    print("SEMANTIC CLUSTER")
    print("=" * 80)

    print(cluster["text"])
    print()

    print(f"Start Time : {cluster['start_time']}")
    print(f"End Time   : {cluster['end_time']}")
    print(f"Duration   : {cluster['duration']}")
    print(f"Forced     : {cluster['forced']}")
    print(f"Score      : {cluster['semantic_score']}")
    print(f"Chunk Count: {cluster['chunk_count']}")


def main():

    json_path = Path("output_files/ted_talk.json")

    if not json_path.exists():
        raise FileNotFoundError(
            f"Cannot find {json_path}"
        )

    print("\nLoading TED Talk JSON...")
    print(json_path)

    asr_chunks = load_chunks(json_path)

    print(f"\nTotal ASR Chunks: {len(asr_chunks)}")

    chunker = SemanticChunker()

    total_clusters = 0
    total_duration = 0.0
    total_chunk_count = 0
    force_release_count = 0

    print("\nStarting Semantic Chunking...\n")

    for idx, chunk in enumerate(asr_chunks, start=1):

        print("-" * 80)
        print(f"INPUT CHUNK {idx}")
        print("-" * 80)

        print(chunk.text)

        result = chunker.process_chunk(chunk)

        if result:

            total_clusters += 1
            total_duration += result["duration"]
            total_chunk_count += result["chunk_count"]

            if result["forced"]:
                force_release_count += 1

            print_cluster(result)

    # Flush remaining buffer
    if not chunker.buffer.is_empty():

        print("\n" + "=" * 80)
        print("FINAL FLUSH")
        print("=" * 80)

        result = chunker._release_buffer(force=True)

        total_clusters += 1
        total_duration += result["duration"]
        total_chunk_count += result["chunk_count"]

        if result["forced"]:
            force_release_count += 1

        print_cluster(result)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Total ASR Chunks     : {len(asr_chunks)}")
    print(f"Semantic Clusters    : {total_clusters}")
    print(f"Force Releases       : {force_release_count}")

    if total_clusters > 0:

        avg_duration = (
            total_duration /
            total_clusters
        )

        avg_cluster_size = (
            total_chunk_count /
            total_clusters
        )

        print(
            f"Average Duration     : "
            f"{avg_duration:.2f}s"
        )

        print(
            f"Average Cluster Size : "
            f"{avg_cluster_size:.2f} chunks"
        )


if __name__ == "__main__":
    main()