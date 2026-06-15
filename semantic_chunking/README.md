# Module 2 – Semantic Chunk Buffering

## Overview

This module is part of the **English–Vietnamese Voice-over System** pipeline.

```text
Audio Input
    ↓
Module 1: ASR + VAD
    ↓
Module 2: Semantic Chunk Buffering
    ↓
Module 3: Translation
    ↓
Module 4: TTS
```

The purpose of this module is to group fragmented ASR outputs into semantically complete units before translation.

Without buffering, translation may be performed on incomplete sentence fragments, resulting in reduced translation quality and loss of context.

---

# Objectives

Given streaming ASR chunks:

```python
ASRChunk(
    text="Although the model is small,",
    start_time=0.0,
    end_time=1.2,
    confidence=0.95,
    silence_after=0.1,
)
```

The module decides whether to:

* HOLD the current buffer
* RELEASE the buffer
* FORCE_RELEASE the buffer

The output is a semantic cluster suitable for translation.

---

# Project Structure

```text
semantic_chunking/
│
├── __init__.py
├── chunk_models.py
├── buffer_manager.py
├── semantic_checker.py
├── decision_engine.py
└── chunk_processor.py
```

---

# Core Components

## ASRChunk

Represents a single ASR segment.

```python
ASRChunk(
    text: str,
    start_time: float,
    end_time: float,
    confidence: float,
    silence_after: float,
)
```

Example:

```python
ASRChunk(
    text="it performs very well",
    start_time=1.2,
    end_time=2.5,
    confidence=1.0,
    silence_after=0.1,
)
```

---

## BufferManager

Responsible for storing incoming ASR chunks.

Functions:

```python
append()
clear()
get_chunks()
get_buffered_text()
get_total_duration()
first_timestamp()
last_timestamp()
```

---

## SemanticChecker

Rule-based semantic completeness checker.

Current signals:

### Terminal punctuation

```text
.
?
!
```

### Connector detection

```text
although
because
while
when
if
since
unless
before
after
...
```

### Verb detection

```text
is
are
was
were
have
do
perform
work
translate
...
```

Future improvements:

* SpaCy dependency parsing
* Transformer classifier
* LLM-based semantic validation

---

## DecisionEngine

Responsible for deciding:

```python
HOLD
RELEASE
FORCE_RELEASE
```

### Signals used

* Semantic completeness
* Terminal punctuation
* Silence duration
* Buffer duration
* Buffer size

### Decision logic

```python
if duration >= force_duration_threshold:
    FORCE_RELEASE

elif semantic_complete and (
        punctuation_release
        or silence_release
):
    RELEASE

else:
    HOLD
```

---

# SemanticChunker

Main orchestration class.

Main APIs:

```python
process_chunk()
process_stream()
flush()
```

Example:

```python
chunker = SemanticChunker()

cluster = chunker.process_chunk(chunk)

if cluster:
    print(cluster["text"])
```

Flush remaining buffer:

```python
last_cluster = chunker.flush()

if last_cluster:
    print(last_cluster["text"])
```

---

# Input Format

Input from Module 1:

```python
ASRChunk(
    text,
    start_time,
    end_time,
    confidence,
    silence_after,
)
```

Example:

```python
ASRChunk(
    text="Although the model is small,",
    start_time=0.0,
    end_time=1.2,
    confidence=0.95,
    silence_after=0.1,
)
```

---

# Output Format

Semantic cluster returned to Module 3.

```python
{
    "text": "...",
    "start_time": 0.0,
    "end_time": 4.0,
    "duration": 4.0,
    "forced": False,
    "semantic_score": 1.0,
    "chunk_count": 3
}
```

Module 3 primarily uses:

```python
cluster["text"]
```

for translation.

---

# Example

Input chunks:

```text
Although the model is small,
it performs very well
on real-time translation tasks.
```

Processing:

```text
Chunk 1 -> HOLD
Chunk 2 -> HOLD
Chunk 3 -> RELEASE
```

Output:

```text
Although the model is small, it performs very well on real-time translation tasks.
```

Output metadata:

```python
{
    "start_time": 0.0,
    "end_time": 4.0,
    "duration": 4.0,
    "chunk_count": 3,
}
```

---

# Testing

## Unit Test

```bash
python tests/test_chunker.py
```

Purpose:

* Verify HOLD / RELEASE logic
* Verify semantic grouping

---

## Fragmented Chunk Test

```bash
python tests/test_fragmented_chunks.py
```

Example result:

```text
Input Chunks      : 12
Released Clusters : 4
Compression Ratio : 3.00
```

This demonstrates successful grouping of fragmented ASR outputs into larger semantic units.

---

## TED Talk Integration Test

```bash
python -m tests.test_ted_pipeline
```

Pipeline:

```text
TED Talk JSON
      ↓
ASRChunk
      ↓
SemanticChunker
      ↓
Semantic Clusters
```

Purpose:

* Verify compatibility with Module 1 output
* Verify real-world pipeline integration

---

# Integration with Module 3

Expected usage:

```python
chunker = SemanticChunker()

for asr_chunk in asr_stream:

    cluster = chunker.process_chunk(asr_chunk)

    if cluster:
        translated = translator.translate(
            cluster["text"]
        )

last_cluster = chunker.flush()

if last_cluster:
    translated = translator.translate(
        last_cluster["text"]
    )
```

---

# Current Status

| Component                    | Status   |
| ---------------------------- | -------- |
| ASRChunk                     | Complete |
| BufferManager                | Complete |
| SemanticChecker              | Complete |
| DecisionEngine               | Complete |
| SemanticChunker              | Complete |
| Unit Testing                 | Complete |
| Fragmented Chunk Testing     | Complete |
| Module 1 Integration Testing | Complete |

Module 2 is ready for integration with the Translation Module.
