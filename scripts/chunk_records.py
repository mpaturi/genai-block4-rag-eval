"""Splits each patient record's `text` field into <=200-char chunks on
sentence boundaries, so long records still fit comfortably within a single
embedding while short records stay as one chunk. See docs/spec.md's
Chunking design section for the full reasoning.
"""
import re

import orjson

# A sentence boundary is a period followed by whitespace. Decimal numbers
# like "27.4" never have whitespace right after the internal period, so
# this rule never mis-splits inside a number.
SENTENCE_BOUNDARY = re.compile(r"(?<=\.)\s+")

# Pinecone's integrated-inference embedding rejects an empty string outright
# (confirmed against the real index in Phase 3), so an empty patient record
# gets this placeholder instead of "" - still produces exactly one chunk,
# and it's short/generic enough to never score as a strong retrieval match.
EMPTY_TEXT_PLACEHOLDER = "No data available."


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Pack sentences into chunks of at most max_chars each."""
    # Defensive case: an empty patient record still produces one chunk,
    # so every patient has >=1 chunk (see spec's Chunking design)
    if text == "":
        return [EMPTY_TEXT_PLACEHOLDER]

    sentences = SENTENCE_BOUNDARY.split(text)

    chunks = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        if not current_sentences:
            # Start a new chunk with this sentence. If the sentence alone
            # is already longer than max_chars, it becomes its own
            # oversized chunk rather than being split or dropped.
            current_sentences = [sentence]
            current_length = len(sentence)
            continue

        # +1 accounts for the single space that joins sentences back together
        length_if_added = current_length + 1 + len(sentence)
        if length_if_added <= max_chars:
            current_sentences.append(sentence)
            current_length = length_if_added
        else:
            # This sentence doesn't fit - close the current chunk and
            # start a new one with it
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_length = len(sentence)

    # Close the final chunk
    chunks.append(" ".join(current_sentences))

    return chunks


def chunk_record(record: dict, max_chars: int = 200) -> list[dict]:
    """Turn one graph_export.jsonl record into one or more chunk dicts."""
    # Validate the shape of external input before using it, so a malformed
    # line fails clearly here instead of crashing deep inside ingest.py
    if "text" not in record:
        raise ValueError(f"record missing 'text' field: {record}")
    if "metadata" not in record:
        raise ValueError(f"record missing 'metadata' field: {record}")

    metadata = record["metadata"]
    if "person_id" not in metadata:
        raise ValueError(f"record metadata missing 'person_id': {record}")

    person_id = metadata["person_id"]
    pieces = chunk_text(record["text"], max_chars=max_chars)

    chunks = []
    for chunk_index, piece in enumerate(pieces):
        chunk = {
            "_id": f"{person_id}_chunk{chunk_index}",
            "chunk_text": piece,
            **metadata,
            "chunk_index": chunk_index,
        }
        chunks.append(chunk)

    return chunks


def chunk_all_records(path: str) -> list[dict]:
    """Read every record in a graph_export.jsonl file and chunk it."""
    all_chunks = []
    with open(path, "rb") as f:
        for line in f:
            record = orjson.loads(line)
            all_chunks.extend(chunk_record(record))
    return all_chunks
