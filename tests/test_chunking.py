"""Unit tests for the chunking logic in chunk_records.py.

TDD: written before chunk_records.py exists - these tests define the
chunking contract chunk_records.py must satisfy. All should fail with an
ImportError until chunk_records.py and chunk_text() are implemented.
"""
from scripts.chunk_records import EMPTY_TEXT_PLACEHOLDER, chunk_text


def test_short_text_stays_as_single_chunk():
    text = "Patient 248, born in the 1960s, Female. Visits: 5."
    assert len(text) <= 200
    result = chunk_text(text, max_chars=200)
    assert result == [text]


def test_empty_text_produces_one_placeholder_chunk():
    # Pinecone rejects an empty chunk_text string (confirmed against the
    # real index), so this substitutes a minimal, non-empty placeholder.
    result = chunk_text("", max_chars=200)
    assert result == [EMPTY_TEXT_PLACEHOLDER]


def test_text_splits_on_sentence_boundaries_packing_multiple_per_chunk():
    # Three 90-char sentences. The first two (181 chars) fit together in
    # one chunk; the third doesn't fit alongside them, so it starts a new
    # chunk. This also proves sentences are packed together up to the
    # limit, not split one-sentence-per-chunk.
    s1 = "A" * 89 + "."
    s2 = "B" * 89 + "."
    s3 = "C" * 89 + "."
    text = f"{s1} {s2} {s3}"

    result = chunk_text(text, max_chars=200)

    assert result == [f"{s1} {s2}", s3]
    assert all(len(chunk) <= 200 for chunk in result)


def test_oversized_single_sentence_becomes_its_own_chunk():
    # A single "sentence" (no internal period) longer than max_chars.
    # Must not crash or truncate - becomes its own oversized chunk.
    oversized_sentence = "D" * 250 + "."
    result = chunk_text(oversized_sentence, max_chars=200)

    assert result == [oversized_sentence]
    assert len(result[0]) > 200
