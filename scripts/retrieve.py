"""Job 1 - searches the Pinecone index for chunks similar to a question.
See docs/spec.md's Retrieval design section: Pinecone embeds the question
server-side (same model as ingestion) and returns ranked chunks with
similarity scores; a chunk counts as relevant only if its score clears the
threshold.
"""
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Must match ingest.py's NAMESPACE - both are the same hardcoded constant
# per docs/spec.md's Pinecone index design (not a .env variable)
NAMESPACE = "patients"

# Direction (higher-is-better) confirmed against the real index in Phase
# 3's self-match check. The value itself was revised in Phase 5, from an
# initial 0.75 down to 0.4: real natural-language question scores (~0.4-0.55
# for genuinely relevant matches) never cleared 0.75, which made every
# question fall back regardless of relevance - see docs/eval_results.md's
# Run 1 vs Run 2 and docs/spec.md's Retrieval design section.
DEFAULT_THRESHOLD = 0.4

# Phase 7's Run 6 (docs/eval_results.md) measured that dropping the
# threshold to 0.2 while a condition/drug metadata filter is active
# raises recall from 0.287 to 0.884 with fallback accuracy holding at a
# perfect 1.000 - safe specifically because a condition/drug filter can
# structurally return zero candidates for an untracked topic (e.g.
# condition="Asthma" matches nobody, full stop). The same 0.2 threshold
# with no filter collapses fallback accuracy to 0.000 (Run 6's unfiltered
# control) - see select_threshold() below for where this is applied.
PERMISSIVE_THRESHOLD = 0.2

# Revision history: initial 5 -> 15 (Run 5) -> 20 (this fixup, Run 7).
# Run 5 (top_k=25, threshold=0.4, unfiltered-style flat threshold) found
# the highest per-question `retrieved` count actually observed was 11
# (q01, q02), so 15 was picked as a ~36% margin above that. But Run 7
# (top_k=20, threshold=0.2 - the permissive threshold condition/drug
# filters now get automatically via select_threshold(), live in api.py
# since PR7's review fixup) showed top_k=15 still truncated 3 of 12
# tested questions (q01, q06, q08), and top_k=20 closed nearly all of
# that gap (recall 0.884 -> 0.977, Run 6 -> Run 7). Since a real caller
# (Block 5) typically sends condition/drug-filtered queries - exactly the
# regime where the permissive threshold is now the common case, not the
# exception - 20 is the better-supported default. See docs/eval_results.md's
# Run 5, Run 7, and Run 9.
DEFAULT_TOP_K = 20

# Ceiling for filtered (condition/drug) queries only - mirrors how
# select_threshold() gates PERMISSIVE_THRESHOLD on the same condition.
# Confirmed empirically against the real Pinecone client (not assumed):
# a condition/drug-filtered top_k up to 25 still returns in the same way
# unfiltered calls do. Deliberately not set higher - a much bigger top_k
# turns a similarity search into a table scan, the wrong tool for a
# fully-structured query. That job belongs to the graph; see Block 5's
# docs/spec.md "what I'd do next" for the planned follow-up.
FILTERED_TOP_K_CEILING = 25

# Confirmed against the installed pinecone client (9.1.0): Index.search()
# takes a `timeout` kwarg directly (seconds, float) - not the older
# query()-style retry/urllib3 timeout convention. A timed-out call raises
# PineconeTimeoutError, confirmed live by calling search() with an
# unreasonably small timeout - it's a TimeoutError/OSError/Exception
# subclass, so api.py's existing `except Exception` in the /query handler
# already converts it into the standard 502, no new handling needed there.
SEARCH_TIMEOUT_SECONDS = 10


def meets_threshold(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """A chunk scoring exactly at the threshold still counts as a match."""
    return score >= threshold


def select_threshold(*, condition: str | None = None, drug: str | None = None) -> float:
    """Picks the score threshold for a query. Returns PERMISSIVE_THRESHOLD
    only when a condition or drug filter is present - those are the only
    filter types that can structurally return zero patients for an
    untracked topic (e.g. condition="Asthma" matches nobody, full stop),
    which is what makes a lower threshold safe. gender/lab/birth_decade
    filters provide no such protection - an off-topic question paired
    with e.g. a gender-only filter would still return a pile of unrelated
    patients, so it must not lower the bar for what counts as a match.
    See Run 6 in docs/eval_results.md for the measured evidence.
    """
    if condition is not None or drug is not None:
        return PERMISSIVE_THRESHOLD
    return DEFAULT_THRESHOLD


# Naming matches Block 5's graph_tool.py exactly, so both tools present
# identical filter shorthand to a caller/agent. The operators differ by
# necessity: Block 5 emits Cypher >/<, this emits Pinecone filter
# operators - both strict inequalities, so "above"/"below" mean the same
# thing (never inclusive) in both tools.
_LAB_PROPERTY = {
    "SBP": "latest_sbp",
    "BMI": "latest_bmi",
    "Glucose": "latest_glucose",
    "HbA1c": "latest_hba1c",
}
_COMPARISON_OP = {"above": "$gt", "below": "$lt"}

# Pinecone's stored metadata uses these full forms, not the eval/API
# shorthand - gender and birth_decade must be translated before building
# a filter, never passed through unmapped (see docs/spec.md's Metadata
# filter design).
_GENDER_VALUE = {"M": "Male", "F": "Female"}


class RAGFilterError(ValueError):
    """Raised when metadata filter arguments are invalid - an unrecognized
    lab/comparison/gender, or a partial lab/comparison/value combination.
    Always raised before any Pinecone call, matching Block 5's
    graph_tool.py fail-fast principle for the same kind of fixed
    whitelist lookup.
    """


def build_metadata_filter(
    condition: str | None = None,
    drug: str | None = None,
    lab: str | None = None,
    comparison: str | None = None,
    value: float | None = None,
    gender: str | None = None,
    birth_decade: int | None = None,
) -> dict | None:
    """Builds a Pinecone metadata filter dict from structured arguments.

    Returns None if nothing was passed, so callers doing pure semantic
    search see no behavior change. lab, comparison, and value must be
    either all present or all absent - a partial combination raises
    rather than silently dropping the incomplete filter.
    """
    lab_args_present = [lab is not None, comparison is not None, value is not None]
    if any(lab_args_present) and not all(lab_args_present):
        raise RAGFilterError(
            "lab, comparison, and value must be given together - "
            f"got lab={lab!r}, comparison={comparison!r}, value={value!r}"
        )

    filter_dict: dict = {}

    if condition is not None:
        filter_dict["conditions"] = condition

    if drug is not None:
        filter_dict["drugs"] = drug

    if all(lab_args_present):
        lab_property = _LAB_PROPERTY.get(lab)
        if lab_property is None:
            raise RAGFilterError(f"unrecognized lab: {lab!r}")
        op = _COMPARISON_OP.get(comparison)
        if op is None:
            raise RAGFilterError(f"unrecognized comparison: {comparison!r}")
        filter_dict[lab_property] = {op: value}

    if gender is not None:
        gender_value = _GENDER_VALUE.get(gender)
        if gender_value is None:
            raise RAGFilterError(f"unrecognized gender: {gender!r}")
        filter_dict["gender"] = gender_value

    if birth_decade is not None:
        filter_dict["year_of_birth_band"] = f"{birth_decade}s"

    return filter_dict or None


_index = None


def _get_index():
    """Lazily builds the Pinecone client and index handle on first call,
    then reuses the same instance on every call after - avoids
    reconnecting on every retrieve() call. Env-var validation happens
    here, once, as part of this one-time setup, rather than on every
    retrieve() call.
    """
    global _index
    if _index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set in .env")
        if not index_name:
            raise RuntimeError("PINECONE_INDEX_NAME not set in .env")

        pc = Pinecone(api_key=api_key)
        _index = pc.Index(name=index_name)
    return _index


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    condition: str | None = None,
    drug: str | None = None,
    lab: str | None = None,
    comparison: str | None = None,
    value: float | None = None,
    gender: str | None = None,
    birth_decade: int | None = None,
) -> list[dict]:
    """Search Pinecone for the top_k chunks most similar to question,
    optionally narrowed by structured metadata filters (Phase 7).

    Returns a list of dicts, each holding the chunk's id, similarity
    score, and full metadata (chunk_text, person_id, etc.) - ranked
    highest score first, same order Pinecone returns.
    """
    # Raises before any Pinecone call if the filter arguments are invalid
    metadata_filter = build_metadata_filter(
        condition=condition,
        drug=drug,
        lab=lab,
        comparison=comparison,
        value=value,
        gender=gender,
        birth_decade=birth_decade,
    )

    index = _get_index()

    search_kwargs = {
        "namespace": NAMESPACE,
        "top_k": top_k,
        "inputs": {"text": question},
        "timeout": SEARCH_TIMEOUT_SECONDS,
    }
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    result = index.search(**search_kwargs)

    chunks = []
    for hit in result.result.hits:
        chunk = {"chunk_id": hit.id, "score": hit.score, **hit.fields}
        # Pinecone stores all numeric metadata as float, but person_id is an
        # identifier, not a quantity - cast back to int here, once, so every
        # caller (api.py's sources list, generate.py's citation text) gets a
        # clean 248 instead of 248.0.
        chunk["person_id"] = int(chunk["person_id"])
        chunks.append(chunk)

    return chunks


def main() -> int:
    """Run a sample query by hand and print each chunk's score and
    threshold decision, so results can be eyeballed for sanity."""
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Which patients have type 2 diabetes and take metformin?"
    )

    print(f"Question: {question}\n")

    try:
        chunks = retrieve(question)
    except RuntimeError as e:
        print(f"FAIL - {e}")
        return 1

    if not chunks:
        print("No chunks returned.")
        return 0

    for chunk in chunks:
        match = meets_threshold(chunk["score"])
        status = "MATCH" if match else "below threshold"
        print(f"{chunk['chunk_id']}  score={chunk['score']:.4f}  ({status})")
        print(f"  {chunk['chunk_text']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
