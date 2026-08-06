"""Verifies the whole pipeline is actually working, not just assumed to be:
credentials valid, index reachable, the Pinecone index holds what
chunk_records.py + ingest.py were supposed to put there, re-ingestion is
truly idempotent, a real query returns results, and the eval harness has
produced a saved score report. See docs/spec.md's functional requirement
#12 and Reproducibility and determinism section.

Polls with a bounded retry loop rather than a single fixed sleep after
each ingestion run, since Pinecone writes are eventually consistent - a
count checked immediately after upsert_records returns can undercount.
"""
import os
import sys
import time

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

from check_connection import (
    check_anthropic,
    check_index_name,
    check_pinecone,
    check_pinecone_query_key,
)
from chunk_records import chunk_all_records
from ingest import NAMESPACE, RECORDS_PATH
from ingest import main as run_ingest
from retrieve import meets_threshold, retrieve

load_dotenv()

POLL_INTERVAL_SECONDS = 2
POLL_MAX_ATTEMPTS = 15

EVAL_RESULTS_PATH = "docs/eval_results.md"
SAMPLE_QUERY = "Which patients have type 2 diabetes and take metformin?"


def get_vector_count(index) -> int:
    """Current vector count for NAMESPACE, or 0 if the namespace is empty
    or doesn't exist yet."""
    stats = index.describe_index_stats()
    summary = stats.namespaces.get(NAMESPACE)
    return summary.vector_count if summary else 0


def poll_for_count(index, expected: int) -> int:
    """Polls up to POLL_MAX_ATTEMPTS times, waiting POLL_INTERVAL_SECONDS
    between checks, until the reported count reaches expected or attempts
    run out. Returns whatever count was last observed either way, so the
    caller decides pass/fail - this function's job is only to give
    eventual consistency a bounded chance to catch up, not to guarantee
    a match.
    """
    for attempt in range(1, POLL_MAX_ATTEMPTS + 1):
        count = get_vector_count(index)
        if count == expected:
            return count
        if attempt < POLL_MAX_ATTEMPTS:
            time.sleep(POLL_INTERVAL_SECONDS)
    return count


def check_credentials() -> bool:
    """Reuses check_connection.py's own checks rather than re-implementing
    auth validation a second time. Includes check_pinecone_query_key() -
    the query-scoped key retrieve.py actually uses at query time - so this
    check doesn't silently stop covering that path now that it's split
    from PINECONE_API_KEY."""
    return (
        check_pinecone()
        and check_pinecone_query_key()
        and check_index_name()
        and check_anthropic()
    )


def check_index_reachable(index_name: str) -> bool:
    """Confirms the index from PINECONE_INDEX_NAME actually exists and
    responds - separate from check_credentials(), which only confirms the
    API key works, not that this specific index is there."""
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        pc.describe_index(index_name)
        print(f"[PASS] Index '{index_name}' is reachable")
        return True
    except Exception as e:
        print(f"[FAIL] Index '{index_name}' is not reachable: {e}")
        return False


def check_chunk_count(index) -> bool:
    """Re-chunks the source file fresh (not a hardcoded number) and
    compares against Pinecone's actual reported vector count."""
    expected = len(chunk_all_records(RECORDS_PATH))
    actual = poll_for_count(index, expected)

    passed = actual == expected
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] Chunk count matches Pinecone vector count: "
        f"expected {expected} (freshly computed from {RECORDS_PATH}), "
        f"got {actual}"
    )
    return passed


def check_idempotency(index) -> bool:
    """Re-runs ingest.py's full delete-and-reload a second time and
    confirms the vector count is unchanged - idempotency verified, not
    assumed."""
    before = get_vector_count(index)
    print(f"Re-running ingest.py (current count: {before})...")

    exit_code = run_ingest()
    if exit_code != 0:
        print(f"[FAIL] ingest.py failed on re-run (exit code {exit_code})")
        return False

    after = poll_for_count(index, before)
    passed = after == before
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] Re-ingestion is idempotent: {before} vectors before, "
        f"{after} vectors after"
    )
    return passed


def check_sample_query() -> bool:
    """Confirms a real question returns at least one chunk that clears the
    score threshold - what the live API actually promises (a relevant
    result), not just the weaker guarantee that some chunk came back at
    all. Uses meets_threshold() the same way run_eval.py does."""
    chunks = retrieve(SAMPLE_QUERY)
    passed = bool(chunks) and meets_threshold(chunks[0]["score"])
    status = "PASS" if passed else "FAIL"
    top_score = chunks[0]["score"] if chunks else None
    print(
        f"[{status}] Sample query returned {len(chunks)} chunk(s), "
        f"top score={top_score}"
    )
    return passed


def check_eval_report_exists() -> bool:
    """Confirms the eval harness has actually been run and its score
    report saved - not just that run_eval.py exists as a file."""
    passed = os.path.exists(EVAL_RESULTS_PATH)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {EVAL_RESULTS_PATH} exists")
    return passed


def main() -> int:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key:
        print("[FAIL] PINECONE_API_KEY not set in .env")
        return 1
    if not index_name:
        print("[FAIL] PINECONE_INDEX_NAME not set in .env")
        return 1

    checks = [check_credentials()]

    if not checks[-1]:
        # No point continuing if credentials are bad - every later check
        # depends on a working Pinecone/Claude connection
        print("\nOne or more checks FAILED. See messages above.")
        return 1

    pc = Pinecone(api_key=api_key)

    try:
        index = pc.Index(name=index_name)
    except NotFoundException:
        print(f"[FAIL] Index '{index_name}' does not exist. Run scripts/create_index.py first.")
        return 1

    checks.append(check_index_reachable(index_name))
    checks.append(check_chunk_count(index))
    checks.append(check_idempotency(index))
    checks.append(check_sample_query())
    checks.append(check_eval_report_exists())

    if all(checks):
        print("\nAll checks PASS.")
        return 0

    print("\nOne or more checks FAILED. See messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
