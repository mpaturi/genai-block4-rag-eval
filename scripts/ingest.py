"""Loads chunked patient records into the existing Pinecone index. Does NOT
create the index - that is create_index.py's job. Delete-and-reload: clears
the `patients` namespace before re-upserting everything, so re-running
ingestion never leaves stale chunks behind (see docs/spec.md's Pinecone
index design section for why overwriting alone isn't enough).
"""
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

from chunk_records import chunk_all_records

load_dotenv()

# Hardcoded, not .env - an internal design decision, not something that
# legitimately differs between people's setups (see docs/spec.md)
NAMESPACE = "patients"

RECORDS_PATH = "data/raw/graph_export.jsonl"

# Pinecone's real per-call limit for upsert_records on integrated-inference
# (text) indexes
BATCH_SIZE = 96


def strip_null_fields(chunk: dict) -> dict:
    """Drop metadata fields whose value is None.

    Some patients have no recorded value for a given lab (e.g. no glucose
    reading), so graph_export.jsonl stores that field as null. Pinecone's
    upsert_records rejects null outright - a field must be a string,
    number, boolean, or list of strings. Omitting the key entirely (rather
    than substituting a fake value like 0) keeps "not measured" honestly
    distinct from a real reading.
    """
    return {key: value for key, value in chunk.items() if value is not None}


def main() -> int:
    # Validate both required env vars are present before calling the API
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key:
        print("FAIL - PINECONE_API_KEY not set in .env")
        return 1
    if not index_name:
        print("FAIL - PINECONE_INDEX_NAME not set in .env")
        return 1

    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        print(f"FAIL - index '{index_name}' does not exist. Run create_index.py first.")
        return 1

    try:
        index = pc.Index(name=index_name)
    except NotFoundException:
        print(
            f"FAIL - index '{index_name}' does not exist. "
            "Run scripts/create_index.py first."
        )
        return 1

    # Clear the namespace before reloading it fresh. On the very first run
    # the namespace doesn't exist yet, so a missing namespace is treated as
    # already-empty, not an error.
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
        print(f"Cleared namespace '{NAMESPACE}'.")
    except NotFoundException:
        print(f"Namespace '{NAMESPACE}' does not exist yet - nothing to clear.")

    chunks = chunk_all_records(RECORDS_PATH)
    chunks = [strip_null_fields(chunk) for chunk in chunks]
    print(f"Chunked {len(chunks)} records from {RECORDS_PATH}.")

    # Upsert in batches - a single call exceeds Pinecone's per-call record
    # limit for integrated-inference upserts
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        index.upsert_records(records=batch, namespace=NAMESPACE)
        print(f"Upserted records {start} to {start + len(batch) - 1}.")

    print(f"Ingestion complete: {len(chunks)} chunks upserted.")

    # One-shot stats check - describe_index_stats can lag behind the true
    # count due to eventual consistency. Full polling verification is
    # Phase 6's verify.py job, not this script's.
    stats = index.describe_index_stats()
    print(f"Index stats (may lag actual count): {stats}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
