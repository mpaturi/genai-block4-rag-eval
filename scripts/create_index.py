"""Creates the Pinecone serverless index with integrated inference, if it
doesn't already exist. Idempotent - safe to re-run. Prints the scoring
metric (cosine/dotproduct = higher-is-better, euclidean = lower-is-better)
so the threshold direction can start being confirmed without needing any
data yet - Phase 3's empirical self-match check confirms it for real once
chunks exist.
"""
import os
import sys
import time

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "llama-text-embed-v2"
FIELD_MAP = {"text": "chunk_text"}
CLOUD = "aws"
REGION = "us-east-1"

POLL_INTERVAL_SECONDS = 2
POLL_MAX_ATTEMPTS = 15


def wait_until_ready(pc: Pinecone, index_name: str):
    """Polls describe_index() in a bounded loop until status.ready is True,
    rather than a single fixed sleep - closes the race where a caller
    (e.g. run_all.py) tries to use the index immediately after creation,
    before Pinecone has actually finished provisioning it.

    Returns the final IndexDescription either way, so the caller can check
    readiness itself and decide pass/fail.
    """
    for attempt in range(1, POLL_MAX_ATTEMPTS + 1):
        desc = pc.describe_index(index_name)
        if desc.status.ready:
            return desc
        if attempt < POLL_MAX_ATTEMPTS:
            time.sleep(POLL_INTERVAL_SECONDS)
    return desc


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

    # Idempotency check: only create the index if it doesn't already exist
    existing_names = pc.list_indexes().names()
    if index_name in existing_names:
        desc = pc.describe_index(index_name)
        print(f"Index '{index_name}' already exists - skipping creation.")
        print(f"Metric: {desc.metric}")
        return 0

    # Integrated inference: Pinecone embeds `chunk_text` server-side, so we
    # never call a separate embedding API anywhere in this codebase
    pc.create_index_for_model(
        name=index_name,
        cloud=CLOUD,
        region=REGION,
        embed={
            "model": EMBEDDING_MODEL,
            "field_map": FIELD_MAP,
        },
    )

    desc = wait_until_ready(pc, index_name)
    if not desc.status.ready:
        print(
            f"FAIL - index '{index_name}' was created but did not become "
            f"ready within {POLL_MAX_ATTEMPTS} attempts "
            f"({POLL_INTERVAL_SECONDS}s apart). Last status: {desc.status.state}"
        )
        return 1

    print(f"Index '{index_name}' created.")
    print(f"Metric: {desc.metric}")
    if desc.metric in ("cosine", "dotproduct"):
        print("-> higher score = more relevant")
    elif desc.metric == "euclidean":
        print("-> lower score = more relevant")
    else:
        print(f"-> unrecognized metric '{desc.metric}', confirm manually")

    return 0


if __name__ == "__main__":
    sys.exit(main())
