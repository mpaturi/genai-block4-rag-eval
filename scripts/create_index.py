"""Creates the Pinecone serverless index with integrated inference, if it
doesn't already exist. Idempotent - safe to re-run. Prints the scoring
metric (cosine/dotproduct = higher-is-better, euclidean = lower-is-better)
so the threshold direction can start being confirmed without needing any
data yet - Phase 3's empirical self-match check confirms it for real once
chunks exist.
"""
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "llama-text-embed-v2"
FIELD_MAP = {"text": "chunk_text"}
CLOUD = "aws"
REGION = "us-east-1"


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

    desc = pc.describe_index(index_name)
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
