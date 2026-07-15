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

# Starting default per docs/spec.md's Retrieval design - confirmed
# higher-is-better against the real index in Phase 3's self-match check
DEFAULT_THRESHOLD = 0.75
DEFAULT_TOP_K = 5


def meets_threshold(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """A chunk scoring exactly at the threshold still counts as a match."""
    return score >= threshold


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search Pinecone for the top_k chunks most similar to question.

    Returns a list of dicts, each holding the chunk's id, similarity
    score, and full metadata (chunk_text, person_id, etc.) - ranked
    highest score first, same order Pinecone returns.
    """
    # Validate env vars here, not just in main() - retrieve() is a library
    # function future scripts (api.py, run_eval.py) will call directly,
    # without ever going through this file's main()
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in .env")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME not set in .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(name=index_name)

    result = index.search(
        namespace=NAMESPACE, top_k=top_k, inputs={"text": question}
    )

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
