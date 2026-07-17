"""FastAPI wrapper exposing the RAG pipeline as POST /query. Wires retrieve.py
(Job 1) and generate.py (Job 2) together per docs/spec.md's API design:
validates input, skips Claude entirely when nothing clears the retrieval
threshold, and returns a clean 502 (never a raw stack trace) if Pinecone or
Claude fails. Per spec's Known limitations, Pinecone and Claude failures
deliberately share one generic error response - not distinguished.

Run with: uvicorn scripts.api:app --reload (from the repo root). Unlike the
other scripts in this project, this file uses `scripts.`-prefixed absolute
imports rather than bare ones - uvicorn resolves the "scripts.api:app"
module path with the repo root already on sys.path (it inserts the current
working directory itself), the opposite situation from `python scripts/x.py`
direct execution used elsewhere.
"""
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from scripts.generate import generate_answer
from scripts.retrieve import meets_threshold, retrieve

logger = logging.getLogger(__name__)

app = FastAPI()

FALLBACK_ANSWER = (
    "I don't know - I couldn't find any patient records relevant to that question."
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, value: int) -> int:
        if not (1 <= value <= 20):
            raise ValueError("top_k must be between 1 and 20")
        return value


@app.post("/query")
def query(request: QueryRequest):
    try:
        chunks = retrieve(request.question, top_k=request.top_k)

        if not chunks or not meets_threshold(chunks[0]["score"]):
            return {"answer": FALLBACK_ANSWER, "sources": [], "retrieved_count": 0}

        relevant_chunks = [c for c in chunks if meets_threshold(c["score"])]
        answer = generate_answer(request.question, relevant_chunks)

    except Exception as e:
        # Server logs the real exception - the caller-facing response
        # deliberately doesn't include it (see below), per spec's API
        # design: "never returns a raw stack trace to the caller"
        logger.exception("query failed")

        # Pinecone and Claude failures deliberately share one generic
        # response per spec's Known limitations - not distinguished
        return JSONResponse(
            status_code=502,
            content={
                "error": "Retrieval service unavailable. Please try again shortly.",
                "detail": type(e).__name__,
            },
        )

    sources = [
        {"person_id": c["person_id"], "chunk_id": c["chunk_id"], "score": c["score"]}
        for c in relevant_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(relevant_chunks),
    }
