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
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from scripts.generate import generate_answer
from scripts.retrieve import (
    DEFAULT_TOP_K,
    FILTERED_TOP_K_CEILING,
    RAGFilterError,
    meets_threshold,
    retrieve,
    select_threshold,
)

app = FastAPI()
logger = logging.getLogger(__name__)

FALLBACK_ANSWER = (
    "I don't know - I couldn't find any patient records relevant to that question."
)


class QueryRequest(BaseModel):
    # max_length=500 - this is the field that actually reaches an LLM
    # prompt (generate_answer(), called unconditionally from /query below),
    # so it's the one genuinely prompt-injection-relevant field here. 500
    # is generous for a natural-language clinical question (the real
    # eval set in data/eval/questions.json tops out at 98 chars) while
    # still bounding how much attacker-controlled text a single request
    # can push toward Claude.
    question: str = Field(max_length=500)
    top_k: int = 20
    # Phase 7 - optional structured metadata filters, all None by default
    # so a request using none of them behaves exactly as before Phase 7.
    #
    # condition/drug/lab below are never seen by generate_answer's prompt -
    # they're only used to build a Pinecone metadata filter
    # (build_metadata_filter() in retrieve.py) and are validated there
    # against fixed whitelists before any Pinecone call. The max_length
    # caps here are not a prompt-injection defense; they just keep
    # obviously-invalid oversized values from propagating past the API
    # boundary at all. 100 comfortably covers real values (longest
    # observed condition/drug name in data/raw/graph_export.jsonl is 25/19
    # chars); lab's whitelist entries (SBP/BMI/Glucose/HbA1c) top out at 7.
    condition: str | None = Field(default=None, max_length=100)
    drug: str | None = Field(default=None, max_length=100)
    lab: str | None = Field(default=None, max_length=20)
    comparison: Literal["above", "below"] | None = None
    value: float | None = None
    gender: Literal["M", "F"] | None = None
    birth_decade: int | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def top_k_in_range(self) -> "QueryRequest":
        # A condition/drug filter gets the higher FILTERED_TOP_K_CEILING,
        # mirroring select_threshold()'s same condition/drug gate below -
        # needs to see both fields, so this can't be a single-field
        # field_validator anymore.
        ceiling = (
            FILTERED_TOP_K_CEILING
            if (self.condition is not None or self.drug is not None)
            else DEFAULT_TOP_K
        )
        if not (1 <= self.top_k <= ceiling):
            raise ValueError(f"top_k must be between 1 and {ceiling}")
        return self

    @model_validator(mode="after")
    def lab_comparison_value_all_or_nothing(self) -> "QueryRequest":
        # condition/drug/gender/birth_decade are each independently
        # optional - only this trio must be given together, since a
        # partial lab filter is ambiguous (see docs/spec.md's Metadata
        # filter design).
        lab_args_present = [
            self.lab is not None,
            self.comparison is not None,
            self.value is not None,
        ]
        if any(lab_args_present) and not all(lab_args_present):
            raise ValueError(
                "lab, comparison, and value must be given together - "
                f"got lab={self.lab!r}, comparison={self.comparison!r}, "
                f"value={self.value!r}"
            )
        return self


@app.post("/query")
def query(request: QueryRequest):
    try:
        chunks = retrieve(
            request.question,
            top_k=request.top_k,
            condition=request.condition,
            drug=request.drug,
            lab=request.lab,
            comparison=request.comparison,
            value=request.value,
            gender=request.gender,
            birth_decade=request.birth_decade,
        )

        threshold = select_threshold(condition=request.condition, drug=request.drug)

        if not chunks or not meets_threshold(chunks[0]["score"], threshold):
            return {"answer": FALLBACK_ANSWER, "sources": [], "retrieved_count": 0}

        relevant_chunks = [c for c in chunks if meets_threshold(c["score"], threshold)]
        answer = generate_answer(request.question, relevant_chunks)

    except RAGFilterError as e:
        # Invalid filter arguments (e.g. an unrecognized lab name) - bad
        # input, not a service failure, so this is a 422, not the generic
        # 502 below.
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid filter arguments.", "detail": str(e)},
        )

    except Exception as e:
        # Full traceback goes to the server log; the response body stays
        # generic - Pinecone and Claude failures deliberately share one
        # generic response per spec's Known limitations, not distinguished
        logger.exception("query failed")
        return JSONResponse(
            status_code=502,
            content={
                "error": "Retrieval service unavailable. Please try again shortly.",
                "detail": type(e).__name__,
            },
        )

    sources = [
        {
            "person_id": c["person_id"],
            "chunk_id": c["chunk_id"],
            "score": c["score"],
            "chunk_text": c["chunk_text"],
        }
        for c in relevant_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_count": len(relevant_chunks),
    }
