"""Job 2 - has Claude write a cited answer from chunks retrieval already
found. See docs/spec.md's Generation design section.

Library function only, no CLI main() - api.py is the only caller, and it
owns the score-check decision (call this or return "I don't know") per the
architecture diagram in docs/spec.md, which draws that check as a separate
step between retrieve.py and generate.py, not inside either one. This
function assumes the chunks it's given have already cleared the threshold;
it doesn't re-check scores itself.
"""
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a clinical assistant answering questions about patients using "
    "only the patient record excerpts provided below. Never use outside "
    "knowledge or make assumptions beyond what the excerpts say. For every "
    "claim you make, cite the person_id of the patient it comes from, "
    "e.g. 'Patient 248 takes Metformin (person_id: 248).' If the provided "
    "excerpts don't fully answer the question, say so rather than "
    "guessing."
)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Ask Claude to answer question using only the given chunks.

    chunks is expected to already be filtered to the ones that individually
    cleared the retrieval score threshold - this function does not filter
    or validate scores itself.
    """
    # Validated here, not just by a caller - this is a library function
    # api.py calls directly, matching retrieve.py's pattern
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    context = "\n\n".join(
        f"[person_id: {chunk['person_id']}] {chunk['chunk_text']}"
        for chunk in chunks
    )
    user_message = f"Question: {question}\n\nPatient record excerpts:\n{context}"

    # Real API errors (auth failure, network issues, etc.) propagate
    # uncaught - converting those into a clean HTTP 502 is api.py's job
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
