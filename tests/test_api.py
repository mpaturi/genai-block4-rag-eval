"""Integration-style tests for scripts/api.py's /query endpoint.

Mocks retrieve() and generate_answer() rather than hitting real Pinecone/
Claude - these tests exist to prove the endpoint wires select_threshold()
in correctly (see scripts/retrieve.py), which no unit test on
select_threshold() alone can prove, since that requires api.py to
actually call it with the right arguments and use the result consistently
for both the fallback check and the relevant_chunks filter.
"""
import anthropic
from fastapi.testclient import TestClient
from pinecone import PineconeTimeoutError

from scripts.api import app

client = TestClient(app)

# Between PERMISSIVE_THRESHOLD (0.2) and DEFAULT_THRESHOLD (0.4) - the one
# score value that tells the two thresholds apart.
MID_THRESHOLD_SCORE = 0.3

FAKE_CHUNK = {
    "chunk_id": "123_chunk0",
    "score": MID_THRESHOLD_SCORE,
    "person_id": 123,
    "chunk_text": "Patient 123, born in the 1980s, Male. Conditions: none.",
}


def test_gender_only_filter_does_not_trigger_permissive_threshold(monkeypatch):
    # No condition/drug - a gender-only filter provides no structural
    # protection against an off-topic question, so this must stay at
    # DEFAULT_THRESHOLD (0.4) and fall back, even though the chunk's score
    # (0.3) would clear PERMISSIVE_THRESHOLD (0.2).
    monkeypatch.setattr("scripts.api.retrieve", lambda *args, **kwargs: [FAKE_CHUNK])

    response = client.post(
        "/query", json={"question": "Which patients have asthma?", "gender": "F"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieved_count"] == 0
    assert body["sources"] == []


def test_condition_filter_allows_permissive_threshold(monkeypatch):
    # A condition filter can select PERMISSIVE_THRESHOLD (0.2) - the same
    # 0.3-scoring chunk that fell back above should now count as a match.
    monkeypatch.setattr("scripts.api.retrieve", lambda *args, **kwargs: [FAKE_CHUNK])
    monkeypatch.setattr("scripts.api.generate_answer", lambda question, chunks: "fake answer")

    response = client.post(
        "/query", json={"question": "Which patients have asthma?", "condition": "Asthma"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieved_count"] == 1
    assert body["answer"] == "fake answer"
    for source in body["sources"]:
        assert isinstance(source["chunk_text"], str)
        assert source["chunk_text"] != ""


# FILTERED_TOP_K_CEILING (25) mirrors select_threshold()'s permissive gate -
# a condition/drug filter raises the top_k ceiling, everything else stays at
# DEFAULT_TOP_K (20). See scripts/retrieve.py's FILTERED_TOP_K_CEILING.


def test_filtered_request_at_ceiling_top_k_succeeds(monkeypatch):
    monkeypatch.setattr("scripts.api.retrieve", lambda *args, **kwargs: [FAKE_CHUNK])
    monkeypatch.setattr("scripts.api.generate_answer", lambda question, chunks: "fake answer")

    response = client.post(
        "/query",
        json={"question": "Which patients have asthma?", "condition": "Asthma", "top_k": 25},
    )

    assert response.status_code == 200


def test_filtered_request_above_ceiling_top_k_is_rejected():
    response = client.post(
        "/query",
        json={"question": "Which patients have asthma?", "condition": "Asthma", "top_k": 26},
    )

    assert response.status_code == 422


def test_unfiltered_request_above_default_top_k_is_still_rejected():
    # No condition/drug filter - the ceiling stays at DEFAULT_TOP_K (20),
    # unchanged by FILTERED_TOP_K_CEILING.
    response = client.post(
        "/query", json={"question": "Which patients have asthma?", "top_k": 21}
    )

    assert response.status_code == 422


def test_unrecognized_lab_name_returns_422_not_502():
    # build_metadata_filter() raises RAGFilterError before any Pinecone
    # call is made (see scripts/retrieve.py), so no mocking is needed here
    # - this is bad input, not a service failure, and must not be lumped
    # into the generic 502 the except Exception branch returns for real
    # Pinecone/Claude outages.
    response = client.post(
        "/query",
        json={
            "question": "Which patients have high cholesterol?",
            "lab": "Cholesterol",
            "comparison": "above",
            "value": 200,
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "Invalid filter arguments."


# scripts/retrieve.py's SEARCH_TIMEOUT_SECONDS and scripts/generate.py's
# GENERATE_TIMEOUT_SECONDS raise PineconeTimeoutError / anthropic.APITimeoutError
# on a hang, confirmed live against the real clients - these tests prove
# api.py's existing broad `except Exception` genuinely catches those real
# exception types and converts them into the standard 502, rather than
# assuming it does because both are technically Exception subclasses.


def test_pinecone_timeout_returns_502_not_a_hang(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise PineconeTimeoutError("timed out")

    monkeypatch.setattr("scripts.api.retrieve", raise_timeout)

    response = client.post("/query", json={"question": "Which patients have asthma?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "PineconeTimeoutError"


def test_over_length_question_returns_422():
    # Phase 11: question is capped at max_length=500 - the field that
    # actually reaches generate_answer's prompt (see QueryRequest in
    # scripts/api.py). One character over the cap must be rejected before
    # Pinecone or Claude is ever called.
    response = client.post(
        "/query", json={"question": "a" * 501}
    )

    assert response.status_code == 422


def test_over_length_condition_returns_422():
    # Phase 11: condition is capped at max_length=100 - a Pinecone metadata
    # filter value only, not something that reaches generate_answer's
    # prompt (see QueryRequest in scripts/api.py). A valid question is
    # included so the oversized condition is the only thing that should
    # trigger this 422.
    response = client.post(
        "/query",
        json={"question": "Which patients have asthma?", "condition": "a" * 101},
    )

    assert response.status_code == 422


def test_over_length_drug_returns_422():
    # Phase 11: drug is capped at max_length=100 - same reasoning as
    # condition above, a Pinecone metadata filter value only. A valid
    # question is included so the oversized drug is the only thing that
    # should trigger this 422.
    response = client.post(
        "/query",
        json={"question": "Which patients have asthma?", "drug": "a" * 101},
    )

    assert response.status_code == 422


def test_over_length_lab_returns_422():
    # Phase 11: lab is capped at max_length=20, not 100 like
    # condition/drug - its whitelist entries (SBP/BMI/Glucose/HbA1c) top
    # out at 7 chars. comparison and value are included, both otherwise
    # valid, so the oversized lab value is the only thing that should
    # trigger this 422 - not the separate lab/comparison/value
    # all-or-nothing validator.
    response = client.post(
        "/query",
        json={
            "question": "Which patients have high blood pressure?",
            "lab": "a" * 21,
            "comparison": "above",
            "value": 140,
        },
    )

    assert response.status_code == 422


def test_claude_timeout_returns_502_not_a_hang(monkeypatch):
    monkeypatch.setattr("scripts.api.retrieve", lambda *args, **kwargs: [FAKE_CHUNK])

    def raise_timeout(*args, **kwargs):
        raise anthropic.APITimeoutError(request=None)

    monkeypatch.setattr("scripts.api.generate_answer", raise_timeout)

    response = client.post(
        "/query", json={"question": "Which patients have asthma?", "condition": "Asthma"}
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "APITimeoutError"
