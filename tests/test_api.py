"""Integration-style tests for scripts/api.py's /query endpoint.

Mocks retrieve() and generate_answer() rather than hitting real Pinecone/
Claude - these tests exist to prove the endpoint wires select_threshold()
in correctly (see scripts/retrieve.py), which no unit test on
select_threshold() alone can prove, since that requires api.py to
actually call it with the right arguments and use the result consistently
for both the fallback check and the relevant_chunks filter.
"""
from fastapi.testclient import TestClient

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
