"""Unit tests for the score-threshold decision in retrieve.py.

TDD: written before retrieve.py exists - these tests define the contract
meets_threshold() must satisfy. All should fail with an ImportError until
retrieve.py and meets_threshold() are implemented. This logic is pure and
deterministic (no live Pinecone call needed), so it's covered directly with
fake scores rather than requiring a real search.
"""
from scripts.retrieve import meets_threshold


def test_score_above_threshold_is_a_match():
    assert meets_threshold(0.9, threshold=0.75) is True


def test_score_below_threshold_is_not_a_match():
    assert meets_threshold(0.5, threshold=0.75) is False


def test_score_exactly_at_threshold_is_a_match():
    # Spec: score >= threshold counts as a match, not strictly greater - a
    # chunk scoring exactly at the cutoff should still be trusted.
    assert meets_threshold(0.75, threshold=0.75) is True
