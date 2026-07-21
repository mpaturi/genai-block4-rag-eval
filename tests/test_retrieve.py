"""Unit tests for the score-threshold decision and metadata filter builder
in retrieve.py.

TDD: written before retrieve.py exists - these tests define the contract
meets_threshold() must satisfy. All should fail with an ImportError until
retrieve.py and meets_threshold() are implemented. This logic is pure and
deterministic (no live Pinecone call needed), so it's covered directly with
fake scores rather than requiring a real search.

build_metadata_filter()'s tests (Phase 7) are pure logic too - no live
Pinecone call needed to verify a dict gets built (or an error gets raised)
correctly from given arguments.
"""
import pytest

from scripts.retrieve import RAGFilterError, build_metadata_filter, meets_threshold


def test_score_above_threshold_is_a_match():
    assert meets_threshold(0.9, threshold=0.75) is True


def test_score_below_threshold_is_not_a_match():
    assert meets_threshold(0.5, threshold=0.75) is False


def test_score_exactly_at_threshold_is_a_match():
    # Spec: score >= threshold counts as a match, not strictly greater - a
    # chunk scoring exactly at the cutoff should still be trusted.
    assert meets_threshold(0.75, threshold=0.75) is True


def test_build_metadata_filter_returns_none_when_nothing_passed():
    # Callers doing pure semantic search must see no behavior change.
    assert build_metadata_filter() is None


def test_build_metadata_filter_condition_only():
    assert build_metadata_filter(condition="Essential hypertension") == {
        "conditions": "Essential hypertension"
    }


def test_build_metadata_filter_drug_only():
    assert build_metadata_filter(drug="Metformin") == {"drugs": "Metformin"}


def test_build_metadata_filter_lab_threshold_above_uses_gt():
    assert build_metadata_filter(lab="SBP", comparison="above", value=140) == {
        "latest_sbp": {"$gt": 140}
    }


def test_build_metadata_filter_lab_threshold_below_uses_lt():
    assert build_metadata_filter(lab="BMI", comparison="below", value=25) == {
        "latest_bmi": {"$lt": 25}
    }


def test_build_metadata_filter_gender_translates_to_stored_value():
    assert build_metadata_filter(gender="M") == {"gender": "Male"}
    assert build_metadata_filter(gender="F") == {"gender": "Female"}


def test_build_metadata_filter_birth_decade_translates_to_stored_string():
    assert build_metadata_filter(birth_decade=1970) == {"year_of_birth_band": "1970s"}


def test_build_metadata_filter_combines_condition_gender_and_birth_decade():
    result = build_metadata_filter(
        condition="Essential hypertension", gender="M", birth_decade=1970
    )
    assert result == {
        "conditions": "Essential hypertension",
        "gender": "Male",
        "year_of_birth_band": "1970s",
    }


def test_build_metadata_filter_combines_condition_and_lab_threshold():
    result = build_metadata_filter(
        condition="Diabetes mellitus type 2", lab="HbA1c", comparison="above", value=7
    )
    assert result == {
        "conditions": "Diabetes mellitus type 2",
        "latest_hba1c": {"$gt": 7},
    }


def test_build_metadata_filter_unrecognized_lab_raises_before_pinecone():
    with pytest.raises(RAGFilterError):
        build_metadata_filter(lab="Cholesterol", comparison="above", value=200)


def test_build_metadata_filter_unrecognized_comparison_raises_before_pinecone():
    with pytest.raises(RAGFilterError):
        build_metadata_filter(lab="SBP", comparison="equal", value=140)


def test_build_metadata_filter_unrecognized_gender_raises_before_pinecone():
    with pytest.raises(RAGFilterError):
        build_metadata_filter(gender="X")


def test_build_metadata_filter_partial_lab_combo_raises():
    with pytest.raises(RAGFilterError):
        build_metadata_filter(lab="SBP")
    with pytest.raises(RAGFilterError):
        build_metadata_filter(lab="SBP", comparison="above")
    with pytest.raises(RAGFilterError):
        build_metadata_filter(comparison="above", value=140)
    with pytest.raises(RAGFilterError):
        build_metadata_filter(value=140)
