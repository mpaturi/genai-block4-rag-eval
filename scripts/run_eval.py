"""Eval harness - computes retrieval precision/recall (micro-averaged
across the answerable questions) and fallback accuracy (across the
deliberately-unanswerable ones), per docs/spec.md's Eval harness design
section.

Calls retrieve.py directly, never generate.py or Claude - all three
metrics are fully computable from retrieval results and the threshold
check alone (see spec's Reproducibility and determinism section), and
this also avoids ~20 unnecessary Claude calls on every eval run.

Reuses build_eval_answer_key.py's ground-truth computation rather than
re-deriving it, so there's exactly one place in the repo that knows how
to compute a question's correct patient-ID set.

Run with: python scripts/run_eval.py [--top-k N] [--threshold T]
(from the repo root, matching build_eval_answer_key.py's own convention -
bare imports below assume scripts/ is on sys.path via direct execution,
the opposite of api.py's `scripts.`-prefixed imports for uvicorn).
"""
import argparse
import json
import sys

from build_eval_answer_key import QUESTIONS_PATH, build_answer_key, load_data
from retrieve import DEFAULT_THRESHOLD, DEFAULT_TOP_K, meets_threshold, retrieve, select_threshold

# Phase 7 filtered-eval mode only: questions.json predates the metadata-
# filter integration and reuses concepts.py's casual/lowercase condition
# and lab phrasing (e.g. "type 2 diabetes", "sbp"), which does not match
# Pinecone's stored metadata strings (Block 3's own clinical naming, e.g.
# "Diabetes mellitus type 2", "SBP" as the _LAB_PROPERTY key). This table
# exists only to let this eval harness reuse questions.json's existing
# phrasing - it is not a gap in retrieve.py or build_metadata_filter()
# itself. A real caller (e.g. Block 5's graph_tool.py) already sends the
# correctly-formatted clinical name straight into an exact-match Cypher
# parameter against Neo4j, which shares Block 3's naming with Pinecone -
# no translation happens on Block 5's side either.
_CONDITION_NAME_TRANSLATION = {
    "type 2 diabetes": "Diabetes mellitus type 2",
    "hypertension": "Essential hypertension",
    "hyperlipidemia": "Hyperlipidemia",
    "anemia": "Anemia",
    "osteoporosis": "Osteoporosis",
    "congestive heart failure": "Congestive heart failure",
    "atrial fibrillation": "Atrial fibrillation",
    "streptococcal pharyngitis": "Streptococcal pharyngitis",
    "a urinary tract infection": "Urinary tract infection",
    "a pulmonary embolism": "Pulmonary embolism",
    "osteoarthritis": "Osteoarthritis",
}

# Only "insulin" actually needs a case fix - every other drug questions.json
# uses is already spelled exactly as Pinecone stores it. An out-of-whitelist
# drug (e.g. "albuterol", used by a deliberately-unanswerable question) has
# no entry and is left untranslated - it won't match any patient's stored
# `drugs` list under any spelling, which is the correct behavior for a
# question that's unanswerable specifically because that drug isn't tracked.
_DRUG_NAME_TRANSLATION = {"insulin": "Insulin"}

# retrieve.py's _LAB_PROPERTY keys, matched to questions.json's lowercase
# lab names. Unlike condition/drug, build_metadata_filter() validates lab
# against a fixed whitelist and raises RAGFilterError for anything else -
# so a lab with no entry here (e.g. "cholesterol") can't be translated at
# all, not just imperfectly matched (see translate_filters()).
_LAB_NAME_TRANSLATION = {"sbp": "SBP", "bmi": "BMI", "glucose": "Glucose", "hba1c": "HbA1c"}

# questions.json stores raw comparison symbols; retrieve()'s `comparison`
# parameter takes "above"/"below". Only strict operators are supported
# (Pinecone $gt/$lt) - a >=/<= question would have no entry here and be
# excluded the same way an unrecognized lab is.
_OPERATOR_TRANSLATION = {">": "above", "<": "below"}


def translate_filters(filters: dict) -> dict | None:
    """Translates one question's filters dict (questions.json's phrasing)
    into retrieve()'s filter keyword arguments.

    Returns None if this question's filters can't be represented by
    build_metadata_filter() at all: min_visit_count has no matching
    parameter, and a lab or comparison outside the translation tables
    above would raise RAGFilterError rather than just fail to match -
    those questions are excluded from the filtered eval run rather than
    silently sent through broken.
    """
    if "min_visit_count" in filters:
        return None

    kwargs: dict = {}

    if "condition" in filters:
        kwargs["condition"] = _CONDITION_NAME_TRANSLATION.get(
            filters["condition"], filters["condition"]
        )
    if "drug" in filters:
        kwargs["drug"] = _DRUG_NAME_TRANSLATION.get(filters["drug"], filters["drug"])
    if "gender" in filters:
        kwargs["gender"] = filters["gender"]
    if "birth_decade" in filters:
        kwargs["birth_decade"] = filters["birth_decade"]

    if "lab" in filters:
        lab = _LAB_NAME_TRANSLATION.get(filters["lab"])
        comparison = _OPERATOR_TRANSLATION.get(filters["operator"])
        if lab is None or comparison is None:
            return None
        kwargs["lab"] = lab
        kwargs["comparison"] = comparison
        kwargs["value"] = filters["value"]

    return kwargs


def run_eval(
    questions: list[dict],
    answer_key: dict[str, set[int]],
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    filtered: bool = False,
    conditional_threshold: bool = False,
) -> dict:
    """Runs every question's text through retrieve(), scores it against
    answer_key, and returns aggregated metrics plus per-question detail.

    Precision/recall are micro-averaged: correct/retrieved/actual counts
    are summed across all answerable questions first, then divided once -
    not averaged per-question - so one question's edge case can't
    disproportionately swing the headline number.

    filtered=True (Phase 7) additionally translates each question's
    filters dict into retrieve()'s metadata-filter arguments via
    translate_filters(). A question whose filters can't be represented
    (min_visit_count, or a lab/comparison outside the translation tables)
    is excluded entirely - not scored, not counted as a miss - and listed
    in the returned "excluded_ids".

    conditional_threshold=True (Phase 7, Run 8) additionally computes
    each question's own threshold via select_threshold() from its
    translated condition/drug filters, matching scripts/api.py's live
    per-request behavior, instead of using the single flat `threshold`
    for every question. Only valid alongside filtered=True - the caller
    (main()) enforces this before run_eval() is ever called.
    """
    total_correct = 0
    total_retrieved = 0
    total_actual = 0
    fallback_correct = 0
    fallback_total = 0
    per_question = []
    excluded_ids = []

    for question in questions:
        filter_kwargs = {}
        if filtered:
            translated = translate_filters(question["filters"])
            if translated is None:
                excluded_ids.append(question["id"])
                continue
            filter_kwargs = translated

        chunks = retrieve(question["question"], top_k=top_k, **filter_kwargs)

        if conditional_threshold:
            question_threshold = select_threshold(
                condition=filter_kwargs.get("condition"), drug=filter_kwargs.get("drug")
            )
        else:
            question_threshold = threshold

        # Same fallback decision api.py makes: nothing retrieved, or the
        # top-ranked chunk doesn't clear threshold. Equivalent to "no
        # chunk clears threshold" since retrieve() returns Pinecone's
        # ranked order (highest score first) - checking just chunks[0]
        # mirrors api.py's actual short-circuit rather than re-deriving
        # the same result a different way.
        triggered_fallback = not chunks or not meets_threshold(chunks[0]["score"], question_threshold)

        # "Retrieved" mirrors exactly what Job 2 would see: only chunks
        # that individually clear the threshold, deduped to unique
        # person_ids since one patient can contribute more than one chunk.
        relevant_chunks = [c for c in chunks if meets_threshold(c["score"], question_threshold)]
        retrieved_ids = {c["person_id"] for c in relevant_chunks}

        if question["answerable"]:
            correct_ids = retrieved_ids & answer_key[question["id"]]
            total_correct += len(correct_ids)
            total_retrieved += len(retrieved_ids)
            total_actual += len(answer_key[question["id"]])
            per_question.append(
                {
                    "id": question["id"],
                    "answerable": True,
                    "retrieved": len(retrieved_ids),
                    "correct": len(correct_ids),
                    "actual": len(answer_key[question["id"]]),
                }
            )
        else:
            fallback_total += 1
            if triggered_fallback:
                fallback_correct += 1
            per_question.append(
                {
                    "id": question["id"],
                    "answerable": False,
                    "triggered_fallback": triggered_fallback,
                }
            )

    precision = total_correct / total_retrieved if total_retrieved else 0.0
    recall = total_correct / total_actual if total_actual else 0.0
    fallback_accuracy = fallback_correct / fallback_total if fallback_total else 0.0

    return {
        "top_k": top_k,
        "threshold": threshold,
        "filtered": filtered,
        "conditional_threshold": conditional_threshold,
        "precision": precision,
        "recall": recall,
        "fallback_accuracy": fallback_accuracy,
        "total_correct": total_correct,
        "total_retrieved": total_retrieved,
        "total_actual": total_actual,
        "fallback_correct": fallback_correct,
        "fallback_total": fallback_total,
        "per_question": per_question,
        "excluded_ids": excluded_ids,
    }


def print_report(metrics: dict) -> None:
    print(
        f"top_k={metrics['top_k']}  threshold={metrics['threshold']}  "
        f"filtered={metrics['filtered']}  conditional_threshold={metrics['conditional_threshold']}\n"
    )
    print(
        f"  Precision:         {metrics['precision']:.3f}  "
        f"({metrics['total_correct']}/{metrics['total_retrieved']})"
    )
    print(
        f"  Recall:            {metrics['recall']:.3f}  "
        f"({metrics['total_correct']}/{metrics['total_actual']})"
    )
    print(
        f"  Fallback accuracy: {metrics['fallback_accuracy']:.3f}  "
        f"({metrics['fallback_correct']}/{metrics['fallback_total']})"
    )
    if metrics["excluded_ids"]:
        print(f"  Excluded (unrepresentable filter): {', '.join(metrics['excluded_ids'])}")
    print()
    for pq in metrics["per_question"]:
        if pq["answerable"]:
            print(f"  {pq['id']}: retrieved={pq['retrieved']} correct={pq['correct']} actual={pq['actual']}")
        else:
            status = "OK (fell back)" if pq["triggered_fallback"] else "MISS (should have fallen back)"
            print(f"  {pq['id']}: {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Translate each question's filters into retrieve()'s metadata "
        "filter arguments (Phase 7). Questions whose filters can't be "
        "represented (min_visit_count, or a lab/comparison outside the "
        "translation tables) are excluded from the run.",
    )
    parser.add_argument(
        "--conditional-threshold",
        action="store_true",
        help="Only valid with --filtered. Instead of using a single flat "
        "--threshold for every question, compute each question's threshold "
        "via select_threshold() from its own translated condition/drug "
        "filters - matching scripts/api.py's live per-request behavior "
        "instead of a fixed value for the whole run.",
    )
    args = parser.parse_args()

    if args.conditional_threshold and not args.filtered:
        parser.error("--conditional-threshold requires --filtered")

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    data = load_data()
    try:
        answer_key = build_answer_key(questions, data)
    except AssertionError as e:
        print(f"FAIL - {e}")
        return 1

    metrics = run_eval(
        questions,
        answer_key,
        top_k=args.top_k,
        threshold=args.threshold,
        filtered=args.filtered,
        conditional_threshold=args.conditional_threshold,
    )
    print_report(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
