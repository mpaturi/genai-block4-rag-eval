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
from retrieve import DEFAULT_THRESHOLD, DEFAULT_TOP_K, meets_threshold, retrieve


def run_eval(
    questions: list[dict],
    answer_key: dict[str, set[int]],
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Runs every question's text through retrieve(), scores it against
    answer_key, and returns aggregated metrics plus per-question detail.

    Precision/recall are micro-averaged: correct/retrieved/actual counts
    are summed across all answerable questions first, then divided once -
    not averaged per-question - so one question's edge case can't
    disproportionately swing the headline number.
    """
    total_correct = 0
    total_retrieved = 0
    total_actual = 0
    fallback_correct = 0
    fallback_total = 0
    per_question = []

    for question in questions:
        chunks = retrieve(question["question"], top_k=top_k)

        # Same fallback decision api.py makes: nothing retrieved, or the
        # top-ranked chunk doesn't clear threshold. Equivalent to "no
        # chunk clears threshold" since retrieve() returns Pinecone's
        # ranked order (highest score first) - checking just chunks[0]
        # mirrors api.py's actual short-circuit rather than re-deriving
        # the same result a different way.
        triggered_fallback = not chunks or not meets_threshold(chunks[0]["score"], threshold)

        # "Retrieved" mirrors exactly what Job 2 would see: only chunks
        # that individually clear the threshold, deduped to unique
        # person_ids since one patient can contribute more than one chunk.
        relevant_chunks = [c for c in chunks if meets_threshold(c["score"], threshold)]
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
        "precision": precision,
        "recall": recall,
        "fallback_accuracy": fallback_accuracy,
        "total_correct": total_correct,
        "total_retrieved": total_retrieved,
        "total_actual": total_actual,
        "fallback_correct": fallback_correct,
        "fallback_total": fallback_total,
        "per_question": per_question,
    }


def print_report(metrics: dict) -> None:
    print(f"top_k={metrics['top_k']}  threshold={metrics['threshold']}\n")
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
    args = parser.parse_args()

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    data = load_data()
    try:
        answer_key = build_answer_key(questions, data)
    except AssertionError as e:
        print(f"FAIL - {e}")
        return 1

    metrics = run_eval(questions, answer_key, top_k=args.top_k, threshold=args.threshold)
    print_report(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
