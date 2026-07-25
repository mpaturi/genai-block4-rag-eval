"""Computes ground-truth answer sets for the eval question set, straight
from the raw OMOP CSVs - never hand-typed, never trusted from Block 3's own
computed output. See docs/spec.md's Eval harness design section: this is
what makes the eval score trustworthy rather than circular.

Visit-count and lab-threshold ground truth are both independently
recomputed from visit_occurrence.csv and measurement.csv respectively,
rather than read from graph_export.jsonl's precomputed visit_count/
latest_* fields - the same reasoning applied consistently to both, even
though only the lab-threshold case was originally called out in spec.
"""
import json
import sys

import pandas as pd

from concepts import (
    CONDITION_NAMES,
    DRUG_NAMES,
    GENDER_CONCEPT_ID,
    MEASUREMENT_BMI,
    MEASUREMENT_GLUCOSE,
    MEASUREMENT_HBA1C,
    MEASUREMENT_SBP,
)

DATA_DIR = "data/raw"
QUESTIONS_PATH = "data/eval/questions.json"

# concepts.py maps id -> name for conditions/drugs; questions.json specifies
# them by name, so these reverse lookups flip the direction. Lowercased on
# both sides for robust matching.
CONDITION_ID_BY_NAME = {name.lower(): cid for cid, name in CONDITION_NAMES.items()}
DRUG_ID_BY_NAME = {name.lower(): did for did, name in DRUG_NAMES.items()}

LAB_CONCEPT_ID = {
    "sbp": MEASUREMENT_SBP,
    "bmi": MEASUREMENT_BMI,
    "glucose": MEASUREMENT_GLUCOSE,
    "hba1c": MEASUREMENT_HBA1C,
}

OPERATORS = {
    ">": lambda series, value: series[series > value],
    ">=": lambda series, value: series[series >= value],
    "<": lambda series, value: series[series < value],
    "<=": lambda series, value: series[series <= value],
}


def load_data() -> dict[str, pd.DataFrame]:
    """Load all five raw CSVs used for ground-truth computation."""
    return {
        "conditions": pd.read_csv(f"{DATA_DIR}/condition_occurrence.csv"),
        "drugs": pd.read_csv(f"{DATA_DIR}/drug_exposure.csv"),
        "persons": pd.read_csv(f"{DATA_DIR}/person.csv"),
        "measurements": pd.read_csv(f"{DATA_DIR}/measurement.csv"),
        "visits": pd.read_csv(f"{DATA_DIR}/visit_occurrence.csv"),
    }


def patients_with_condition(conditions_df: pd.DataFrame, condition_name: str) -> set[int]:
    """Returns an empty set for a condition name outside the whitelist (e.g.
    one of the deliberately-unanswerable questions) rather than raising -
    that's the correct ground truth for "no such condition is tracked",
    not a bug to crash on.
    """
    concept_id = CONDITION_ID_BY_NAME.get(condition_name.lower())
    if concept_id is None:
        return set()
    matches = conditions_df[conditions_df["condition_concept_id"] == concept_id]
    return set(matches["person_id"])


def patients_with_drug(drugs_df: pd.DataFrame, drug_name: str) -> set[int]:
    """Same reasoning as patients_with_condition: an out-of-whitelist drug
    name is correctly "no patients", not a KeyError.
    """
    concept_id = DRUG_ID_BY_NAME.get(drug_name.lower())
    if concept_id is None:
        return set()
    matches = drugs_df[drugs_df["drug_concept_id"] == concept_id]
    return set(matches["person_id"])


def patients_matching_demographic(
    persons_df: pd.DataFrame,
    gender: str | None = None,
    birth_decade: int | None = None,
) -> set[int]:
    """gender is 'M' or 'F' (matches concepts.py's GENDER_CONCEPT_ID keys).
    birth_decade is the decade's start year (e.g. 1970 for "the 1970s") -
    inclusive of both endpoints, so 1970 matches years 1970 through 1979.
    """
    df = persons_df
    if gender is not None:
        df = df[df["gender_concept_id"] == GENDER_CONCEPT_ID[gender]]
    if birth_decade is not None:
        df = df[(df["year_of_birth"] >= birth_decade) & (df["year_of_birth"] <= birth_decade + 9)]
    return set(df["person_id"])


def patients_with_visit_count_at_least(visits_df: pd.DataFrame, minimum: int) -> set[int]:
    """Independently recomputed from visit_occurrence.csv - not read from
    graph_export.jsonl's precomputed visit_count field, for the same
    independence reason lab thresholds are recomputed from measurement.csv.
    """
    counts = visits_df.groupby("person_id").size()
    return set(counts[counts >= minimum].index)


def patients_with_lab_threshold(
    measurements_df: pd.DataFrame, lab_name: str, operator: str, value: float
) -> set[int]:
    """Keeps each patient's latest reading by date for the given lab,
    mirroring Block 1's "latest value per concept per patient" logic, then
    compares against value using operator (>, >=, <, or <=).

    Drops non-positive values before picking the latest reading - source
    data uses -1 as a sentinel for "no reading" and none of the tracked
    labs can be zero or negative. Matches Block 3's load_measurements
    logic, which drops these rows for the same reason. Must happen before
    picking the latest reading, or a patient whose most recent row is a
    sentinel loses an otherwise-valid earlier reading.
    """
    concept_id = LAB_CONCEPT_ID.get(lab_name.lower())
    if concept_id is None:
        return set()
    lab_rows = measurements_df[measurements_df["measurement_concept_id"] == concept_id]
    lab_rows = lab_rows[lab_rows["value_as_number"] > 0]
    latest = (
        lab_rows.sort_values("measurement_date")
        .groupby("person_id")
        .tail(1)
        .set_index("person_id")["value_as_number"]
    )
    matches = OPERATORS[operator](latest, value)
    return set(matches.index)


def compute_answer_set(question: dict, data: dict[str, pd.DataFrame]) -> set[int]:
    """Intersects whichever filters a question specifies - a co-occurrence
    question naturally combines condition + drug + demographic this way,
    since a patient must appear in every matching set to count.
    """
    filters = question["filters"]
    sets = []

    if "condition" in filters:
        sets.append(patients_with_condition(data["conditions"], filters["condition"]))
    if "drug" in filters:
        sets.append(patients_with_drug(data["drugs"], filters["drug"]))
    if "gender" in filters or "birth_decade" in filters:
        sets.append(
            patients_matching_demographic(
                data["persons"],
                gender=filters.get("gender"),
                birth_decade=filters.get("birth_decade"),
            )
        )
    if "min_visit_count" in filters:
        sets.append(patients_with_visit_count_at_least(data["visits"], filters["min_visit_count"]))
    if "lab" in filters:
        sets.append(
            patients_with_lab_threshold(
                data["measurements"], filters["lab"], filters["operator"], filters["value"]
            )
        )

    if not sets:
        raise ValueError(f"Question '{question['id']}' has no recognized filters")

    result = sets[0]
    for s in sets[1:]:
        result = result & s
    return result


def build_answer_key(questions: list[dict], data: dict[str, pd.DataFrame]) -> dict[str, set[int]]:
    """Computes every question's answer set and asserts the labeling is
    honest along the way: an answerable question must get a non-empty set,
    and a deliberately-unanswerable question must get an empty one. Fails
    loudly on a mislabeled question rather than silently corrupting the
    eval score later - see docs/spec.md's Eval harness design.
    """
    answer_key = {}
    for question in questions:
        answer_set = compute_answer_set(question, data)
        answer_key[question["id"]] = answer_set

        if question["answerable"] and not answer_set:
            raise AssertionError(
                f"{question['id']} is labeled answerable but computed answer set is empty"
            )
        if not question["answerable"] and answer_set:
            raise AssertionError(
                f"{question['id']} is labeled unanswerable but computed answer set is "
                f"non-empty: {sorted(answer_set)}"
            )

    return answer_key


def main() -> int:
    data = load_data()

    try:
        with open(QUESTIONS_PATH) as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(
            f"FAIL - {QUESTIONS_PATH} not found. This script computes answer "
            "keys for questions that already exist - write data/eval/"
            "questions.json first."
        )
        return 1

    try:
        answer_key = build_answer_key(questions, data)
    except AssertionError as e:
        print(f"FAIL - {e}")
        return 1

    print(f"Computed answer keys for {len(questions)} questions - all labels honest.\n")
    for question in questions:
        count = len(answer_key[question["id"]])
        label = "answerable" if question["answerable"] else "unanswerable"
        print(f"  {question['id']} ({label}): {count} matching patients")

    return 0


if __name__ == "__main__":
    sys.exit(main())
