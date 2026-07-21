# Eval Results

This file is produced by `scripts/run_eval.py` (Phase 5) and is committed as
evidence for the "documents ≥1 experiment" acceptance criterion — see
`docs/spec.md`'s Eval harness design section.

All runs below use the same 20-question set in `data/eval/questions.json`
(15 answerable, 5 deliberately unanswerable); each run's specific `top_k`
and `threshold` settings are listed in its own heading below. Ground truth
was independently recomputed by `scripts/build_eval_answer_key.py` and
spot-checked by hand (see Spot-check section below) before any run.

## Known limitations

- **`visit_count` includes dirty-data duplicate rows, not just distinct
  encounters.** Block 1's generator caps real visits at `VISITS_PER_PERSON`
  (2) per patient, but `_inject_dirty_data()` runs afterward and duplicates
  some visit rows as deliberately injected data-quality noise — one patient
  ends up with 4 visit rows, 343 with 3. `build_eval_answer_key.py`'s
  high-burden/visit-count questions (q10-q12 in `data/eval/questions.json`)
  count these raw rows, duplicates included, which is intentional: that's
  exactly what `graph_export.jsonl`'s `visit_count` metadata field also
  counts, so ground truth stays consistent with what the RAG pipeline
  actually ingests and can retrieve. It does mean "3+ visits" in these
  questions doesn't always mean 3+ distinct clinical encounters — for some
  patients it's 2 real visits plus 1 injected duplicate.

## Run 1 — default settings (`top_k=5`, `threshold=0.75`)

| Metric | Value |
|---|---|
| Precision | 0.000 (0/0) |
| Recall | 0.000 (0/192) |
| Fallback accuracy | 1.000 (5/5) |

Every one of the 15 answerable questions retrieved zero chunks above
threshold — none of their real similarity scores reached 0.75 (observed
top scores for answerable questions ranged ~0.42–0.55). Fallback accuracy
is a perfect 1.000, but that number is not meaningful here: at this
threshold, *every* question falls back regardless of whether it's actually
answerable, so a system that always says "I don't know" would score
identically. The 0.75 default was carried over from the empirical
self-match check in Phase 3, which queried Pinecone using a chunk's own
exact text (score 0.82–0.84) — a very different, much easier case than a
natural-language question being matched against chunk text.

## Run 2 — experiment: lower `threshold` to 0.4 (`top_k=5`, `threshold=0.4`)

| Metric | Value |
|---|---|
| Precision | 0.197 (14/71) |
| Recall | 0.073 (14/192) |
| Fallback accuracy | 0.800 (4/5) |

**What changed and why:** `threshold` was lowered from 0.75 to 0.4, chosen
by inspecting real retrieval scores across all 20 questions beforehand
(not guessed) — the lowest top score among the 15 answerable questions was
0.419 (q14), so 0.4 was the highest threshold that would let every
answerable question's top hit clear at all.

**Result:** Precision and recall move from completely dead (0.000/0.000)
to non-zero (0.197/0.073) — retrieval starts actually returning correct
patients instead of nothing. The cost is fallback accuracy dropping from
1.000 to 0.800: `q20` ("chronic kidney disease" — outside the 11-condition
whitelist) has a top score of 0.431, just above the new 0.4 cutoff, so it
incorrectly stops falling back and is treated as a match. This is exactly
the dual-purpose-threshold tradeoff `docs/spec.md`'s Known limitations
section already calls out: the same threshold gates both "is this
relevant enough" and "is this in scope at all," and those two questions
don't always agree at any single cutoff. Recall staying low (0.073) even
after lowering the threshold reflects `top_k=5` capping retrieval at 5
chunks per question while several answerable questions have 15–22 correct
patients in their ground-truth set — recall is structurally bounded by
`top_k` regardless of threshold, so a further experiment raising `top_k`
(not attempted here, since spec requires only one changed setting) would
likely move recall further than threshold tuning alone can.

**Adopted as the new default:** based on this experiment, `0.4` was
subsequently adopted as `scripts/retrieve.py`'s `DEFAULT_THRESHOLD`,
replacing the original `0.75` — `0.75` returned zero relevant results for
every real question tested, which is worse than the precision/fallback
tradeoff `0.4` introduces. Run 1 and Run 2 above are left as originally
recorded; they're the documented experiment this file exists to capture,
not something to retroactively rewrite.

## Run 3 — experiment: raise `top_k` to 25 (`top_k=25`, `threshold=0.4`)

| Metric | Value |
|---|---|
| Precision | 0.115 (35/305) |
| Recall | 0.182 (35/192) |
| Fallback accuracy | 0.800 (4/5) |

**What changed and why:** `top_k` was raised from 5 to 25, `threshold` held
at 0.4 (Run 2's value) so `top_k` is the only variable changed relative to
Run 2. There are 192 correct patients total across the 15 answerable
questions (~13 per question on average). With `top_k=5`, at most 5 correct
patients could be retrieved per question, so the theoretical ceiling on
recall was roughly 75/192 ≈ 0.39 — meaning Run 2's 0.073 recall was partly
capped by `top_k`, not solely reflecting retrieval quality. `top_k=25` was
chosen because it's above every question's actual answer-set size (4–22
patients, per-question), so `top_k` no longer caps any question's recall
at all — whatever recall results now reflects retrieval quality and the
threshold, not the `top_k` limit.

**Result:** Recall roughly doubled+ — from 0.073 (Run 2) to 0.182, a real,
meaningful move, confirming `top_k=5` was indeed a genuine bottleneck in
Run 2. But recall is still low even with the `top_k` cap effectively
removed (0.182 is well short of 1.0, and even short of Run 2's own
75/192≈0.39 theoretical ceiling under the old cap) — so retrieval quality
and the 0.4 threshold, not `top_k`, are now the dominant remaining limiter.
The per-question detail confirms this directly: several questions (q06,
q07, q09, q14) retrieved fewer than 25 chunks even though `top_k=25` was
requested, because fewer than 25 candidates cleared the 0.4 threshold —
`top_k` is no longer the constraint for those questions. Precision traded
off in the other direction, dropping from 0.197 to 0.115, since asking for
more candidates per question also pulls in more false positives. Fallback
accuracy is unchanged at 0.800 (`q20` is still the same, sole miss) — as
expected, since the fallback decision depends only on the single top-ranked
chunk's score, which `top_k` does not affect.

## Spot-check: computed answer keys verified by hand

Per `docs/tasks.md`'s Phase 5 checklist, 3 questions were checked against
the raw CSVs independently of `build_eval_answer_key.py`'s own code path:

- **q04** (female, born 1950s, atrial fibrillation, Warfarin) → patient
  `204`: `person.csv` confirms `gender_concept_id=2` (F), `year_of_birth=1950`;
  `condition_occurrence.csv` includes concept `7` (atrial fibrillation);
  `drug_exposure.csv` includes concept `12` (Warfarin). Correct.
- **q10** (male, born 1960s, ≥3 visits) → patient `71`: `person.csv`
  confirms `gender_concept_id=1` (M), `year_of_birth=1965`;
  `visit_occurrence.csv` has exactly 3 rows for this patient. Correct.
- **q13** (female, born 1970s, hypertension, latest SBP > 140) → patient
  `1550`: `person.csv` confirms `gender_concept_id=2` (F), `year_of_birth=1975`;
  `condition_occurrence.csv` includes concept `2` (hypertension);
  `measurement.csv`'s only SBP reading for this patient is 168 on
  2025-11-19. Correct.

All 3 matched their filters exactly. Combined with
`build_eval_answer_key.py`'s own assertion check (all 20 questions labeled
honestly — no answerable question computed an empty set, no unanswerable
question computed a non-empty one), this gives reasonable confidence the
ground truth is trustworthy.
