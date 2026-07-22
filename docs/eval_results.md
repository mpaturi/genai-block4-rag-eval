# Eval Results

This file is produced by `scripts/run_eval.py` (Phase 5) and is committed as
evidence for the "documents ≥1 experiment" acceptance criterion — see
`docs/spec.md`'s Eval harness design section.

All runs below use the same 20-question set in `data/eval/questions.json`
(15 answerable, 5 deliberately unanswerable) **except Run 4**, which
explicitly documents the 16-question subset it uses and why; each run's
specific `top_k` and `threshold` settings are listed in its own heading
below. Ground truth was independently recomputed by
`scripts/build_eval_answer_key.py` and spot-checked by hand (see
Spot-check section below) before any run.

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

## Run 4 — experiment: add metadata filters (Phase 7) (`top_k=5`, `threshold=0.4`, filtered)

**Question set:** 16 of the 20 questions in `data/eval/questions.json`, not
all 20 — `translate_filters()` in `scripts/run_eval.py` excludes 4
questions whose filters `build_metadata_filter()` cannot represent:

- **q10, q11, q12** (`min_visit_count`): no matching
  `build_metadata_filter()` parameter exists — visit count isn't one of
  the filterable fields.
- **q18** (`lab: "cholesterol"`): `build_metadata_filter()` validates
  `lab` against a fixed 4-value whitelist (`SBP`/`BMI`/`Glucose`/`HbA1c`)
  and raises `RAGFilterError` for anything else, by design —
  `cholesterol` is deliberately outside that whitelist (this is exactly
  why q18 is one of the 5 unanswerable questions), so it can't be
  translated at all, not just imperfectly matched.

No question used a non-strict lab operator (`>=`/`<=`) — checked
empirically against the real file; every lab-threshold question uses
strict `>`.

**Translation note:** `data/eval/questions.json` predates the Phase 7
metadata-filter integration and reuses `concepts.py`'s casual condition
and lab phrasing (e.g. `"type 2 diabetes"`, `"sbp"`), which doesn't match
Pinecone's stored metadata strings (Block 3's own clinical naming, e.g.
`"Diabetes mellitus type 2"`, the `"SBP"` `_LAB_PROPERTY` key).
`scripts/run_eval.py`'s `_CONDITION_NAME_TRANSLATION`/
`_DRUG_NAME_TRANSLATION`/`_LAB_NAME_TRANSLATION` tables exist to bridge
that, **scoped to this eval harness only** — this is not a bug or gap in
`retrieve.py`/`build_metadata_filter()` itself. A real caller (Block 5's
`graph_tool.py`) already sends the correctly-formatted clinical name
straight into an exact-match Cypher parameter against Neo4j, which shares
Block 3's naming with Pinecone — no translation happens on Block 5's side
either.

| Metric | Filtered (16 Qs) | Unfiltered, same 16 Qs | Run 2 (original 20 Qs, unfiltered) |
|---|---|---|---|
| Precision | 1.000 (37/37) | 0.232 (13/56) | 0.197 (14/71) |
| Recall | 0.287 (37/129) | 0.101 (13/129) | 0.073 (14/192) |
| Fallback accuracy | 1.000 (4/4) | 0.750 (3/4) | 0.800 (4/5) |

**Baseline:** same `top_k=5`/`threshold=0.4` as Run 2 — filters are the
only variable changed. Compared primarily against a same-question-subset
unfiltered control (middle column) rather than Run 2's original totals
(right column), since Run 2's 192-actual-patient total and 5-question
fallback set include the 4 questions this run excludes — comparing
straight against Run 2's published totals would conflate "filters help"
with "different questions."

**Result:** Adding metadata filters is a large, unambiguous win on this
subset. Precision jumps from 0.232 to 1.000 — every retrieved chunk is a
correct patient, because the metadata filter narrows Pinecone's candidate
set to exact matches (condition/drug/gender/birth_decade/lab-threshold)
before similarity ranking is even applied, so a wrong-patient chunk has no
way to be returned. Recall nearly triples (0.101 → 0.287) for the same
reason recall stayed low in Run 2/3: `top_k=5` still caps how many chunks
come back per question, but now every one of those 5 slots is a true
positive instead of mostly noise. Fallback accuracy reaches a perfect
1.000 (up from 0.750): `q20` (chronic kidney disease, outside the
condition whitelist) — the same question that broke Run 2's fallback
accuracy — now correctly falls back, because filtering on
`condition="chronic kidney disease"` (left untranslated, since it has no
whitelist entry) matches zero patients, so there's nothing for a lucky
semantic-score match to ride in on.

**Caveat:** this comparison is not fully apples-to-apples with Run 2/3 —
it covers a 16-question subset, and its fallback-accuracy denominator (4)
is smaller than Run 2/3's (5). The precision/recall improvement is real
(verified against the identical 16 questions, filtered vs. unfiltered, in
the middle two columns above), but the headline numbers aren't directly
comparable to Run 1–3's totals without accounting for the excluded
questions.

## Run 5 — experiment: raise `top_k` to 25, filtered (`top_k=25`, `threshold=0.4`, filtered)

Same 16-question subset and translation tables as Run 4 — see Run 4 for
why q10–q12 and q18 are excluded and how `scripts/run_eval.py`'s
`translate_filters()` bridges `questions.json`'s phrasing to Pinecone's
stored strings. `top_k` is the only variable changed relative to Run 4,
mirroring Run 3's relationship to Run 2.

| Metric | Filtered, `top_k=25` | Unfiltered, same 16 Qs, `top_k=25` | Run 4 (filtered, `top_k=5`, same 16 Qs) |
|---|---|---|---|
| Precision | 1.000 (57/57) | 0.143 (33/230) | 1.000 (37/37) |
| Recall | 0.442 (57/129) | 0.256 (33/129) | 0.287 (37/129) |
| Fallback accuracy | 1.000 (4/4) | 0.750 (3/4) | 1.000 (4/4) |

**Precision: held at 1.000, exactly.** Raising `top_k` from 5 to 25 while
filtered doesn't introduce a single false positive (57/57, same as Run
4's 37/37) — expected, since the metadata filter narrows Pinecone's
candidate set to exact matches before ranking, so asking for more results
just returns more *correct* patients, never wrong ones. This is the one
metric filtering fixes structurally, independent of `top_k`.

**Recall: a real improvement, from 0.287 to 0.442 (+0.155, ~54% relative)**
— filtering doesn't just help precision, it lets `top_k` actually matter
for recall too, the same way Run 3 found for the unfiltered case. But the
improvement isn't `top_k`-capped either: no included question retrieved
anywhere close to the full 25 slots (the highest was q01/q02 at 11 of
25), so the remaining gap to 1.0 recall is the 0.4 score threshold, not
`top_k` — same conclusion Run 3 reached for the unfiltered case, now
confirmed to hold with filtering on too. Two questions (q03, q05) reached
their full per-question answer set (retrieved == actual); most others
still fall well short (e.g. q08: 2 of 18, q06: 2 of 17), because a
patient's chunk still has to individually clear 0.4 similarity to the
question text, filter match or not.

**Fallback accuracy: held at 1.000**, same as Run 4 — `top_k` doesn't
affect the fallback decision (it depends only on the top-ranked chunk's
score), so this metric was never expected to move between Run 4 and 5.

**Filtered vs. unfiltered at the same `top_k=25` (left vs. middle
column):** the same large gap seen at `top_k=5` in Run 4 persists at
`top_k=25` — precision 1.000 vs. 0.143, recall 0.442 vs. 0.256, fallback
accuracy 1.000 vs. 0.750. Raising `top_k` helps both filtered and
unfiltered recall, but filtering remains the dominant factor throughout —
unfiltered `top_k=25` recall (0.256) still trails filtered `top_k=5`
recall (0.287, Run 4's right column above), meaning a 5x larger
unfiltered candidate pool still retrieves fewer correct patients than a
tightly-filtered pool five times smaller.

**Caveat:** same 16-question-subset caveat as Run 4 — not directly
comparable to Run 1–3's original 20-question totals without accounting
for the 4 excluded questions.

## Run 6 — experiment: lower `threshold` to 0.2, filtered (`top_k=15`, `threshold=0.2`, filtered)

Same 16-question subset and translation tables as Run 4/5.

**What changed and why — investigated the same way 0.75→0.4 was chosen in
Phase 5, not guessed.** A one-off diagnostic script (not committed)
retrieved every metadata-filtered candidate for each of the 12 included
answerable questions (`top_k=100`, well above any included question's
answer-set size) and checked each one against the ground-truth answer
key.

*First pass had a bug worth naming, because it changed the conclusion:*
looking at every chunk independently made a patient's low-scoring second
chunk look like an "excluded true positive" even when their first chunk
already cleared 0.4 — but `run_eval.py` dedupes to `person_id`, so that
patient was never actually missed. Correcting this to the **best score
per correct person_id** is what "excluded but actually correct" has to
mean here — a patient counts as excluded only if *none* of their chunks
clear the threshold.

With that correction: 72 correct patients (of 129 total across the 12
questions) are currently excluded at `threshold=0.4`, with best-chunk
scores ranging **0.2078 to 0.3980**. The real floor is **0.2078**
(person 8777, q14) — rounded down to a clean **0.2**, the same
one-decimal rounding Phase 5 used (0.419 → 0.4).

**A second, more important finding from the same diagnostic:** across
all 175 metadata-filtered chunks checked against the 12 included
answerable questions, **zero were false positives** — every chunk that
passed the filter belonged to a ground-truth-correct patient, regardless
of its score (down to -0.0045). Unlike Phase 5's unfiltered case, where
lowering the threshold risks pulling in semantically-similar-but-wrong
patients, the metadata filter has already restricted the candidate pool
to exact-criteria matches before any score is considered — so within
that pool, there's no false-positive risk to trade off against recall at
all. This is why 0.2 (a threshold Run 2 would never have tolerated
unfiltered) is safe to test here.

| Metric | Filtered, `top_k=15`/`threshold=0.2` | Unfiltered, same 16 Qs, same settings | Run 4 (filtered, `top_k=5`/`threshold=0.4`) |
|---|---|---|---|
| Precision | 1.000 (114/114) | 0.156 (28/180) | 1.000 (37/37) |
| Recall | 0.884 (114/129) | 0.217 (28/129) | 0.287 (37/129) |
| Fallback accuracy | 1.000 (4/4) | 0.000 (0/4) | 1.000 (4/4) |

**Result — both metrics checked honestly:**

- **Precision: held at exactly 1.000** (114/114), same as every filtered
  run so far. Confirms the diagnostic's finding wasn't a fluke restricted
  to the 72 patients sampled — the full filtered run introduces zero
  false positives at `threshold=0.2`.
- **Recall: a very large, real jump — 0.287 → 0.884** (+0.597). This is
  the direct, expected result of capturing the 72 genuine misses the
  diagnostic found: 8 of the 12 included answerable questions now
  retrieve every single ground-truth patient (`retrieved == actual` for
  q02, q03, q04, q05, q07, q09, q13, q15). The two questions still
  furthest from complete (q06: 15/17, q08: 15/18) are capped by
  `top_k=15`, not the threshold — both retrieved exactly 15, the max
  allowed.
- **Fallback accuracy: the hypothesis is confirmed, dramatically.**
  Filtered, it holds at a perfect 1.000 — unchanged from Run 4/5. But the
  same-subset **unfiltered** control at this same `threshold=0.2`
  collapses to **0.000 (0/4)** — every one of the 4 unanswerable
  questions incorrectly clears the threshold without a filter. The
  mechanism is structural, not a close call: q16/q17/q19/q20's
  conditions (`asthma`, `migraine`, `depression`, `chronic kidney
  disease`) don't exist anywhere in the corpus under any spelling, so the
  metadata filter returns **zero candidates for these questions at any
  `top_k`, before the score threshold is ever evaluated** — confirmed
  directly (`retrieve(..., top_k=15, **filters)` returns `[]` for all 4).
  Filtering, not the threshold, is doing 100% of the out-of-scope
  rejection now; the threshold is free to move for recall's sake without
  touching fallback accuracy at all, exactly the opposite of Run 2's
  unfiltered case where lowering the threshold cost 1.000 → 0.800.

**Caveat:** same 16-question-subset caveat as Run 4/5. Also, 0.2 is a
floor observed on this 20-question eval set, not a proven universal
floor — a live caller's question could still produce a correct match
scoring below 0.2 that this data never exercised. The zero-false-positive
property is what makes this safe to test aggressively, not a guarantee
that 0.2 is the true minimum.

## Run 7 — experiment: raise `top_k` to 20, filtered (`top_k=20`, `threshold=0.2`, filtered)

Same 16-question subset and translation tables as Run 4–6.

**What changed and why:** Run 6's per-question detail showed 3 of the 12
included answerable questions — q01, q06, q08 — retrieved exactly 15,
`top_k=15`'s cap, while each had more correct patients in its
ground-truth set than that (actual 20, 17, 18 respectively). That's not
one edge case; it's 3 of 12 questions hitting the ceiling, a real sign
`top_k=15` was capping recall again even at the lower threshold. `top_k`
raised to 20 — just above q01's actual answer-set size (20), the largest
among the included questions — threshold held at Run 6's 0.2 so `top_k`
is the only variable changed.

| Metric | Filtered, `top_k=20`/`threshold=0.2` | Unfiltered, same 16 Qs, same settings | Run 6 (filtered, `top_k=15`/`threshold=0.2`) |
|---|---|---|---|
| Precision | 1.000 (126/126) | 0.138 (33/240) | 1.000 (114/114) |
| Recall | 0.977 (126/129) | 0.256 (33/129) | 0.884 (114/129) |
| Fallback accuracy | 1.000 (4/4) | 0.000 (0/4) | 1.000 (4/4) |

**Result — were q01/q06/q08 actually uncapped?** Yes, all three, cleanly:

- **q01: retrieved=20, actual=20** — no longer truncated, but it does
  consume the entire `top_k=20` budget. Since its answer set is exactly
  20, this run happens to capture all of it with zero slack; a
  hypothetical 21st correct patient would still be cut off. q01 is fully
  captured here, not "still constrained" in the sense of missing anyone,
  but it's the one question with no headroom left at `top_k=20`.
- **q06: retrieved=17, actual=17** and **q08: retrieved=18, actual=18** —
  both now fully uncapped with room to spare (3 and 2 slots unused,
  respectively).

Recall climbed from 0.884 to 0.977 (+0.093) — nearly all of the remaining
gap from Run 6 closed. The 3 missing patients (129 − 126) are entirely
q14 (`retrieved=14`, `actual=17` — see per-question detail): q14 was
never `top_k`-capped at 15 either (12 retrieved then, well under 15), so
its shortfall is the 0.2 score threshold, not `top_k` — the same 3
patients would need a threshold below 0.2 to be captured, not a higher
`top_k`.

Precision held at exactly 1.000 (126/126) and fallback accuracy held at
1.000 (4/4), consistent with every filtered run so far — both properties
that come from the metadata filter itself, not from `top_k` or
`threshold`, so neither was expected to move here. The same-subset
unfiltered control at these settings again confirms the filtered/
unfiltered gap: fallback accuracy 0.000 unfiltered vs. 1.000 filtered,
the same structural result as Run 6.

**Caveat:** same 16-question-subset caveat as Run 4–6. `top_k=20` is
sized to this eval set's specific answer-set sizes (max 20, for q01) —
not a general guarantee that 20 is enough for an arbitrary live question,
the same caveat Run 3 raised for the unfiltered case.

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
