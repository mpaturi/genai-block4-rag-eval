# Block 4 Tasks

Branching rule (see `docs/spec.md` Phases section): each phase gets its own
branch and its own PR (6 PRs total). `phase-1-spec` branches from `main`.
Every phase after that branches from the **tip of the previous phase's
branch**, not `main` — because the previous phase's PR may not be merged
yet when the next phase starts. If the previous PR *has* already merged
into `main` by the time you start the next phase, branch from `main`
instead and skip the stacked-base step below.

Each phase's steps below follow this shape:
```
git checkout <previous-phase-branch>
git pull                                   # only if that branch is remote-tracked
git checkout -b <this-phase-branch>
# ...do the work, commit as you go...
git push -u origin <this-phase-branch>
gh pr create --base <previous-phase-branch> --title "..." # stacked PR
```
When you open the PR, set `--base` to the previous phase's branch if it's
still unmerged (stacked PR), or `main` if it already merged.

## Phase 1 — Spec (`phase-1-spec`, base: `main`)

- [ ] `git checkout main && git checkout -b phase-1-spec`
- [ ] Write `docs/spec.md`
- [ ] Write `docs/plan.md`
- [ ] Write `docs/tasks.md`
- [ ] Commit `docs/spec.md`, `docs/plan.md`, `docs/tasks.md`
- [ ] `git push -u origin phase-1-spec`
- [ ] Open PR1: base `main` (spec review)

## Phase 2 — Setup (`phase-2-setup`, base: `phase-1-spec`)

- [x] `git checkout phase-1-spec && git checkout -b phase-2-setup`
- [x] Create `requirements.txt`, `.env.example`
- [x] Confirm `.gitignore` covers `.env`
- [x] `pip install -r requirements.txt`, then pin exact versions
- [x] Create `scripts/check_connection.py`
- [x] Run connection smoke test — confirm both Pinecone and Claude keys OK
- [x] Create `scripts/create_index.py`
- [x] Run — confirm index created, note the printed scoring metric (cosine — higher is more relevant)
- [x] Re-run — confirm no error, no duplicate index
- [x] Commit Phase 2 files
- [ ] `git push -u origin phase-2-setup`
- [ ] Open PR2: base `phase-1-spec` (or `main` if PR1 already merged)

## Phase 3 — Ingest (`phase-3-ingest`, base: `phase-2-setup`)

- [x] `git checkout phase-2-setup && git checkout -b phase-3-ingest`
- [x] Copy `graph_export.jsonl` from Block 3's `data/export/` into `data/raw/`
- [x] Compute `text` length distribution against the real file; confirm the
      200-char threshold still fits (see spec's Chunking design)
- [x] Create `tests/test_chunking.py` — cover short text, a split case, the
      oversized-sentence fallback, and empty text
- [x] Create `scripts/chunk_records.py`
- [x] `pytest tests/test_chunking.py` — all pass
- [x] Create `scripts/ingest.py`
- [x] Run — confirm vector count roughly matches patient count
- [x] Re-run — confirm identical count (idempotency, verified not assumed)
- [x] Run one self-match query — confirm a chunk's own text returns itself
      as top match (confirms score direction empirically)
- [x] Test the empty-text chunk path against real Pinecone — confirm
      accept/reject behavior for an empty `chunk_text` string; if rejected,
      adjust `chunk_records.py`'s fallback (see spec's Chunking design)
- [x] Commit `scripts/chunk_records.py`, `scripts/ingest.py`,
      `tests/test_chunking.py`, `data/raw/graph_export.jsonl`
- [x] `git push -u origin phase-3-ingest`
- [x] Open PR3: base `phase-2-setup` (or `main` if PR2 already merged)

## Phase 4 — Retrieve + Generate (`phase-4-retrieve-generate`, base: `phase-3-ingest`)

- [ ] `git checkout phase-3-ingest && git checkout -b phase-4-retrieve-generate`
- [ ] Create `tests/test_retrieve.py` — cover above/below/exactly-at threshold
- [ ] Create `scripts/retrieve.py`
- [ ] `pytest tests/test_retrieve.py` — all pass
- [ ] Run a real query by hand — confirm results look sensible
- [ ] Create `scripts/generate.py`
- [ ] Create `scripts/api.py`
- [ ] Test all 3 response shapes by hand: a real match, no match, a
      simulated upstream failure (temporarily invalid API key)
- [ ] Test 422 validation by hand: an empty/whitespace-only `question`,
      and an out-of-range `top_k` (e.g. 0 or 21)
- [ ] Commit `scripts/retrieve.py`, `scripts/generate.py`, `scripts/api.py`,
      `tests/test_retrieve.py`
- [ ] `git push -u origin phase-4-retrieve-generate`
- [ ] Open PR4: base `phase-3-ingest` (or `main` if PR3 already merged)

## Phase 5 — Eval (`phase-5-eval`, base: `phase-4-retrieve-generate`)

- [ ] `git checkout phase-4-retrieve-generate && git checkout -b phase-5-eval`
- [ ] Copy `condition_occurrence.csv`, `drug_exposure.csv`, `person.csv`,
      `measurement.csv` from Block 1 into `data/raw/`
- [ ] Create `scripts/build_eval_answer_key.py`
- [ ] Write `data/eval/questions.json` (≥20 questions — co-occurrence,
      demographic, high-burden/visit, lab-threshold, and ≥5 deliberately
      unanswerable)
- [ ] Run the answer-key builder's assertion check — confirm no question
      is mislabeled (answerable has matches, unanswerable has none)
- [ ] Spot-check a few computed answer keys by hand against the CSVs
- [ ] Create `scripts/run_eval.py` (calls `retrieve.py` only, never Claude)
- [ ] Run once at default settings — record precision/recall/fallback accuracy
- [ ] Change one setting (`top_k` or the threshold), run again
- [ ] Write `docs/eval_results.md` — both runs, both metrics, a short note
      on what changed and why
- [ ] Commit `data/raw/*.csv`, `scripts/build_eval_answer_key.py`,
      `data/eval/questions.json`, `scripts/run_eval.py`, `docs/eval_results.md`
- [ ] `git push -u origin phase-5-eval`
- [ ] Open PR5: base `phase-4-retrieve-generate` (or `main` if PR4 already merged)

## Phase 6 — Verify + Docs (`phase-6-verify-docs`, base: `phase-5-eval`)

- [ ] `git checkout phase-5-eval && git checkout -b phase-6-verify-docs`
- [ ] Create `scripts/verify.py`
- [ ] Create `scripts/run_all.py` (check_connection → create_index →
      chunk+ingest → verify — does not include `run_eval.py`)
- [ ] Run `python scripts/verify.py` — all checks PASS
- [ ] Run `python scripts/run_all.py` — completes end-to-end
- [ ] Write `README.md` (setup, architecture, AI-assisted workflow note)
- [ ] Commit `scripts/verify.py`, `scripts/run_all.py`, `README.md`
- [ ] `git push -u origin phase-6-verify-docs`
- [ ] Open PR6: base `phase-5-eval` (or `main` if PR5 already merged) —
      ready for mentor review
