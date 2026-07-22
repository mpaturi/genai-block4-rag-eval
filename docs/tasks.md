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

- [x] `git checkout main && git checkout -b phase-1-spec`
- [x] Write `docs/spec.md`
- [x] Write `docs/plan.md`
- [x] Write `docs/tasks.md`
- [x] Commit `docs/spec.md`, `docs/plan.md`, `docs/tasks.md`
- [x] `git push -u origin phase-1-spec`
- [x] Open PR1: base `main` (spec review)

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
- [x] `git push -u origin phase-2-setup`
- [x] Open PR2: base `phase-1-spec` (or `main` if PR1 already merged)

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

- [x] `git checkout phase-3-ingest && git checkout -b phase-4-retrieve-generate`
- [x] Create `tests/test_retrieve.py` — cover above/below/exactly-at threshold
- [x] Create `scripts/retrieve.py`
- [x] `pytest tests/test_retrieve.py` — all pass
- [x] Run a real query by hand — confirm results look sensible
- [x] Create `scripts/generate.py`
- [x] Create `scripts/api.py`
- [x] Test all 3 response shapes by hand: a real match, no match, a
      simulated upstream failure (temporarily invalid API key)
- [x] Test 422 validation by hand: an empty/whitespace-only `question`,
      and an out-of-range `top_k` (e.g. 0 or 21)
- [x] Commit `scripts/retrieve.py`, `scripts/generate.py`, `scripts/api.py`,
      `tests/test_retrieve.py`
- [x] `git push -u origin phase-4-retrieve-generate`
- [x] Open PR4: base `phase-3-ingest` (or `main` if PR3 already merged)

## Phase 5 — Eval (`phase-5-eval`, base: `phase-4-retrieve-generate`)

- [x] `git checkout phase-4-retrieve-generate && git checkout -b phase-5-eval`
- [x] Copy `condition_occurrence.csv`, `drug_exposure.csv`, `person.csv`,
      `measurement.csv`, `visit_occurrence.csv` from Block 1 into `data/raw/`
- [x] Copy `scripts/concepts.py` from Block 1
- [x] Create `scripts/build_eval_answer_key.py`
- [x] Write `data/eval/questions.json` (≥20 questions — co-occurrence,
      demographic, high-burden/visit, lab-threshold, and ≥5 deliberately
      unanswerable)
- [x] Run the answer-key builder's assertion check — confirm no question
      is mislabeled (answerable has matches, unanswerable has none)
- [x] Spot-check a few computed answer keys by hand against the CSVs
- [x] Create `scripts/run_eval.py` (calls `retrieve.py` only, never Claude)
- [x] Run once at default settings — record precision/recall/fallback accuracy
- [x] Change one setting (`top_k` or the threshold), run again
- [x] Write `docs/eval_results.md` — both runs, both metrics, a short note
      on what changed and why
- [x] Commit `data/raw/*.csv`, `scripts/concepts.py`,
      `scripts/build_eval_answer_key.py`, `data/eval/questions.json`,
      `scripts/run_eval.py`, `docs/eval_results.md`
- [x] `git push -u origin phase-5-eval`
- [x] Open PR5: base `phase-4-retrieve-generate` (or `main` if PR4 already merged)

## Phase 6 — Verify + Docs (`phase-6-verify-docs`, base: `phase-5-eval`)

- [x] `git checkout phase-5-eval && git checkout -b phase-6-verify-docs`
- [x] Create `scripts/verify.py`
- [x] Create `scripts/run_all.py` (check_connection → create_index →
      chunk+ingest → verify — does not include `run_eval.py`)
- [x] Run `python scripts/verify.py` — all checks PASS
- [x] Run `python scripts/run_all.py` — completes end-to-end
- [x] Write `README.md` (setup, architecture, AI-assisted workflow note)
- [x] Commit `scripts/verify.py`, `scripts/run_all.py`, `README.md`
- [x] `git push -u origin phase-6-verify-docs`
- [x] Open PR6: base `phase-5-eval` (or `main` if PR5 already merged) —
      ready for mentor review

## Phase 7 — Metadata filter (`phase-7-metadata-filter`, base: `phase-6-verify-docs`)

- [x] `git checkout phase-6-verify-docs && git checkout -b phase-7-metadata-filter`
- [x] Verify against the installed `pinecone` client (not docs samples):
      `filter` is a flat kwarg on `index.search()`, multiple filter keys
      AND-combine, and equality on a list-of-strings field matches
      "contains" — confirmed with live calls against the real index
- [x] Add `_LAB_PROPERTY`/`_COMPARISON_OP`/`_GENDER_VALUE` and
      `build_metadata_filter()` to `scripts/retrieve.py`, naming matched
      exactly to Block 5's `graph_tool.py`
- [x] Extend `retrieve()`'s signature with the optional filter fields;
      unfiltered calls must behave identically to before this phase
- [x] Add unit tests for `build_metadata_filter()` to
      `tests/test_retrieve.py` (pure logic, no live Pinecone call)
- [x] `pytest tests/test_retrieve.py` — all pass
- [x] Add the same optional fields to `scripts/api.py`'s `QueryRequest`,
      with the lab/comparison/value all-or-nothing validator
- [x] Run the full test suite — confirm nothing else broke
- [x] Commit `scripts/retrieve.py`, `scripts/api.py`, `tests/test_retrieve.py`
- [x] Fix `scripts/api.py` to log `/query` failures server-side
      (`logger.exception`) without leaking tracebacks into the response
- [x] Replace `retrieve.py`'s per-call Pinecone client construction with
      a module-level lazy singleton (`_get_index()`)
- [x] Restore the explicit `index_name not in pc.list_indexes().names()`
      check in `scripts/ingest.py`, matching `create_index.py`'s pattern
- [x] Treat whitespace-only text as empty in `chunk_records.py`'s
      `chunk_text()` (`not text.strip()`, not `text == ""`)
- [x] Fix `scripts/verify.py`'s `main()` to fail cleanly (`[FAIL]`, exit
      1) instead of crashing on a nonexistent index name — same fix
      applied independently on both `phase-6-verify-docs` and this branch
- [x] Add `test_build_metadata_filter_unrecognized_gender_raises_before_pinecone`
      to `tests/test_retrieve.py`
- [x] Merge `phase-6-verify-docs` into `phase-7-metadata-filter` to bring
      in the four regression fixes and the `verify.py` fix phase-6 got
      independently — resolved conflicts by reading the merged files, not
      trusting the merge blindly
- [x] Add `run_eval.py --filtered` mode with a translation table bridging
      `questions.json`'s casual phrasing to Pinecone's stored strings
- [x] Document Run 4 (filtered eval, `top_k=5`/`threshold=0.4`) in
      `docs/eval_results.md` — precision/recall/fallback accuracy vs. the
      same 16-question subset unfiltered
- [x] Document Run 5 (filtered, `top_k=25`) — confirms recall's
      remaining gap is the score threshold, not `top_k`, once filtered
- [x] Investigate the real score floor among genuinely-missed correct
      patients (corrected per-patient methodology, not per-chunk) and
      document Run 6 (filtered, `threshold=0.2`) — recall 0.287→0.884,
      fallback accuracy holds at 1.000 filtered vs. collapses to 0.000
      unfiltered at the same threshold
- [x] Document Run 7 (filtered, `top_k=20`) — closes the `top_k=15` cap
      found in Run 6's per-question detail (q01/q06/q08)
- [x] Raise `retrieve.py`'s `DEFAULT_TOP_K` and `api.py`'s
      `QueryRequest.top_k` default from 5 to 15, backed by Run 5's
      observed per-question ceiling (11), not a round guess
- [x] `git push -u origin phase-7-metadata-filter`
- [x] Open PR7: base `phase-6-verify-docs` (or `main` if PR6 already
      merged) — opened by the user

**PR7 review fixup (Leone):** the live API always used
`DEFAULT_THRESHOLD` regardless of which filter was active, so a
condition/drug filter never got the permissive threshold Run 6 showed
was safe specifically for those two filter types.

- [x] Add `PERMISSIVE_THRESHOLD` and `select_threshold()` to
      `scripts/retrieve.py` — permissive only for `condition`/`drug`
      filters, never `gender`/`lab`/`birth_decade`, since only
      condition/drug can structurally return zero candidates for an
      untracked topic
- [x] Wire `select_threshold()` into `scripts/api.py`'s `/query` handler
      — both the fallback check and the `relevant_chunks` filter must use
      the same computed threshold
- [x] Add unit tests for `select_threshold()` to `tests/test_retrieve.py`
- [x] Add `tests/test_api.py` — integration-style test proving a
      gender-only filter does *not* go permissive end-to-end (the one
      thing `select_threshold()`'s own unit tests can't prove by
      themselves)
- [x] Update `docs/spec.md`'s Known limitations and Retrieval design
      sections to describe `select_threshold()` and cite Run 6
- [x] Run the full test suite — confirm nothing else broke
- [x] Re-run `python scripts/run_eval.py --filtered --threshold 0.2` as a
      regression sanity check against the already-documented Run 6
      numbers (this run itself is unaffected — `run_eval.py` isn't wired
      to `select_threshold()`, only `api.py` is)
- [x] Manually hit `POST /query` once with a condition filter and an
      out-of-scope-sounding question (goes permissive), and once with
      only a gender filter and the same question (stays at
      `DEFAULT_THRESHOLD`, falls back)
- [x] Commit referencing the review feedback, push to
      `phase-7-metadata-filter` — no new PR, lands as additional commits
      on PR7
