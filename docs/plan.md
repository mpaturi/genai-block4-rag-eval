# Block 4: RAG Service Implementation Plan

**Goal:** Chunk and load Block 3's patient corpus into Pinecone, answer
natural-language questions with cited, grounded answers via Claude, expose
it as an API, and prove it works with a reproducible eval harness.

**Architecture:** A one-time ingestion pipeline (chunk → upload to
Pinecone) feeds a query-time pipeline (retrieve → generate) wrapped in a
FastAPI service. An eval harness calls retrieval directly (never Claude) to
score precision, recall, and fallback accuracy against a ground-truth
answer key computed from Block 1's OMOP CSVs. Full reasoning for every
decision below lives in `docs/spec.md` — this file is the "what to build
and in what order," not the "why."

**Tech Stack:** Python 3.11, pinecone, anthropic, fastapi, uvicorn, pandas,
python-dotenv, orjson, pytest — exact versions pinned once installed
(Task 1).

## Global Constraints

- Python 3.11
- All scripts load credentials via `python-dotenv` from `.env`
- `.env` is git-ignored; `.env.example` is committed
- `requirements.txt` uses `==` pins, never `>=`
- No cloud service except Pinecone and the Claude API — no AWS, no Neo4j
- `ingest.py` always deletes and rebuilds the whole Pinecone namespace —
  never assumes partial state
- `run_eval.py` calls `retrieve.py` directly — it never calls `generate.py`
  or Claude

## File map

| File | Task | Purpose |
|---|---|---|
| `requirements.txt`, `.env.example`, `.gitignore` | 1 | Environment setup |
| `scripts/check_connection.py` | 2 | Smoke test both API keys |
| `scripts/create_index.py` | 3 | Create the Pinecone index (idempotent) |
| `scripts/chunk_records.py`, `tests/test_chunking.py` | 4 | Split long patient text into chunks |
| `scripts/ingest.py` | 5 | Upload chunks to Pinecone (delete + reload) |
| `scripts/retrieve.py`, `tests/test_retrieve.py` | 6 | Job 1 — search Pinecone |
| `scripts/generate.py`, `scripts/api.py` | 7 | Job 2 + FastAPI `POST /query` |
| `data/raw/*.csv`, `scripts/build_eval_answer_key.py`, `data/eval/questions.json` | 8 | Eval ground truth |
| `scripts/run_eval.py`, `docs/eval_results.md` | 9 | Eval scoring + experiment |
| `scripts/run_all.py`, `scripts/verify.py`, `README.md` | 10 | One-command setup + verification |

---

### Task 1: Environment setup

**Files:** `requirements.txt`, `.env.example`, `.gitignore`

**Interfaces:**
- Produces: a Python environment with all dependencies installed, and a
  `.env` template with `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
  `ANTHROPIC_API_KEY`

**Key decisions:**
- `requirements.txt` pins exact versions, filled in after `pip install`
- `.gitignore` (Python template + `.env`) already exists from repo creation
  — just confirm `.env` is actually covered

- [ ] Create `requirements.txt`, `.env.example`
- [ ] `cp .env.example .env` and fill in real keys once you have them
- [ ] `pip install -r requirements.txt`, then freeze exact versions back
      into `requirements.txt`
- [ ] Commit: `feat(setup): add Python deps and env template`

---

### Task 2: Connection check script

**Files:** `scripts/check_connection.py`

**Interfaces:**
- Consumes: `.env` credentials
- Produces: exits 0 and prints confirmation for both Pinecone and Claude on
  success; exits 1 with a clear message on failure

**Key decisions:**
- Checks both `PINECONE_API_KEY` and `ANTHROPIC_API_KEY` — Block 3's
  version only had one service to check, this one has two
- Catches auth errors separately from network/timeout errors for a useful
  message

- [ ] Create `scripts/check_connection.py`
- [ ] Run — confirm both keys report OK
- [ ] Commit: `feat(setup): add Pinecone and Claude connection smoke test`

---

### Task 3: Pinecone index creation

**Files:** `scripts/create_index.py`

**Interfaces:**
- Consumes: `.env` credentials
- Produces: a serverless Pinecone index named `PINECONE_INDEX_NAME`, using
  integrated inference with `llama-text-embed-v2` and `field_map:
  {"text": "chunk_text"}`

**Key decisions:**
- Checks whether the index already exists before creating — safe to re-run
- **Prints the metric name the index was created with** (cosine/dotproduct
  = higher-is-better, euclidean = lower-is-better) — needs no data, so it
  runs in Phase 2. The empirical self-match check (Task 5) confirms the
  direction for real once chunks exist

- [ ] Create `scripts/create_index.py`
- [ ] Run — confirm index created, note the printed scoring metric
- [ ] Re-run — confirm it does not error or create a duplicate index
- [ ] Commit: `feat(setup): add idempotent Pinecone index creation`

---

### Task 4: Chunking

**Files:** `scripts/chunk_records.py`, `tests/test_chunking.py`,
`data/raw/graph_export.jsonl` (copied from Block 3)

**Interfaces:**
- Consumes: `data/raw/graph_export.jsonl`
- Produces: a list of chunk records (`_id`, `chunk_text`, `person_id`, and
  the rest of the patient metadata, plus `chunk_index`)

**Key decisions:**
- 200-character threshold, whole sentences packed in one at a time (see
  spec's Chunking design for the reasoning)
- Oversized single-sentence fallback, tested with synthetic data since
  real data won't trigger it
- `tests/test_chunking.py` covers: short text (1 chunk), text needing a
  split (2+ chunks), the oversized-sentence fallback, and empty text

- [ ] Copy `graph_export.jsonl` from `genai-block3-graph-kb/data/export/`
      into `data/raw/`
- [ ] Compute the `text` field's character-length distribution against the
      copied file; confirm 200 chars still gives "most patients single-
      chunk, high-burden patients split" (see spec's Chunking design)
- [ ] Create `scripts/chunk_records.py`
- [ ] Create `tests/test_chunking.py`
- [ ] `pytest tests/test_chunking.py` — all pass
- [ ] During Phase 3 ingestion, test the empty-text chunk path against real
      Pinecone — confirm whether integrated inference accepts an empty
      `chunk_text` string; if rejected, adjust `chunk_records.py` to
      substitute a minimal placeholder instead (see spec's Chunking design)
- [ ] Commit: `feat(ingest): add chunking logic and unit tests`

---

### Task 5: Ingestion

**Files:** `scripts/ingest.py`

**Interfaces:**
- Consumes: chunk records from `chunk_records.py`, the index from
  `create_index.py`
- Produces: every chunk upserted into the `patients` namespace

**Key decisions:**
- Deletes the whole `patients` namespace first, then upserts the full
  current chunk set — true idempotency, not just "same ID overwrites" (see
  spec)
- Treats a missing namespace (the very first run, nothing to delete yet)
  as already-empty, not an error
- Upserts in batches, not one call — ~15,000 chunks exceeds the
  integrated-inference upsert's per-call record/payload limit
- Fails with a clear message if the index doesn't exist yet, rather than
  creating one with guessed settings
- Allows a brief wait after upserting before treating a count check as
  final (Pinecone writes are not instantly queryable)

- [ ] Create `scripts/ingest.py`
- [ ] Run — confirm all batches upload and vector count roughly matches
      patient count
- [ ] Re-run — confirm the count is identical (idempotency, not assumed)
- [ ] Run one self-match query (search using a known chunk's own text,
      confirm it returns itself as top match) — confirms score direction
      empirically, per spec's Retrieval design
- [ ] Commit: `feat(ingest): add idempotent ingestion into Pinecone`

---

### Task 6: Retrieval (Job 1)

**Files:** `scripts/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: a question string, optional `top_k` (default 5)
- Produces: ranked chunks with scores, plus a boolean "did this clear the
  threshold" decision

**Key decisions:**
- Score threshold starts at 0.75 — confirmed against the real scoring
  direction printed by `create_index.py` in Task 3, not assumed
- The threshold comparison is pure logic, unit-tested with fake scores —
  no live Pinecone call needed for that part

- [ ] Create `scripts/retrieve.py`
- [ ] Create `tests/test_retrieve.py`
- [ ] `pytest tests/test_retrieve.py` — all pass
- [ ] Run a real query by hand — confirm results look sensible
- [ ] Commit: `feat(query): add retrieval with score-threshold logic`

---

### Task 7: Generation (Job 2) and API

**Files:** `scripts/generate.py`, `scripts/api.py`

**Interfaces:**
- Consumes: retrieval results
- Produces: `POST /query` — cited answer, "I don't know," or a clear 502
  error, per the three response shapes in the spec

**Key decisions:**
- Claude is only called when the best retrieval score clears the
  threshold, and only receives the chunks that individually clear it too
  — not every chunk in `top_k` — otherwise a fixed "I don't know" string,
  no API call
- `top_k` validated to 1–20 (default 5); `question` rejected if empty or
  whitespace-only — both return HTTP 422 before Pinecone is called
- Claude is instructed to cite `person_id` per claim (no automated check
  that it actually does — documented limitation)
- Pinecone/Claude failures caught and returned as one generic 502, never a
  raw stack trace

- [ ] Create `scripts/generate.py`
- [ ] Create `scripts/api.py`
- [ ] Run the API locally, test all 3 response shapes by hand (a match, no
      match, and a simulated failure — e.g. a temporarily invalid API key)
- [ ] Commit: `feat(query): add generation and FastAPI endpoint`

---

### Task 8: Eval ground truth

**Files:** `data/raw/condition_occurrence.csv`, `drug_exposure.csv`,
`person.csv`, `measurement.csv`, `scripts/build_eval_answer_key.py`,
`data/eval/questions.json`

**Interfaces:**
- Consumes: Block 1's OMOP CSVs
- Produces: `data/eval/questions.json` (≥20 questions) with a
  programmatically computed correct-patient-ID set per question

**Key decisions:**
- Ground truth is computed with pandas, not hand-typed: condition/drug/
  demographic/burden questions reuse Block 3's Cypher-query logic;
  lab-threshold questions are computed straight from `measurement.csv`
  (Block 3 never queried labs — see spec's Relationship to Block 3)
- Question mix: co-occurrence, demographic-filtered, high-burden/visit,
  lab-threshold, and deliberately unanswerable questions to test "I don't
  know" (at least 5 of the 20, per spec — fewer would make fallback
  accuracy meaningless)
- `build_eval_answer_key.py` asserts the labeling is honest: answerable
  questions get a non-empty computed set, unanswerable ones get an empty
  set — fails loudly on a mismatch instead of relying only on spot-checks

- [ ] Copy the 4 OMOP CSVs from Block 1 into `data/raw/`
- [ ] Create `scripts/build_eval_answer_key.py`
- [ ] Write `data/eval/questions.json` (≥20 questions)
- [ ] Run the answer-key builder's assertion check — confirm no question
      is mislabeled
- [ ] Spot-check a few answer keys by hand against the CSVs
- [ ] Commit: `feat(eval): add eval questions and ground-truth builder`

---

### Task 9: Eval runner and experiment

**Files:** `scripts/run_eval.py`, `docs/eval_results.md`

**Interfaces:**
- Consumes: `questions.json`, the answer key, `retrieve.py`
- Produces: `docs/eval_results.md` with precision/recall, fallback
  accuracy, and the required parameter experiment

**Key decisions:**
- Calls `retrieve.py` only — never `generate.py` or Claude (see spec's
  Reproducibility section for why)
- Two separate metrics reported, never blended into one number
- Retrieved chunks deduped to unique `person_id`s before scoring; precision
  is 0 (not a divide-by-zero error) when nothing was retrieved (only
  threshold-clearing chunks count as "retrieved," matching Job 2's filter
  in Task 7 — not the raw top_k list)
- Precision/recall are micro-averaged across all answerable questions
  (summed correct ÷ summed retrieved/actually-correct), not averaged
  per-question
- Run only after confirming the index is fully loaded — Task 5's idempotent
  re-run and Task 6's manual query test already establish this; `verify.py`
  (Task 10) doesn't exist yet at this point and isn't a precondition here

- [ ] Create `scripts/run_eval.py`
- [ ] Run once at the default settings — record the score
- [ ] Change one setting (`top_k` or the threshold) and run again
- [ ] Write `docs/eval_results.md` — both runs, both metrics, a short note
      on what changed and why
- [ ] Commit: `feat(eval): add eval runner and document parameter experiment`

---

### Task 10: Orchestrator, verification, README

**Files:** `scripts/run_all.py`, `scripts/verify.py`, `README.md`

**Interfaces:**
- Consumes: everything above
- Produces: one-command setup (`run_all.py`), a pass/fail check script
  (`verify.py`), and full setup docs

**Key decisions:**
- `run_all.py` runs check_connection → create_index → chunk+ingest →
  verify, in that order — does not include `run_eval.py`
- `verify.py` checks: credentials valid, index reachable, chunk count
  matches what chunking should produce, re-running ingestion gives an
  identical count, a sample query returns results, and
  `docs/eval_results.md` exists
- Polls with a bounded retry loop for eventual consistency, not a single
  fixed sleep (see spec's Reproducibility section)

- [ ] Create `scripts/verify.py`
- [ ] Create `scripts/run_all.py`
- [ ] Run `python scripts/verify.py` — all checks PASS
- [ ] Run `python scripts/run_all.py` — completes end-to-end
- [ ] Write `README.md` (setup, architecture, AI-assisted workflow note)
- [ ] Commit: `feat(verify): add orchestrator, verification, and README`
