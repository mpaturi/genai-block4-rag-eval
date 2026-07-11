# Block 4 Specification

## Project title

RAG Service with Retrieval Evaluation — Pinecone-backed Clinical Q&A

## Acceptance criteria

> **Project — RAG service with a real eval harness**
> Ingests a real corpus (reuse Block 3's domain), chunks + embeds, stores in
> a vector DB, answers questions with cited sources; runs as an API; returns
> "I don't know" when retrieval is empty; spec written and committed first.
> Eval harness measures retrieval precision/recall (or context relevance) on
> a labelled set of ≥20 questions, produces a reproducible score, and
> documents ≥1 experiment where a parameter change moved the metric.

## Goal

Build a RAG service on top of Block 3's patient corpus so that:
- long patient summaries are split into focused chunks before storage
- chunks are embedded and stored in Pinecone (integrated inference — Pinecone
  embeds the text server-side; we never call a separate embedding API)
- natural-language clinical questions get answered by Claude, grounded only
  in the chunks retrieval actually found, with patient sources cited
- the system says "I don't know" instead of guessing when nothing relevant
  is found
- the service runs as an HTTP API
- an eval harness proves retrieval quality with a reproducible precision/
  recall score against a known-correct answer key, and documents at least
  one experiment where changing a setting moved that score

## Problem statement

Block 3 built a graph database that can answer relationship questions
precisely — but only via Cypher, a query language most people don't know.
A person who wants to ask "which patients have diabetes and take metformin?"
in plain English has no way in. RAG closes that gap: it finds the patient
records that are semantically relevant to a question, then has an LLM turn
those records into a plain-English, cited answer — without inventing facts,
because the LLM is only shown what retrieval actually found.

## Relationship to Block 3

Data source: `data/export/graph_export.jsonl` from `genai-block3-graph-kb`
is copied into this repo's `data/raw/` (same pattern Block 3 used for
Block 1's CSVs — copy source artifacts into the repo, don't reach across
repos at runtime).

**Also copied in:** `condition_occurrence.csv`, `drug_exposure.csv`, and
`person.csv` (the same Block 1 OMOP files Block 3 used). Reason: Block 3's
`graph_export.jsonl` metadata only stores *counts* (`condition_count`,
`drug_count`), not the actual condition/drug names as structured, filterable
fields — those names only exist inside the free-text `text` field. To build
a trustworthy eval answer key ("which patients have condition X and drug Y")
we need to recompute that join directly from the OMOP CSVs with pandas, the
same way Block 3's Cypher queries did — not by string-matching the `text`
field, which would be fragile and would defeat the purpose of having an
independent answer key.

Block 3 artifacts reused:
- `graph_export.jsonl` (11,424 patient records, `text` + `metadata` fields)
- `condition_occurrence.csv`, `drug_exposure.csv`, `person.csv` (for eval
  ground-truth computation only — never touched by the live RAG pipeline)
- The 4 Cypher queries' *logic* (reimplemented as pandas filters against the
  CSVs, since this repo has no Neo4j connection)

Block 3 artifacts NOT reused:
- Neo4j itself — this repo never connects to a graph database at runtime
- Docker — no local database this block; Pinecone is a managed cloud service

## Architecture

```
Setup (Phase 2, run once before ingestion):

scripts/check_connection.py   (smoke test — confirms PINECONE_API_KEY and
                               ANTHROPIC_API_KEY are valid before anything
                               else runs; same role as Block 3's
                               check_connection.py)
scripts/create_index.py       (creates the Pinecone serverless index if it
                               does not already exist — idempotent; checks
                               for the index name first, does not error on
                               a second run)


Ingestion pipeline (run once, offline, before any questions are asked):

data/raw/graph_export.jsonl (11,424 patient records)
       |
       v
scripts/chunk_records.py     (split long `text` fields into <=150-char
                               pieces on sentence boundaries; short records
                               pass through as a single chunk)
       |
       v
scripts/ingest.py             (upload chunks to the already-created Pinecone
                               index via upsert_records; Pinecone embeds
                               each chunk's text server-side; fails with a
                               clear message if the index from
                               create_index.py does not exist yet)
       |
       v
Pinecone index (integrated inference, cloud-hosted)


Query-time pipeline (runs once per question):

question (from API caller, or from the eval harness)
       |
       v
scripts/retrieve.py           (Job 1 — search Pinecone, get ranked chunks
                               with similarity scores)
       |
       v
   score check: best match >= threshold?
       |                              |
      yes                             no
       |                              |
       v                              v
scripts/generate.py            fixed "I don't know" response
(Job 2 — Claude API writes            (Claude API is not called —
 a cited answer from the               saves cost, avoids the model
 retrieved chunks)                     guessing on empty context)
       |                              |
       v                              v
              scripts/api.py (FastAPI — wraps both steps behind POST /query;
                              catches Pinecone/Claude failures and returns
                              a clear error response instead of crashing)


Eval harness (run on demand, not part of the live API):

data/raw/*.csv (Block 1 OMOP source) --> scripts/build_eval_answer_key.py
       |                                        |
       v                                        v
data/eval/questions.json (>=20 Qs)  +  ground-truth patient ID sets
       |                                        |
       +--------------------+-------------------+
                             v
                    scripts/run_eval.py
                    (runs each question through retrieve.py ONLY — not the
                     full pipeline — compares retrieved patient IDs and
                     the threshold pass/fail decision to the answer key,
                     computes precision/recall and fallback accuracy,
                     saves a score report. Claude is never called during
                     eval: see Eval harness design for why.)


Unit tests (alongside the deterministic code, not part of the eval harness):

tests/test_chunking.py        (pytest — chunking logic is pure and
                               deterministic, no API calls, so it gets
                               normal unit tests: boundary cases, empty
                               text, exactly-150-char text, text that must
                               split into 2+ pieces using synthetic
                               examples that guarantee the split path runs)
tests/test_retrieve.py        (pytest — the score-threshold decision is
                               also pure logic once given a score; tested
                               with fake scores above/below/exactly at the
                               threshold, no live Pinecone call needed)


One-command setup (Phase 6, mirrors Block 3's run_all.py):

scripts/run_all.py            (runs check_connection -> create_index ->
                               chunk_records + ingest -> verify, in order,
                               as one command. Does NOT run run_eval.py —
                               eval is a separate, repeatable step you run
                               on purpose, not something that should fire
                               automatically every time you set up the
                               project.)
```

## Tech stack

| Component | Notes |
|---|---|
| Python | 3.11 |
| pinecone | Python SDK, integrated inference support (`create_index_for_model`, `upsert_records`, `search`) — exact version pinned once installed |
| anthropic | Python SDK for Claude API — exact version pinned once installed |
| fastapi | API framework |
| uvicorn | the web server that actually runs the FastAPI app |
| pandas | CSV joins for eval ground-truth computation |
| python-dotenv | loads `.env` credentials |
| orjson | fast JSON read/write for chunking and eval scripts |
| pytest | unit tests for chunking and threshold-decision logic |

> Exact version numbers get filled in during Phase 2 setup, once installed —
> per project rule, `requirements.txt` uses `==` pins, never `>=`.

## Credentials and configuration

`.env` (git-ignored), `.env.example` committed as a template. Required
variables:

| Variable | Purpose |
|---|---|
| `PINECONE_API_KEY` | Pinecone authentication |
| `PINECONE_INDEX_NAME` | Name of the integrated-inference index this project creates |
| `ANTHROPIC_API_KEY` | Claude API authentication |

All scripts load credentials via `python-dotenv`. No credentials are ever
hardcoded or logged. `scripts/check_connection.py` is the first thing run
after setup, so a bad key is caught immediately instead of mid-ingestion.

## Chunking design

- Rule: if a patient's `text` field is **≤ 150 characters**, it stays as a
  single chunk. If longer, whole sentences are added to a chunk one at a
  time — keep adding the next sentence as long as the chunk still fits
  within 150 characters, then close it and start a new chunk with that
  sentence. This is deliberately "group whole sentences up to the limit,"
  not "one sentence = one chunk," which could produce needlessly choppy
  single-sentence chunks even when two short sentences would comfortably
  fit together.
- **Why 150, not a round number like 400:** this dataset only has 3 possible
  conditions and 6 possible drugs (Block 1's whitelist), so even the
  worst-case patient produces roughly 220–230 characters of text. A 400-
  character threshold would mean chunking *never fires* on real data, and
  the multi-chunk path would ship untested against the actual corpus. 150
  is chosen so the genuinely high-burden patients (the same ones Block 3's
  Q3 query surfaces) actually do split, giving the logic real data to run
  against.
- **Empty text:** a patient with an empty `text` field still produces one
  chunk — a single empty-string chunk — rather than being dropped, so the
  "every patient produces ≥1 chunk" invariant (Expected statistics) holds.
  Such a chunk simply never scores as a strong retrieval match.
- **Fallback case:** if a single sentence is itself longer than 150
  characters (not expected here, but not guaranteed forever if the text
  template changes), it becomes its own oversized chunk rather than
  crashing or silently truncating. `tests/test_chunking.py` covers this
  and the multi-chunk split with synthetic long text — a safety net that
  doesn't depend on this specific dataset happening to contain a
  long-enough real record.
- Chunk ID format: `{person_id}_chunk{n}` (e.g. `9981_chunk0`, `9981_chunk1`),
  stored under the `_id` key — Pinecone's `upsert_records` API requires
  `_id` specifically (not `id`) to recognize it as the record identifier.
  Guarantees uniqueness and makes it obvious which patient a chunk belongs
  to just from the ID.
- Every chunk carries the parent patient's full metadata (`person_id`,
  `gender`, `year_of_birth_band`, `condition_count`, `drug_count`,
  `visit_count`) plus a `chunk_index` field — so no matter which chunk gets
  retrieved, we know exactly which patient and which piece of their record
  it came from.

Example — a short record (stays as 1 chunk):

```json
{
  "_id": "248_chunk0",
  "chunk_text": "Patient 248, born in the 1960s, Female. Conditions: Type 2 diabetes mellitus. Drugs: Metformin, Lisinopril. Visits: 5.",
  "person_id": 248,
  "gender": "Female",
  "year_of_birth_band": "1960s",
  "condition_count": 1,
  "drug_count": 2,
  "visit_count": 5,
  "chunk_index": 0
}
```

## Pinecone index design

- Index type: serverless, integrated inference (Pinecone embeds server-side
  — we send text, not vectors)
- Embedding model: `llama-text-embed-v2` (current top-performing Pinecone-
  hosted model on public benchmarks as of this writing; outperforms OpenAI's
  text-embedding-3-large on retrieval benchmarks)
- `field_map`: `{"text": "chunk_text"}` — tells Pinecone which field in each
  upserted record holds the text to embed
- Cloud/region: AWS, `us-east-1` (Pinecone's free serverless tier region)
- `scripts/create_index.py` owns index creation and is idempotent — it
  checks whether an index with `PINECONE_INDEX_NAME` already exists before
  attempting to create one, so re-running it is always safe
- `scripts/ingest.py` does **not** create the index — it assumes
  `create_index.py` has already run, and fails with a clear, actionable
  error message if the index is missing, rather than silently creating one
  with possibly-wrong settings
- All chunks live in a single fixed namespace — a namespace is just a
  named folder inside the index. **This is a hardcoded constant in the
  code (`NAMESPACE = "patients"`), not a `.env` variable** — unlike the
  API keys and index name, which legitimately differ between people's
  setups, the namespace name is an internal design decision that should
  stay the same everywhere. It does not appear in the Credentials and
  configuration table above for that reason.
- **In short: `ingest.py` deletes everything and rebuilds it fresh, every
  run.** That single rule is what makes it safe to re-run ("idempotent" —
  a fancy word for "running it twice doesn't break anything or leave a
  mess"). Here's why that rule is necessary, not just extra caution:
  overwriting an existing chunk ID isn't enough on its own — if the
  chunking logic ever changes (e.g. the threshold gets tuned in Phase 5)
  and a patient who used to produce 2 chunks now produces 1, the old
  second chunk's ID would never get overwritten, but it also wouldn't get
  deleted, leaving a stale, invisible leftover corrupting retrieval
  forever. Deleting the whole namespace and reloading it fresh every time
  sidesteps that entirely — simpler and more obviously correct than
  carefully figuring out which old chunks to remove, at the cost of
  re-embedding everything each run (a small cost at ~15,000 short chunks).
  On the very first run the namespace doesn't exist yet, so the delete
  step must not fail when there's nothing to delete — `ingest.py` treats
  a missing namespace as already-empty, not an error. Chunks are upserted
  in batches, not one giant call — Pinecone's integrated-inference upsert
  has a per-call record/payload limit, and ~15,000 chunks exceeds it.
  Operational consequences of this strategy (the brief empty-index window,
  the short delay before new data is queryable, and the required
  idempotency check) are covered together in Reproducibility and
  determinism below.

## Expected statistics

Mirroring Block 3's approach — `scripts/verify.py` computes these numbers
by actually running the chunking logic against the source file at check
time, not from a hardcoded guess baked into this doc:

| Metric | Expected |
|---|---|
| Source patient records | 11,424 (must match Block 3's `graph_export.jsonl` record count) |
| Pinecone vector count | ≥ 11,424 — exactly 11,424 if no patient's text exceeds 150 characters, higher if some high-burden patients split into 2+ chunks. The precise number is computed by `scripts/verify.py` re-running `chunk_records.py` against the source file and counting the output, then compared against Pinecone's actual reported vector count. |
| Repeat-ingestion vector count | Must be identical to the first run's count (idempotency check — see Reproducibility and determinism) |

## Retrieval design (Job 1)

- Input: a question string, optional `top_k` (default 5)
- Pinecone embeds the question automatically (same model as ingestion) and
  returns the `top_k` most similar chunks, each with a similarity score
  (0 to 1) and its full metadata
- **Score threshold: 0.75 (starting default).** If the single best-matching
  chunk scores below this threshold, retrieval is treated as empty — nothing
  relevant was found
- This threshold is one of the two candidate parameters for the required
  eval experiment in Phase 5 (the other being `top_k`) — Phase 5 will run
  the eval at two different values and document which one scores better,
  rather than treating 0.75 as fixed
- The threshold comparison itself (`score >= threshold`) is pure logic once
  a score exists, so it is unit-tested directly with fake scores in
  `tests/test_retrieve.py` — no live Pinecone call needed to verify this
  part is correct
- **Must confirm, not assume: does a higher number mean "more relevant"?**
  This spec assumes yes — that's Pinecone's typical default. But it isn't
  guaranteed. If the index ends up set up so a *lower* number means more
  relevant instead (this happens with some scoring methods), then `score
  >= threshold` would quietly do the opposite of what we want — keep the
  worst matches and throw away the best ones — with no crash or error to
  reveal it. This is confirmed in two steps, since no chunks exist until
  Phase 3: Phase 2's `create_index.py` prints the metric name at creation
  time (cosine/dotproduct mean higher-is-better, euclidean means
  lower-is-better — no data needed for this part). Phase 3, right after
  the first real ingestion, runs one empirical check — search using a
  chunk's own text and confirm it comes back as its own best match —
  before any threshold logic is trusted.

## Generation design (Job 2)

- Model: Claude (`claude-haiku-4-5`) — fast and inexpensive, appropriate for
  short, grounded answers
- If retrieval's best score is below the threshold, **Claude is not called
  at all.** The API returns a fixed message (e.g. "I don't know — I
  couldn't find any patient records relevant to that question.") directly.
  This is deterministic, costs nothing, and removes any chance of the model
  guessing when there's no real context.
- If retrieval succeeds, Claude receives: the original question, and only
  the chunks that individually score at or above the threshold (not every
  chunk in the `top_k` list — a weak match that rode along with a strong
  one should not reach the model). Instructed to answer using only the
  provided chunks, and to cite the `person_id` for every claim it makes.
- **Known limitation:** nothing automatically checks that the `person_id`s
  Claude cites actually match the chunks it was given — correctness relies
  on prompt instructions alone, not a code-level verification step. This is
  a conscious scope decision for this block, not an oversight (see Scope).

## API design

Single endpoint, `POST /query`. This is also the interface contract Block 5
is expected to call as one of its agent's tools ("Agent uses ≥2 tools, one
of them your RAG service") — so the request/response shape below should be
treated as stable, the same way Block 3's export shape was written for
Block 4 to consume.

Request:
```json
{ "question": "Which patients have type 2 diabetes and take metformin?", "top_k": 5 }
```
`top_k` is optional (default 5), validated to an integer between 1 and 20
— out-of-range or invalid values return HTTP 422 before Pinecone is ever
called. An empty or whitespace-only `question` also returns HTTP 422,
same as an invalid `top_k`.

Response (relevant results found):
```json
{
  "answer": "Two patients match: Patient 248 (Type 2 diabetes, taking Metformin and Lisinopril) and Patient 5107 (Type 2 diabetes and hypertension, taking Metformin and Insulin).",
  "sources": [
    { "person_id": 248, "chunk_id": "248_chunk0", "score": 0.91 },
    { "person_id": 5107, "chunk_id": "5107_chunk0", "score": 0.88 }
  ],
  "retrieved_count": 2
}
```

Response (nothing relevant found):
```json
{
  "answer": "I don't know — I couldn't find any patient records relevant to that question.",
  "sources": [],
  "retrieved_count": 0
}
```

Response (upstream failure — Pinecone or Claude unreachable, rate-limited,
or times out):
```json
{
  "error": "Retrieval service unavailable. Please try again shortly.",
  "detail": "pinecone_timeout"
}
```
Returned with an HTTP 502 status code. The server logs the underlying
exception but never crashes or returns a raw stack trace to the caller.
For the Phase 4 manual test, this path can be triggered without a real
outage — temporarily set an invalid `PINECONE_API_KEY` or
`ANTHROPIC_API_KEY`, confirm the 502, then restore the real key.

## Eval harness design

- `data/eval/questions.json`: at least 20 questions, generated with a
  mix of intent:
  - condition + drug co-occurrence questions (e.g. diabetes + metformin)
  - demographic-filtered questions (e.g. male patients born in the 1970s
    with hypertension)
  - high-burden / visit-count questions (mirroring Block 3's Q3 and Q4)
  - deliberately out-of-scope questions with no correct answer in this
    dataset (e.g. asking about a condition outside the 3-condition
    whitelist), which should trigger "I don't know" — these test the
    fallback path, not just the happy path. At least 5 of the 20 questions
    must be in this category — fallback accuracy computed over 1–2
    questions would be too noisy to mean anything
- **Ground truth is computed, not hand-typed.** `scripts/
  build_eval_answer_key.py` recomputes each question's exact correct
  patient-ID set directly from the OMOP CSVs with pandas (same logic as
  Block 3's Cypher queries) — typing out which of 11,424 patients match a
  filter by hand would be error-prone and unverifiable. It also asserts
  the labeling is honest: every answerable question's computed set is
  non-empty, and every deliberately-unanswerable question's set is empty —
  failing loudly if a question is mislabeled, not just relying on the
  manual spot-check below
- **Two separate metrics are reported, not blended into one number:**
  - **Retrieval precision/recall** — computed only across the questions
    that have a real answer set (patient-ID overlap between retrieved and
    ground truth). "Retrieved" here means the same set Job 2 would
    actually see — only chunks that individually clear the score
    threshold, not every chunk in the raw `top_k` list. Scoring against
    the unfiltered list would measure a more lenient system than what the
    API actually serves. Those threshold-clearing chunks are first deduped
    to unique `person_id`s (a patient can contribute more than one chunk),
    then: precision = correct / retrieved (0, not a divide-by-zero error,
    when nothing was retrieved); recall = correct / total actually correct.
    The single reported precision and recall are **micro-averaged** across
    all answerable questions — total correct summed over all questions,
    divided by total retrieved (or total actually-correct) summed over all
    questions — not a per-question average, so one question's edge case
    can't disproportionately swing the headline number.
  - **Fallback accuracy** — computed only across the deliberately
    unanswerable questions: the percentage that correctly triggered
    "I don't know." This is a different kind of correctness (did it
    decline or not) and does not belong in the same average as precision/
    recall, which is a set-overlap metric.
- **`run_eval.py` calls `retrieve.py` directly, never `generate.py` or
  Claude** — all three metrics are fully computable from retrieval results
  and the threshold check alone (see Reproducibility and determinism for
  why that also makes the score more trustworthy). This also avoids ~20
  unnecessary Claude calls, and their cost/latency/rate-limit risk, on
  every eval run.
- `scripts/run_eval.py` produces a saved, reproducible score report,
  committed to the repo at `docs/eval_results.md` — running it twice on
  unchanged code must produce the same score. Despite being script-
  generated output, this file is committed (not `.gitignore`d) because
  it's the required evidence for the "documents ≥1 experiment" acceptance
  criterion.
- **Required experiment:** run the full eval twice with one setting changed
  (`top_k` or the score threshold), record both scores (both metrics) side
  by side, and write a short note on what changed and why
- **No minimum passing score is required by this spec.** The acceptance
  criteria only require a reproducible score and a documented experiment —
  not hitting a specific precision/recall target. Whatever the honest score
  is, it gets reported.

## Reproducibility and determinism

This section exists because a RAG pipeline has both deterministic and
non-deterministic parts, and it matters a lot which is which:

- **Retrieval (Job 1) is deterministic, and that's exactly why it's the
  thing we score.** Embedding a piece of text with `llama-text-embed-v2`
  is not a sampling process — the same text always produces the same
  vector. Given a stable, fully-loaded index, the same question will
  retrieve the same ranked chunks every time. This is why precision,
  recall, and fallback accuracy — the metrics the eval harness reports —
  are computed purely from retrieval results, never from Claude's written
  answer.
- **Generation (Job 2) is not deterministic**, by design of how LLMs work
  — the same prompt can produce slightly different wording on different
  calls. This is precisely why generation quality is deliberately *not*
  part of the reproducible score (see Known limitations). If a future
  block wants to grade answer quality, that needs a different approach
  (e.g. an LLM-as-judge eval with explicitly acknowledged variance), not
  the precision/recall numbers this spec defines.
- **Two assumptions this reproducibility relies on, stated explicitly
  rather than left implicit:**
  1. Pinecone's search is effectively exact at our scale (~15,000
     vectors). Very large vector databases sometimes use a faster
     "close enough" search shortcut that can break near-ties differently
     between runs — not a real concern at our size, but worth naming.
  2. `llama-text-embed-v2` is a named, versioned model — we're relying on
     Pinecone keeping its behavior stable over time. If Pinecone ever
     changes what that name points to, scores computed months apart could
     drift even with unchanged code. "Reproducible" in this spec means
     same-code, same-index, same-model-version — not eternally frozen.
- **Operational consequences of delete-and-reload ingestion (see Pinecone
  index design):** three things follow directly from that strategy, and
  `scripts/verify.py` is what guards against all three. First, Pinecone
  writes are eventually consistent — a short delay can exist between an
  upsert returning and the vector being queryable, so `verify.py` polls
  with a bounded retry loop (short delay between attempts, a capped
  attempt count) rather than a single fixed sleep or an immediate count
  check, since Pinecone doesn't guarantee how long consistency takes.
  Second, deleting the namespace before reloading it means the index is
  briefly empty or partial mid-ingestion, so the API and eval harness must
  never run concurrently with an active ingestion — before `verify.py`
  exists (Phases 3–5), Phase 3's idempotent re-run and Phase 4's manual
  query test are what confirm the index is safely loaded; `verify.py`
  (Phase 6) is the final, comprehensive check that ties everything together
  at the end, not a real-time gate you run before every query. Third,
  idempotency is *verified*, not assumed: Phase 6 re-runs `ingest.py` a
  second time and confirms the vector count is unchanged, the same check
  Block 3 ran on its graph loader.

## Phases

| Phase | Branch | Deliverables |
|---|---|---|
| 1 | `phase-1-spec` | `docs/spec.md`, `docs/plan.md`, `docs/tasks.md` |
| 2 | `phase-2-setup` | `requirements.txt`, `.env.example`, `.gitignore`, `scripts/check_connection.py`, `scripts/create_index.py` |
| 3 | `phase-3-ingest` | `scripts/chunk_records.py`, `scripts/ingest.py`, `tests/test_chunking.py`, `data/raw/graph_export.jsonl` (copied from Block 3) |
| 4 | `phase-4-retrieve-generate` | `scripts/retrieve.py`, `scripts/generate.py`, `scripts/api.py`, `tests/test_retrieve.py` |
| 5 | `phase-5-eval` | `data/raw/condition_occurrence.csv`, `drug_exposure.csv`, `person.csv` (copied from Block 1, needed only here for ground truth), `data/eval/questions.json`, `scripts/build_eval_answer_key.py`, `scripts/run_eval.py`, `docs/eval_results.md` (includes the parameter experiment) |
| 6 | `phase-6-verify-docs` | `scripts/run_all.py`, `scripts/verify.py`, `README.md` |

Each phase gets its own PR (6 PRs total). `phase-1-spec` branches from
`main`; every phase after that branches from the tip of the previous
phase's branch rather than `main`, since an earlier phase's PR may not be
merged yet when the next phase starts — this is what Block 3 actually did.
See `docs/tasks.md` for the exact branch-from/PR sequence.

## Scope

Block 4 does not include:
- A chat interface, conversation memory, or multi-turn follow-up questions
- Authentication or access control on the API
- Real-time ingestion of new patients — this is a static, one-time load of
  Block 3's export
- A reranking model or hybrid (keyword + vector) search
- Support for questions about conditions/drugs outside the 3-condition,
  6-drug whitelist established in Block 1 — those are expected to correctly
  produce "I don't know," not an error

## Known limitations

Documented consciously, not discovered later — full reasoning lives with
each design section, this is just the index:
- No automated grounding check on Claude's citations (see Generation design)
- 150-char chunking threshold is dataset-specific, not general-purpose (see Chunking design)
- No minimum eval score is enforced (see Eval harness design)
- Pinecone and Claude failures share one generic error response (see API design)

## Functional requirements

Block 4 must:
1. Provide a connectivity smoke test (`scripts/check_connection.py`) that
   confirms `PINECONE_API_KEY` and `ANTHROPIC_API_KEY` are valid before any
   ingestion or eval work begins.
2. Chunk `graph_export.jsonl` records — records ≤ 150 characters stay
   whole; longer records split on sentence boundaries. Covered by unit
   tests using synthetic long text, independent of whether real data
   happens to trigger a split.
3. Create a Pinecone serverless index using integrated inference (no
   separate embedding API call anywhere in this codebase), via an
   idempotent `scripts/create_index.py` that is safe to re-run.
4. Upload all chunks to the existing Pinecone index idempotently, into a
   single fixed namespace, by deleting that namespace's contents and
   re-upserting the full current chunk set on every run — re-running
   ingestion must never leave duplicate or orphaned vectors behind, even
   if the chunking logic changes between runs.
5. Accept a natural-language question and return the top-k most similar
   chunks with similarity scores (Job 1 / retrieval). The score-threshold
   decision is covered by unit tests using fake scores. Before any
   threshold logic is trusted, confirm empirically which direction the
   index's metric actually points (higher-is-better vs. lower-is-better) —
   do not assume.
6. When the best retrieval score is below the configured threshold, skip
   the Claude call and return a fixed "I don't know" response.
7. When retrieval succeeds, call Claude with only the retrieved chunks and
   the question, and require the response to cite `person_id` per claim
   (Job 2 / generation).
8. Expose the pipeline as a FastAPI service with a `POST /query` endpoint
   that returns a clear error response (HTTP 502, no stack trace) if
   Pinecone or Claude is unreachable, rate-limited, or times out.
9. Provide a labelled eval set of ≥20 questions with a programmatically
   computed (not hand-typed) ground-truth answer key.
10. Compute and save two separate, reproducible metrics from the eval set:
    retrieval precision/recall (answerable questions) and fallback accuracy
    (deliberately unanswerable questions) — computed by calling `retrieve.py`
    directly, never `generate.py` or Claude.
11. Run and document ≥1 experiment showing a parameter change moving the
    eval score(s).
12. Include a verification script confirming: credentials are valid, the
    Pinecone index is reachable, the stored chunk count matches what the
    chunking logic should produce from the source file, re-running
    ingestion a second time produces an identical vector count
    (idempotency, verified not assumed), a sample query returns non-empty
    results, and the eval harness has produced a saved score report.
13. Run the eval harness only after confirming the index is fully and
    correctly loaded — in practice, that means after Phase 3's idempotent
    re-run and Phase 4's manual query test have already passed, since
    `scripts/verify.py` doesn't exist until Phase 6. Never run eval against
    a partially-ingested index.
14. Provide `scripts/run_all.py` so the whole setup-through-verify pipeline
    runs end-to-end from one command, the same way Block 1 and Block 3
    required. `run_eval.py` stays a separate command on purpose — it's
    something you run deliberately, possibly more than once with different
    settings, not something that should fire automatically.

## Success criteria

Block 4 is complete when:
- `scripts/check_connection.py` confirms both API keys are valid
- `scripts/ingest.py` loads all chunked patient records into Pinecone
  idempotently, confirmed by re-running it and seeing an identical vector
  count both times
- `pytest` passes for chunking and threshold-decision unit tests
- `POST /query` returns cited, grounded answers for in-scope questions, a
  correct "I don't know" for out-of-scope questions, and a clear error
  response (not a crash) if Pinecone or Claude is unreachable
- `scripts/run_eval.py` produces reproducible precision/recall and fallback
  accuracy scores against the programmatically-computed answer key
- At least one documented experiment shows a parameter change moving the
  score(s)
- `scripts/verify.py` passes all checks
- `scripts/run_all.py` sets up and loads everything with one command
- `README.md` documents setup, architecture, and the AI-assisted workflow
