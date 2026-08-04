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
`person.csv` (the same Block 1 OMOP files Block 3 used) — needed because
`graph_export.jsonl` metadata only stores *counts* (`condition_count`,
`drug_count`), not condition/drug names as structured, filterable fields.
`measurement.csv` and `visit_occurrence.csv` are also copied in, for a
different reason — see below.

Block 3 artifacts reused:
- `graph_export.jsonl` (11,436 patient records, `text` + `metadata` fields)
- `condition_occurrence.csv`, `drug_exposure.csv`, `person.csv` (for eval
  ground-truth computation only — never touched by the live RAG pipeline)
- The 4 Cypher queries' *logic* (reimplemented as pandas filters against the
  CSVs, since this repo has no Neo4j connection)

Block 1 artifacts reused directly (not via Block 3):
- `measurement.csv` — the four lab values are already structured in
  `graph_export.jsonl`'s metadata, but they're Block 3's own computed
  "latest per patient" output. Using `measurement.csv` instead means
  lab-threshold ground truth is independently recomputed, not just
  trusted from Block 3's pipeline — the same principle applied to
  conditions/drugs above, just for a different reason (there, the
  structured data is simply absent from the JSONL; here, it's present
  but not independently sourced). Replicates Block 1's "latest value per
  concept per patient" logic (`measurement_concept_id` mapping, copied
  into this repo as `scripts/concepts.py`).
- `visit_occurrence.csv` — same reasoning as `measurement.csv`, applied to
  `visit_count`: that field is also already present in `graph_export.jsonl`'s
  metadata as Block 3's own precomputed output, so high-burden/visit-count
  eval ground truth is recomputed from `visit_occurrence.csv` instead of
  trusted from the JSONL, for the same independence reason.
- `scripts/concepts.py` — Block 1's Synthea-code-to-concept-ID mapping,
  copied in so `build_eval_answer_key.py` can map condition/drug/measurement
  names the same way Block 1 and Block 3 did, without redefining the
  whitelist a third time.

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

data/raw/graph_export.jsonl (11,436 patient records)
       |
       v
scripts/chunk_records.py     (split long `text` fields into <=200-char
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
                               text, exactly-200-char text, text that must
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
| pinecone==9.1.0 | Python SDK, integrated inference support (`create_index_for_model`, `upsert_records`, `search`) |
| anthropic==0.116.0 | Python SDK for Claude API |
| fastapi==0.139.0 | API framework |
| uvicorn==0.51.0 | the web server that actually runs the FastAPI app |
| pandas==3.0.3 | CSV joins for eval ground-truth computation |
| python-dotenv==1.2.2 | loads `.env` credentials |
| orjson==3.11.9 | fast JSON read/write for chunking and eval scripts |
| pytest==9.1.1 | unit tests for chunking and threshold-decision logic |

> Versions pinned in Phase 2 setup, installed into `.venv`. Per project
> rule, `requirements.txt` uses `==` pins, never `>=`.

## Credentials and configuration

`.env` (git-ignored), `.env.example` committed as a template. Required
variables:

| Variable | Purpose |
|---|---|
| `PINECONE_API_KEY` | Full-access Pinecone authentication — `create_index.py`, `ingest.py`, `verify.py` |
| `PINECONE_QUERY_API_KEY` | Scoped Pinecone authentication (DataPlaneViewer, query only) — `retrieve.py`, i.e. the live query path exposed via `POST /query` |
| `PINECONE_INDEX_NAME` | Name of the integrated-inference index this project creates |
| `ANTHROPIC_API_KEY` | Claude API authentication |

All scripts load credentials via `python-dotenv`. No credentials are ever
hardcoded or logged. `scripts/check_connection.py` is the first thing run
after setup, so a bad key is caught immediately instead of mid-ingestion.

**Key scoping (Phase 11):** the query path (`retrieve.py`, and therefore
`POST /query`) uses a separate Pinecone API key scoped to DataPlaneViewer,
query-only access, rather than the same full-access key the offline
ingestion/admin scripts use. The live API is the one thing external
callers (Block 5) actually reach; scoping its key means a compromise of
that path cannot write or delete index data, even though the credential
that reaches it is the one most exposed to untrusted input.

## Chunking design

- Rule: if a patient's `text` field is **≤ 200 characters**, it stays as a
  single chunk. If longer, whole sentences are added to a chunk one at a
  time — keep adding the next sentence as long as the chunk still fits
  within 200 characters, then close it and start a new chunk with that
  sentence. This is deliberately "group whole sentences up to the limit,"
  not "one sentence = one chunk," which could produce needlessly choppy
  single-sentence chunks even when two short sentences would comfortably
  fit together.
- **Why 200, not the old 150 or a round number like 400:** this dataset has
  11 conditions and 17 drugs (not the old 3-condition/6-drug estimate), and
  real records in `data/raw/graph_export.jsonl` range 77–348 characters.
  150 splits 28.7% of patients — too many to call "only high-burden." 400
  splits almost none (real max is 348), repeating the same
  untested-multi-chunk-path problem this doc originally flagged. 200 splits
  7.8% (893 patients): most stay single-chunk, and high-burden patients
  (Block 3's Q3 population) still get real multi-chunk coverage.
- **Empty text:** a patient with an empty `text` field still produces one
  chunk — rather than being dropped, so the "every patient produces ≥1
  chunk" invariant (Expected statistics) holds. Such a chunk simply never
  scores as a strong retrieval match. This path is defensive only — Block
  3's template always emits a full sentence even for a patient with
  nothing to report (e.g. "Conditions: none. Drugs: none."), so real data
  never actually hits it. **Confirmed in Phase 3 against the real index:
  Pinecone's integrated inference rejects an empty string for
  `chunk_text`** (`400 INVALID_ARGUMENT`, embedding error). `chunk_text()`
  substitutes the placeholder string `"No data available."`
  (`EMPTY_TEXT_PLACEHOLDER` in `scripts/chunk_records.py`) instead of `""`
  in this case.
- **Fallback case:** if a single sentence is itself longer than 200
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
- **Sanitization (Phase 11):** `scripts/sanitize.py`'s
  `sanitize_chunk_text()`, called from `ingest.py` before upsert, strips
  conversation-turn markers (`System:`/`Human:`/`Assistant:`/`User:`
  prefixes, chat-template delimiter tokens) from `chunk_text`.
  `chunk_text` flows verbatim into `generate.py`'s LLM prompt and into
  `/query`'s `sources[].chunk_text` citation field — sanitizing once at
  ingestion, the single point every chunk is written to Pinecone, covers
  both readers from one place, rather than only covering the prompt if it
  were done in `generate.py` instead. See `scripts/sanitize.py`'s module
  docstring for the full reasoning.
- Every chunk carries the parent patient's full metadata (`person_id`,
  `gender`, `year_of_birth_band`, `condition_count`, `drug_count`,
  `visit_count`, `latest_sbp`, `latest_bmi`, `latest_glucose`,
  `latest_hba1c`) plus a `chunk_index` field — so no matter which chunk
  gets retrieved, we know exactly which patient and which piece of their
  record it came from.

Example — a short record (stays as 1 chunk):

```json
{
  "_id": "248_chunk0",
  "chunk_text": "Patient 248, born in the 1960s, Female. Conditions: Type 2 diabetes mellitus. Drugs: Metformin, Lisinopril. Visits: 5. Latest labs: HbA1c 7.2%, Glucose 142 mg/dL, SBP 138 mmHg, BMI 28.4.",
  "person_id": 248,
  "gender": "Female",
  "year_of_birth_band": "1960s",
  "condition_count": 1,
  "drug_count": 2,
  "visit_count": 5,
  "latest_sbp": 138,
  "latest_bmi": 28.4,
  "latest_glucose": 142,
  "latest_hba1c": 7.2,
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
| Source patient records | 11,436 (must match Block 3's `graph_export.jsonl` record count) |
| Pinecone vector count | ≥ 11,436 — higher whenever a patient's text exceeds the 200-char threshold and splits into 2+ chunks (real data: ~7.8% of patients, see Chunking design). The precise number is computed by `scripts/verify.py` re-running `chunk_records.py` against the source file and counting the output, then compared against Pinecone's actual reported vector count. |
| Repeat-ingestion vector count | Must be identical to the first run's count (idempotency check — see Reproducibility and determinism) |

## Retrieval design (Job 1)

- Input: a question string, optional `top_k` (default 20, revision
  history: initial 5 → 15 (Phase 7, Run 5) → 20 (this later fixup, Run
  7)). Run 5 (`top_k=25`, `threshold=0.4`, a flat unfiltered-style
  threshold) measured the highest per-question `retrieved` count actually
  observed across all 12 answerable questions in the filtered eval subset
  at 11 (q01 and q02) — every other question retrieved fewer — so 15 was
  picked as a ~36% margin above that. But once `select_threshold()`
  shipped live in `scripts/api.py` (see below), condition/drug-filtered
  queries automatically get the permissive 0.2 threshold instead of 0.4 —
  and Run 7 (`top_k=20`, `threshold=0.2`) showed `top_k=15` still
  truncated 3 of 12 tested questions (q01, q06, q08) under that
  threshold, while `top_k=20` closed nearly all of that gap (recall
  0.884 → 0.977, Run 6 → Run 7). Since a real caller (Block 5) typically
  sends condition/drug-filtered queries — exactly the regime where the
  permissive threshold is now the common case, not the exception — 20 is
  the better-supported default. See `docs/eval_results.md`'s Run 5, Run
  7, and Run 9, and `scripts/retrieve.py`'s `DEFAULT_TOP_K` for where
  this lives in code.
- **`FILTERED_TOP_K_CEILING` (25) — a higher `top_k` ceiling for
  condition/drug-filtered queries only, mirroring how
  `select_threshold()` gates `PERMISSIVE_THRESHOLD` on the same
  condition.** Confirmed empirically against the real Pinecone client
  (not assumed): a condition/drug-filtered `top_k=25` returns cleanly.
  Deliberately not set higher — a much bigger `top_k` turns a similarity
  search into a table scan, the wrong tool for a fully-structured query;
  that job belongs to the graph (see Block 5's `docs/spec.md` "what I'd
  do next" for the planned follow-up). See `docs/eval_results.md`'s Run
  11 and `scripts/retrieve.py`'s `FILTERED_TOP_K_CEILING` for where this
  lives in code.
- Pinecone embeds the question automatically (same model as ingestion) and
  returns the `top_k` most similar chunks, each with a similarity score
  (0 to 1) and its full metadata
- **Score threshold: 0.4 (default, revised from an initial 0.75 in Phase
  5).** If the single best-matching chunk scores below this threshold,
  retrieval is treated as empty — nothing relevant was found. The initial
  0.75 was carried over from Phase 3's self-match check (querying with a
  chunk's own exact text, which scored 0.82–0.84) — a much easier case
  than a natural-language question matched against chunk text. Phase 5's
  eval run showed real question scores cluster ~0.4–0.55, so 0.75 made
  every question fall back regardless of relevance (0 precision/recall
  across all 15 answerable questions). See `docs/eval_results.md`'s Run 1
  vs Run 2 for the measured comparison and `scripts/retrieve.py`'s
  `DEFAULT_THRESHOLD` for where this lives in code.
- This threshold is one of the two candidate parameters for the required
  eval experiment in Phase 5 (the other being `top_k`) — Phase 5 ran the
  eval at two different values (0.75 and 0.4) and documented which one
  scores better; 0.4 was subsequently adopted as the new default
- **Permissive threshold for condition/drug-filtered queries (Phase 7).**
  `scripts/api.py`'s `/query` handler no longer always uses
  `DEFAULT_THRESHOLD` — it computes the threshold per request via
  `scripts/retrieve.py`'s `select_threshold(condition=..., drug=...)`,
  which returns `PERMISSIVE_THRESHOLD` (0.2) whenever a `condition` or
  `drug` filter is present, and `DEFAULT_THRESHOLD` (0.4) otherwise. Both
  the fallback check and the `relevant_chunks` filter use the same
  computed threshold. See Known limitations for why only `condition`/
  `drug` (not `gender`/`lab`/`birth_decade`) can safely trigger this, and
  `docs/eval_results.md`'s Run 6 for the measured evidence.
- See Known limitations for why this single threshold does double duty as
  both the relevance gate and the out-of-scope gate.
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

## Metadata filter design (Phase 7)

- Adds optional structured filters alongside semantic search: `condition`,
  `drug`, `lab`+`comparison`+`value`, `gender`, `birth_decade` — these
  narrow Pinecone's search to chunks whose metadata matches, on top of
  (not instead of) the existing similarity search and score threshold.
- Built entirely on metadata already present in the index: Block 3's
  `conditions`/`drugs` list fields (added in Phase 6's data refresh) plus
  the existing `gender` and `year_of_birth_band` fields — no new metadata
  or re-ingestion required.
- Lab/comparison naming (`_LAB_PROPERTY` mapping `"SBP"/"BMI"/"Glucose"/
  "HbA1c"` to the stored `latest_*` field names, `_COMPARISON_OP` mapping
  `"above"/"below"`) matches Block 5's `graph_tool.py` exactly, so both
  tools present identical shorthand to a caller/agent. The underlying
  operator differs by necessity: Block 5 emits Cypher `>`/`<`, this emits
  Pinecone filter operators `$gt`/`$lt` — both strict inequalities, so
  "above"/"below" mean the same thing (never inclusive) in both tools.
- Value mapping, not pass-through: `gender` shorthand `"M"`/`"F"` maps to
  Pinecone's stored `"Male"`/`"Female"`; `birth_decade` (an int, e.g.
  `1970`) maps to Pinecone's stored `"{decade}s"` string (e.g. `"1970s"`).
  Passing either field through unmapped would silently match nothing,
  since Pinecone's stored values never look like `"M"` or `1970`.
- `lab`, `comparison`, `value` are a trio: either all three are present or
  all three are absent. A partial combination raises before any Pinecone
  call — an incomplete lab filter is a caller bug, not a "no filter"
  fallback. `condition`, `drug`, `gender`, `birth_decade` are each
  independently optional; any combination, or none, is valid.
- An unrecognized `lab` or `comparison` value raises before ever calling
  Pinecone (`.get()` lookups against the fixed whitelists above, never raw
  indexing) — same fail-fast principle as Block 5's `graph_tool.py`.
- No filter fields passed → the filter builder returns `None`/empty, and
  retrieval behaves identically to pre-Phase-7 — existing callers (eval
  harness, prior API requests) see no behavior change.
- **Verified against the installed `pinecone` client (v9.1.0), not docs
  samples, before wiring this up for real** — docs samples mix calling
  conventions across client versions:
  - `index.search()` takes `filter` as a flat keyword argument (confirmed
    from the installed client's own method signature) — no need for the
    nested `query={"inputs": ..., "top_k": ..., "filter": ...}` form.
  - Multiple top-level filter keys AND-combine, not OR — confirmed with a
    live call combining a `conditions` filter, a `gender` filter, and a
    `latest_sbp` `$gt` filter together; every returned hit satisfied all
    three simultaneously.
  - Equality filtering against a list-of-strings metadata field
    (`conditions`, `drugs`) matches "list contains this value," not
    literal equality against the whole list — confirmed with a live call:
    filtering on `conditions: "Essential hypertension"` correctly matched
    chunks whose `conditions` list held that value alongside others, not
    only chunks whose list was exactly `["Essential hypertension"]`.
- Not in scope: forwarding these fields from Block 5's `rag_tool.py` —
  that's a separate change in the Block 5 repo once this merges.

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
{ "question": "Which patients have type 2 diabetes and take metformin?", "top_k": 20 }
```
`top_k` is optional (default 20), validated to an integer between 1 and a
ceiling that depends on whether a `condition` or `drug` filter is present
in the same request — 20 unfiltered, 25 filtered (`FILTERED_TOP_K_CEILING`)
— out-of-range or invalid values return HTTP 422 before Pinecone is ever
called. An empty or whitespace-only `question` also returns HTTP 422,
same as an invalid `top_k`.

**Field length limits (Phase 11):** `question` (max 500 chars), `condition`
(max 100), `drug` (max 100), and `lab` (max 20) are all capped; an
over-length value returns HTTP 422. `question` is the one field that
actually reaches `generate_answer`'s prompt, so its cap is the genuinely
prompt-injection-relevant one; `condition`/`drug`/`lab` only ever become
Pinecone metadata filter values (validated against fixed whitelists in
`build_metadata_filter()`) and are capped just to keep obviously-invalid
oversized input from propagating past the API boundary.

**Phase 7 — optional metadata filters,** all `None`/absent by default (a
request with none of these behaves exactly as before Phase 7):
```json
{
  "question": "Which hypertensive men have a high latest SBP?",
  "condition": "Essential hypertension",
  "drug": null,
  "lab": "SBP",
  "comparison": "above",
  "value": 140,
  "gender": "M",
  "birth_decade": 1970
}
```
`lab`, `comparison`, `value` must be all present or all absent — a partial
combination returns HTTP 422 before Pinecone is called. `condition`,
`drug`, `gender`, `birth_decade` are each independently optional. See
Metadata filter design for the value-mapping and validation details.

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
  - condition + drug co-occurrence questions, always paired with a
    demographic filter (gender and/or year_of_birth_band) so the
    ground-truth set stays bounded — e.g. "male patients in their 1970s
    with diabetes, taking metformin," never a bare condition+drug pair
  - demographic-filtered questions (e.g. male patients born in the 1970s
    with hypertension)
  - high-burden / visit-count questions (mirroring Block 3's Q3 and Q4)
  - lab-threshold questions, always paired with a demographic filter for
    the same bounding reason — e.g. "female patients with hypertension
    and BMI over 30," never a bare "BMI over 30" — exercises the
    `latest_sbp`/`latest_bmi`/`latest_glucose`/`latest_hba1c` fields
    Block 3 added
  - deliberately out-of-scope questions with no correct answer in this
    dataset (e.g. asking about a condition outside the 11-condition
    whitelist), which should trigger "I don't know" — these test the
    fallback path, not just the happy path. At least 5 of the 20 questions
    must be in this category — fallback accuracy computed over 1–2
    questions would be too noisy to mean anything
- **Why every answerable question needs a demographic filter:** a bare
  condition+drug or lab-threshold question can match hundreds of patients
  (e.g. diabetes + metformin alone matches 222 of 11,436 real patients),
  while retrieval only returns `top_k` (max 20) chunks. Recall would be
  structurally capped near zero regardless of retrieval quality — the
  metric would measure `top_k` size, not retrieval quality. A demographic
  filter keeps ground-truth sets small enough that a good `top_k` can
  plausibly cover them.
- **Ground truth is computed, not hand-typed.** `scripts/
  build_eval_answer_key.py` recomputes each question's exact correct
  patient-ID set directly from the OMOP CSVs with pandas — condition/drug/
  demographic/burden questions reuse Block 3's Cypher-query logic;
  lab-threshold questions are computed straight from `measurement.csv`
  instead (see Relationship to Block 3, since Block 3 never queried labs).
  The same independence reasoning applies to high-burden/visit-count
  questions: ground truth for those is recomputed from `visit_occurrence.csv`
  rather than trusted from `graph_export.jsonl`'s precomputed `visit_count`
  field — using that field directly would mean trusting the same Block 3
  pipeline the eval is supposed to be independently checking, exactly the
  gap `measurement.csv` closes for lab thresholds above.
  Typing out which of 11,436 patients match a filter by hand would be
  error-prone and unverifiable. It also asserts
  the labeling is honest: every answerable question's computed set is
  non-empty, and every deliberately-unanswerable question's set is empty —
  failing loudly if a question is mislabeled, not just relying on the
  manual spot-check below. It also flags any answerable question whose
  computed ground-truth set is disproportionately larger than `top_k` —
  the concrete size threshold (e.g. 3-4x `top_k`) is set once real set
  sizes are known from running the builder against the actual data, not
  guessed here
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
| 5 | `phase-5-eval` | `data/raw/condition_occurrence.csv`, `drug_exposure.csv`, `person.csv`, `measurement.csv`, `visit_occurrence.csv` (copied from Block 1, needed only here for ground truth), `scripts/concepts.py` (copied from Block 1, concept-ID mappings for the answer-key builder), `data/eval/questions.json`, `scripts/build_eval_answer_key.py`, `scripts/run_eval.py`, `docs/eval_results.md` (includes the parameter experiment) |
| 6 | `phase-6-verify-docs` | `scripts/run_all.py`, `scripts/verify.py`, `README.md` |
| 7 | `phase-7-metadata-filter` | `scripts/retrieve.py` (`build_metadata_filter()`), `scripts/api.py` (new `QueryRequest` fields), `tests/test_retrieve.py` |

Each phase gets its own PR (7 PRs total). `phase-1-spec` branches from
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
- Support for questions about conditions/drugs outside the 11-condition,
  17-drug whitelist established in Block 1 — those are expected to correctly
  produce "I don't know," not an error

## Known limitations

Documented consciously, not discovered later — full reasoning lives with
each design section, this is just the index:
- No automated grounding check on Claude's citations (see Generation design)
- 200-char chunking threshold is dataset-specific, not general-purpose (see Chunking design)
- No minimum eval score is enforced (see Eval harness design)
- Pinecone and Claude failures share one generic error response (see API design)
- Score threshold serves two jobs at once — relevance gating and
  out-of-scope rejection — which may want different cutoffs; templated
  summaries sharing clinical vocabulary make it possible for an
  out-of-scope question to still score above threshold against an
  unrelated in-scope patient. This is not just theoretical:
  `docs/eval_results.md`'s Run 2 shows it happening for real — lowering
  the unfiltered threshold to 0.4 caused one deliberately-unanswerable
  question (chronic kidney disease, outside the whitelist) to incorrectly
  clear the cutoff and skip the fallback. **Partially addressed in Phase
  7:** `scripts/retrieve.py`'s `select_threshold()` now uses a second,
  more permissive threshold (`PERMISSIVE_THRESHOLD = 0.2`) whenever a
  `condition` or `drug` metadata filter is present, since only those two
  filter types can structurally return zero candidates for an untracked
  topic (a `condition`/`drug` value outside the corpus simply matches no
  patient, filter or no filter) — `gender`/`lab`/`birth_decade` filters
  give no such protection and always keep the stricter default. Run 6
  (`docs/eval_results.md`) is the measured evidence: the same 0.2
  threshold raises filtered recall from 0.287 to 0.884 with fallback
  accuracy holding at 1.000, while the identical 0.2 threshold with no
  filter collapses fallback accuracy to 0.000. This is a scoped fix, not
  a complete one — an off-topic question paired with a condition/drug
  filter that happens to share vocabulary with a real, tracked condition
  (e.g. asking about "diabetes symptoms" while filtering on
  `condition="Diabetes mellitus type 2"`) could still slip through the
  permissive threshold, since the filter alone doesn't verify the
  question itself is genuinely about that condition.

## Functional requirements

Block 4 must:
1. Provide a connectivity smoke test (`scripts/check_connection.py`) that
   confirms `PINECONE_API_KEY` and `ANTHROPIC_API_KEY` are valid before any
   ingestion or eval work begins.
2. Chunk `graph_export.jsonl` records — records ≤ 200 characters stay
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
