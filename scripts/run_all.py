"""One-command setup: runs the full setup-through-verify pipeline in order
- check_connection -> create_index -> ingest (chunks + upserts) -> verify.
Mirrors Block 3's run_all.py. Does NOT run run_eval.py, which stays a
separate, deliberate command you run on purpose, possibly more than once
with different settings - not something that should fire automatically
every time you set up the project (see docs/spec.md's Architecture
section).

chunk_records.py has no step of its own here - it's a library ingest.py
already calls internally, not a standalone script with a main().

Run with: python scripts/run_all.py (from the repo root).
"""
import sys

import check_connection
import create_index
import ingest
import verify

STEPS = [
    ("check_connection", check_connection.main),
    ("create_index", create_index.main),
    ("ingest (chunk + upload)", ingest.main),
    ("verify", verify.main),
]


def main() -> int:
    for name, step in STEPS:
        print(f"\n=== {name} ===")
        exit_code = step()
        if exit_code != 0:
            print(f"\nFAIL - {name} failed (exit code {exit_code}). Stopping.")
            return 1

    print("\nrun_all.py complete - setup, ingestion, and verification all passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
