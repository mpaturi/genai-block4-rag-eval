"""Prompt-injection defense for chunk_text - decision: sanitize here, called
from ingest.py at ingestion time, not from generate.py right before the
prompt is built.

chunk_text originates from patient note text and flows verbatim into two
places: generate.py's LLM prompt (generate_answer() joins every retrieved
chunk's chunk_text straight into the user message) and api.py's /query
response (`sources[].chunk_text`, added for Block 6 citation support).
Sanitizing once here, at the single point every chunk is written to
Pinecone (see ingest.py's main()), cleans both consumers from one place.
Sanitizing only inside generate.py's prompt-building step instead would
leave the citation text returned to callers unsanitized - since chunk_text
already reaches Pinecone and comes back out through /query regardless of
whether Claude is ever called, ingestion is the point that actually covers
every reader. This makes ingestion the one authoritative layer; a later
Block 6 phase testing citation safety is confirming that existing
guarantee, not expected to add its own redundant sanitization pass.

Kept in its own module, not inline in ingest.py: ingest.py uses bare
sibling imports (`from chunk_records import ...`) meant for direct
execution (`python scripts/ingest.py`), which breaks if ingest.py itself
is imported as `scripts.ingest` (e.g. from a test). This module has no
sibling imports, so it can be imported either way.

Rather than trying to blocklist injection phrasing (trivially bypassed by
rewording, and gives false confidence it "caught" prompt injection), this
strips the structural markers an injection needs to fake a new
conversation turn: role prefixes ("System:", "Human:", "Assistant:",
"User:") and chat-template delimiter tokens ("[INST]", "<|...|>",
"### Instruction"). generate.py's SYSTEM_PROMPT already tells Claude to
treat excerpts as data, not instructions - this is a second, code-level
layer that does not depend on the model actually obeying that
instruction.
"""
import re

# Matches a role marker at start-of-line (^, with (?m) - kept in case a
# newline ever does appear) OR right after sentence-ending punctuation
# (./!/?) - the realistic shape for this corpus, since real chunk_text
# (chunk_records.py's _make_text() output) never contains a newline at
# all (confirmed against the full 11,436-record source file, not just a
# sample). The lookbehind is zero-width, so the punctuation itself is
# never consumed/stripped - only the role marker and colon are removed.
# A plain mid-sentence word immediately followed by a colon (e.g.
# "Cardiovascular system: normal.") is NOT matched, since neither
# alternative is satisfied there - "system" isn't at start-of-line, and
# the character immediately before it (through any run of whitespace) is
# an ordinary word character, not one of .!?.
_ROLE_MARKER_RE = re.compile(
    r"(?im)(?:^|(?<=[.!?]))\s*(system|human|assistant|user)\s*:\s*"
)
_CHAT_DELIMITER_RE = re.compile(
    r"(?i)\[/?(?:INST|SYS)\]|<\|.*?\|>|#{2,}\s*(?:instructions?|response)\b"
)


def sanitize_chunk_text(text: str) -> str:
    """Strip conversation-turn markers from patient text before it is
    stored in Pinecone. See this module's docstring for why this runs at
    ingestion rather than at prompt-build time."""
    text = _ROLE_MARKER_RE.sub("", text)
    text = _CHAT_DELIMITER_RE.sub("", text)
    return text


def sanitize_chunk(chunk: dict) -> dict:
    """Returns a copy of chunk with its chunk_text field sanitized."""
    chunk = dict(chunk)
    chunk["chunk_text"] = sanitize_chunk_text(chunk["chunk_text"])
    return chunk
