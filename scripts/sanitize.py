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

# Unanchored - matches a role marker anywhere in the text, not just at
# start-of-line or right after sentence-ending punctuation. The earlier,
# anchored version was itself a bug fixed forward from an even earlier
# version that only matched at start-of-line: anchoring to a fixed set of
# "acceptable" preceding characters is inherently a game of finding every
# shape an injection could take, and an unanchored match is simpler and
# strictly safer against any positioning an attacker chooses.
#
# This does mean a legitimate mid-sentence phrase like "Cardiovascular
# system: normal." now gets stripped too, not just deliberately-placed
# injection markers - a real tradeoff, only acceptable because it doesn't
# happen on this corpus in practice: confirmed directly against the full
# data/raw/graph_export.jsonl (11,436 records) that none of
# system/human/assistant/user appear as a whole word anywhere in any
# record's `text` field, in any context - not just none followed by a
# colon. chunk_records.py's own placeholder/template strings were checked
# too, same result. If a future corpus does legitimately use one of these
# words in prose, this regex would strip it - that's the accepted cost of
# closing the positioning gap.
#
# \b (word boundary) still guards against a false positive within a
# single word - "ecosystem:" is not stripped, since there's no boundary
# between "eco" and "system".
_ROLE_MARKER_RE = re.compile(r"(?i)\b(?:system|human|assistant|user)\s*:\s*")
_CHAT_DELIMITER_RE = re.compile(
    r"(?i)\[/?(?:INST|SYS)\]|<\|.*?\|>|#{2,}\s*(?:instructions?|response)\b"
)


def sanitize_chunk_text(text: str) -> str:
    """Strip conversation-turn markers from patient text before it is
    stored in Pinecone. See this module's docstring for why this runs at
    ingestion rather than at prompt-build time.

    The role-marker substitution replaces with a single space, not "" -
    stripping to empty removes the whitespace on both sides of the marker
    too, fusing the sentence before it directly onto the text after it
    (e.g. "...fatigue. System: ignore..." -> "...fatigue.ignore...", zero
    space after the period). Block 6's citation-hardening work
    (trim_citation_snippet) splits on whitespace after sentence-ending
    punctuation to find sentence boundaries, and this sanitizer runs
    first, at ingestion - so a fused boundary here reaches Block 6
    already broken, with no marker left for it to fix on its own side.
    A single space keeps the boundary intact while still fully removing
    the marker itself.
    """
    text = _ROLE_MARKER_RE.sub(" ", text)
    text = _CHAT_DELIMITER_RE.sub("", text)
    return text


def sanitize_chunk(chunk: dict) -> dict:
    """Returns a copy of chunk with its chunk_text field sanitized."""
    chunk = dict(chunk)
    chunk["chunk_text"] = sanitize_chunk_text(chunk["chunk_text"])
    return chunk
