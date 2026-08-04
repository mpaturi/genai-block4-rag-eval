"""Confirms PINECONE_QUERY_API_KEY is actually restricted to query access -
not just documented as scoped to DataPlaneViewer in the Pinecone console,
but verified by making a real write call with it and watching Pinecone
itself reject it.

This is a live-credential integration test, not a unit test - it makes a
real network call against the real Pinecone service using the real
query-scoped key, and needs PINECONE_QUERY_API_KEY / PINECONE_INDEX_NAME
set in the environment (see .env.example). It intentionally does NOT run
by default in CI: CI generally shouldn't hold live cloud credentials for a
test whose entire point is "prove a credential can't do something
destructive" - a misconfigured or leaked CI secret here is exactly the
failure mode this key-scoping change exists to limit the blast radius of.
It's meant to be run manually, locally, after creating the scoped key in
the Pinecone console (see docs/spec.md's Key scoping note) and populating
.env - e.g. `pytest tests/test_pinecone_key_scope.py -v` - or in a CI job
that deliberately opts in with a secret scoped for this purpose only, kept
separate from the default test run.
"""
import os

import pytest
from pinecone import Pinecone
from pinecone.exceptions import ForbiddenException, PineconeApiException

from scripts.retrieve import NAMESPACE

pytestmark = pytest.mark.skipif(
    not os.environ.get("PINECONE_QUERY_API_KEY") or not os.environ.get("PINECONE_INDEX_NAME"),
    reason="requires live PINECONE_QUERY_API_KEY and PINECONE_INDEX_NAME",
)


def test_query_scoped_key_cannot_upsert():
    api_key = os.environ["PINECONE_QUERY_API_KEY"]
    index_name = os.environ["PINECONE_INDEX_NAME"]

    pc = Pinecone(api_key=api_key)
    index = pc.Index(name=index_name)

    with pytest.raises((ForbiddenException, PineconeApiException)) as exc_info:
        index.upsert_records(
            records=[{"_id": "phase11_scope_test", "chunk_text": "should be rejected"}],
            namespace=NAMESPACE,
        )
    assert exc_info.value.status == 403


def test_query_scoped_key_cannot_delete():
    api_key = os.environ["PINECONE_QUERY_API_KEY"]
    index_name = os.environ["PINECONE_INDEX_NAME"]

    pc = Pinecone(api_key=api_key)
    index = pc.Index(name=index_name)

    with pytest.raises((ForbiddenException, PineconeApiException)) as exc_info:
        index.delete(ids=["phase11_scope_test"], namespace=NAMESPACE)
    assert exc_info.value.status == 403
