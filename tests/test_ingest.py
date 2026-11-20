"""Ingestion: versioning, atomic switch, failure injection, crash recovery."""
import threading
import time

import pytest

from app.chunking import IngestValidationError
from app.embeddings import embed
from app.ingest import Ingestor, IngestStorageError, VersionConflictError
from app.registry import RegistryError
from app.store import collection_name
from tests.conftest import policy_text


def staging_collections(store):
    return [n for n in store.list_collections() if "__staging_" in n]


def answers_from(store, registry, company, question):
    """What a reader would see: resolve the active version, query its collection."""
    version = registry.get_active_version(company)
    return store.query(collection_name(company, version), question, n=3)


# ------------------------------------------------------------- happy path

def test_first_ingest_activates_v1(ingestor, registry, store):
    result = ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    assert result.status == "activated" and result.previous_version is None
    assert registry.get_active_version("acme") == "v1"
    doc = registry.get_document("acme", "v1")
    assert doc["status"] == "active" and doc["chunk_count"] == result.chunk_count
    assert store.count("acme__v1") == result.chunk_count
    assert staging_collections(store) == []


def test_new_version_replaces_old_and_only_it_answers(ingestor, registry, store):
    ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    result = ingestor.ingest("acme", "v2", policy_text("acme", "v2"))

    assert result.previous_version == "v1"
    assert registry.get_active_version("acme") == "v2"
    assert registry.get_document("acme", "v1")["status"] == "retired"
    assert not store.exists("acme__v1")          # old chunks are gone (gc delay 0 in tests)

    hits = answers_from(store, registry, "acme", "What is the penalty for a second no-show?")
    assert all(h["metadata"]["version"] == "v2" for h in hits)
    assert any("₹200" in h["text"] for h in hits)
    assert not any("Written warning" in h["text"] for h in hits)   # the v1 rule


# ------------------------------------------------------------- failure injection

def test_failure_halfway_through_writing_leaves_old_version_answering(ingestor, registry, store, monkeypatch):
    ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    real_add = store.add_chunks

    def add_half_then_crash(name, chunks):
        real_add(name, chunks[: len(chunks) // 2])
        raise RuntimeError("simulated disk failure mid-write")

    monkeypatch.setattr(store, "add_chunks", add_half_then_crash)
    with pytest.raises(IngestStorageError, match="simulated disk failure"):
        ingestor.ingest("acme", "v2", policy_text("acme", "v2"))

    assert registry.get_active_version("acme") == "v1"
    assert registry.get_document("acme", "v1")["status"] == "active"
    assert registry.get_document("acme", "v2")["status"] == "failed"
    assert staging_collections(store) == [] and not store.exists("acme__v2")
    hits = answers_from(store, registry, "acme", "What is the penalty for a second no-show?")
    assert hits and all(h["metadata"]["version"] == "v1" for h in hits)

    # A failed label may be retried once the problem is fixed.
    monkeypatch.setattr(store, "add_chunks", real_add)
    assert ingestor.ingest("acme", "v2", policy_text("acme", "v2")).status == "activated"
    assert registry.get_active_version("acme") == "v2"


def test_verification_failure_aborts_without_switching(ingestor, registry, store, monkeypatch):
    ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    monkeypatch.setattr(store, "count", lambda name: 1)   # pretend chunks went missing
    with pytest.raises(IngestStorageError, match="verification failed"):
        ingestor.ingest("acme", "v2", policy_text("acme", "v2"))
    assert registry.get_active_version("acme") == "v1"
    assert store.exists("acme__v1") and not store.exists("acme__v2")
    assert staging_collections(store) == []


def test_failure_at_the_switch_itself_leaves_old_version(ingestor, registry, store, monkeypatch):
    ingestor.ingest("acme", "v1", policy_text("acme", "v1"))

    def broken_activate(*args, **kwargs):
        raise RegistryError("simulated database failure during commit")

    monkeypatch.setattr(registry, "activate", broken_activate)
    with pytest.raises(IngestStorageError):
        ingestor.ingest("acme", "v2", policy_text("acme", "v2"))
    assert registry.get_active_version("acme") == "v1"
    assert not store.exists("acme__v2")          # renamed collection cleaned up too
    assert registry.get_document("acme", "v2")["status"] == "failed"


def test_invalid_document_writes_nothing(ingestor, registry, store):
    with pytest.raises(IngestValidationError, match="labelled company 'globex'"):
        ingestor.ingest("acme", "v1", policy_text("globex", "v1"))
    assert registry.get_document("acme", "v1") is None
    assert registry.get_active_version("acme") is None
    assert store.list_collections() == []


# ------------------------------------------------------------- crash recovery

def test_recover_cleans_up_after_a_crash(ingestor, registry, store):
    ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    # Simulate a process that died mid-ingest of v2, plus a retired version whose GC never ran.
    registry.insert_staging("acme", "v2", "Acme", "2026-10-01", "sha")
    store.create("acme__v2__staging_deadbeef")
    store.create("globex__v0")

    dropped = ingestor.recover()

    assert set(dropped) == {"acme__v2__staging_deadbeef", "globex__v0"}
    assert registry.get_document("acme", "v2")["status"] == "failed"
    assert registry.get_active_version("acme") == "v1" and store.exists("acme__v1")


# ------------------------------------------------------------- idempotency and labels

def test_same_content_reingested_is_a_no_op(ingestor, registry):
    ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    again = ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    assert again.status == "already_active" and again.version == "v1"


def test_version_label_cannot_be_reused(ingestor):
    ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    ingestor.ingest("acme", "v2", policy_text("acme", "v2"))
    modified_v1 = policy_text("acme", "v1").replace("4 hours", "5 hours")
    with pytest.raises(VersionConflictError, match="already exists"):
        ingestor.ingest("acme", "v1", modified_v1)     # v1 is retired; label is burnt


# ------------------------------------------------------------- grace period and concurrency

def test_old_version_dropped_only_after_grace_period(registry, store):
    slow_gc = Ingestor(registry, store, embed, gc_delay_s=0.5)
    slow_gc.ingest("acme", "v1", policy_text("acme", "v1"))
    slow_gc.ingest("acme", "v2", policy_text("acme", "v2"))
    assert store.exists("acme__v1")              # in-flight readers can still finish
    deadline = time.time() + 5
    while store.exists("acme__v1") and time.time() < deadline:
        time.sleep(0.1)
    assert not store.exists("acme__v1")


