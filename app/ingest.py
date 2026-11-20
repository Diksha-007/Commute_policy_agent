"""Atomic, versioned policy ingestion.

    Phase A  prepare   parse + validate + chunk (pure), record 'staging' row,
                       write chunks into a uniquely named staging collection,
                       verify, rename to "<company>__<version>".
                       Readers cannot see any of this: the pointer is unchanged.
    Phase B  commit    ONE SQLite transaction flips active_version.
    Phase C  cleanup   drop the previous version's collection after a grace period.

If anything in Phase A or B fails, the staging data is removed, the row is marked
'failed', and the previously active version keeps answering, untouched.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

from app.chunking import IngestValidationError, chunk_document
from app.embeddings import EmbedFn
from app.registry import Registry, RegistryError
from app.store import PolicyStore, collection_name

log = logging.getLogger(__name__)


class VersionConflictError(Exception):
    """The version label was already used for this company."""


class IngestStorageError(Exception):
    """Writing or verifying the new version failed; the old version is untouched."""


@dataclass(frozen=True)
class IngestResult:
    status: str                      # "activated" | "already_active"
    company_id: str
    version: str
    chunk_count: int
    previous_version: Optional[str]


class Ingestor:
    def __init__(
        self,
        registry: Registry,
        store: PolicyStore,
        embed_fn: EmbedFn,
        *,
        gc_delay_s: float = 30.0,
        breakpoint_percentile: float = 85.0,
        min_chars: int = 150,
        max_chars: int = 900,
        rows_per_chunk: int = 4,
    ):
        self.registry = registry
        self.store = store
        self.embed_fn = embed_fn
        self.gc_delay_s = gc_delay_s
        self._chunk_opts = dict(
            breakpoint_percentile=breakpoint_percentile, min_chars=min_chars,
            max_chars=max_chars, rows_per_chunk=rows_per_chunk,
        )
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, company_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(company_id, threading.Lock())

    # ------------------------------------------------------------ ingest

    def ingest(self, company_id: str, version: str, markdown: str) -> IngestResult:
        sha = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

        with self._lock_for(company_id):   # one ingestion at a time per company
            # Idempotency: the same content is already live -> nothing to do.
            active = self.registry.get_active_version(company_id)
            if active:
                active_doc = self.registry.get_document(company_id, active)
                if active_doc and active_doc["content_sha256"] == sha:
                    return IngestResult("already_active", company_id, active,
                                        active_doc["chunk_count"] or 0, None)

            # Phase A.1: pure. Raises IngestValidationError; nothing written yet.
            # Runs before the label check so a wrong-company document is reported
            # as such, rather than as a version clash.
            header, chunks = chunk_document(
                markdown, expected_company=company_id, expected_version=version,
                embed_fn=self.embed_fn, **self._chunk_opts,
            )

            existing = self.registry.get_document(company_id, version)
            if existing and existing["status"] != "failed":
                raise VersionConflictError(
                    f"version {version!r} of {company_id!r} already exists ({existing['status']}); "
                    "upload changes under a new version label"
                )

            try:
                self.registry.insert_staging(company_id, version, header.doc_name,
                                             header.effective_date, sha)
            except RegistryError as e:
                raise VersionConflictError(str(e)) from e

            staging = f"{company_id}__{version}__staging_{uuid.uuid4().hex[:8]}"
            final = collection_name(company_id, version)
            try:
                # Phase A.2: write + verify in a collection nobody reads.
                self.store.create(staging)
                self.store.add_chunks(staging, chunks)
                self._verify(staging, company_id, version, chunks)
                # A leftover with the final name can only come from an earlier
                # failed attempt (the label is not active/retired, checked above).
                self.store.drop(final)
                self.store.rename(staging, final)

                # Phase B: the single switch point.
                previous = self.registry.activate(company_id, version, len(chunks))
            except Exception as e:
                log.error("ingest failed for %s %s: %s", company_id, version, e)
                self._safe_drop(staging)
                if self.registry.get_active_version(company_id) != version:
                    self._safe_drop(final)
                self.registry.mark_failed(company_id, version)
                if isinstance(e, IngestValidationError):
                    raise
                raise IngestStorageError(f"ingestion of {company_id} {version} failed: {e}") from e

        # Phase C: outside the lock; failure here is harmless (retired = unreachable).
        if previous and previous != version:
            self._schedule_gc(collection_name(company_id, previous))
        log.info("activated %s %s (%d chunks, previous=%s)", company_id, version, len(chunks), previous)
        return IngestResult("activated", company_id, version, len(chunks), previous)

    def _verify(self, name: str, company_id: str, version: str, chunks) -> None:
        count = self.store.count(name)
        if count != len(chunks):
            raise IngestStorageError(f"verification failed: stored {count} chunks, expected {len(chunks)}")
        for meta in self.store.all_metadata(name):
            if meta.get("company_id") != company_id or meta.get("version") != version:
                raise IngestStorageError(f"verification failed: foreign metadata {meta}")
        probe = chunks[0].metadata["section_title"]
        if not self.store.query(name, probe, n=1):
            raise IngestStorageError("verification failed: smoke query returned nothing")

    # ------------------------------------------------------------ cleanup

    def _safe_drop(self, name: str) -> None:
        try:
            self.store.drop(name)
        except Exception as e:  # cleanup must never mask the original error
            log.warning("could not drop collection %s: %s", name, e)

    def _schedule_gc(self, name: str) -> None:
        """Drop an old version's collection after a grace period, so a request that
        resolved the old version just before the flip can still finish."""
        if self.gc_delay_s <= 0:
            self._safe_drop(name)
            return
        t = threading.Timer(self.gc_delay_s, self._safe_drop, args=(name,))
        t.daemon = True
        t.start()

    def recover(self) -> list[str]:
        """Run at startup, before serving. Any 'staging' row means a crash mid-ingest:
        mark it failed. Any collection that is not an ACTIVE version (staging
        leftovers, retired versions whose GC never ran) is dropped.
        Returns the names of the dropped collections."""
        for doc in self.registry.documents_with_status("staging"):
            self.registry.mark_failed(doc["company_id"], doc["version"])
        keep = {collection_name(c, v) for c, v in self.registry.active_versions().items()}
        dropped = [name for name in self.store.list_collections() if name not in keep]
        for name in dropped:
            self._safe_drop(name)
        return dropped
