"""Version registry (SQLite): which policy versions exist, and which one is ACTIVE.

The `active_version` table is the single switch that decides what answers.
Readers always resolve company -> active version here first, and only then open
that version's Chroma collection. Flipping the pointer is one SQLite
transaction, so readers see either the old version or the new one, never a mix.

Document status lifecycle:
    staging --(verified + pointer flipped)--> active --(newer version flipped)--> retired
    staging --(any error / crash recovery)--> failed
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    company_id     TEXT NOT NULL,
    version        TEXT NOT NULL,
    doc_name       TEXT NOT NULL,
    effective_date TEXT,
    status         TEXT NOT NULL CHECK (status IN ('staging', 'active', 'retired', 'failed')),
    chunk_count    INTEGER,
    content_sha256 TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (company_id, version)
);
CREATE TABLE IF NOT EXISTS active_version (
    company_id TEXT PRIMARY KEY,
    version    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class RegistryError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A new connection per call keeps this safe across threads; it is always closed.
        # isolation_level=None: we issue BEGIN/COMMIT ourselves where it matters.
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------ reads

    def get_active_version(self, company_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM active_version WHERE company_id = ?", (company_id,)
            ).fetchone()
        return row["version"] if row else None

    def get_document(self, company_id: str, version: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE company_id = ? AND version = ?", (company_id, version)
            ).fetchone()
        return dict(row) if row else None

    def documents_with_status(self, status: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]

    def active_versions(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT company_id, version FROM active_version").fetchall()
        return {r["company_id"]: r["version"] for r in rows}

    # ------------------------------------------------------------ writes

    def insert_staging(self, company_id: str, version: str, doc_name: str,
                       effective_date: str, sha256: str) -> None:
        """Record a new version as 'staging'. A label that previously FAILED may
        be retried; a label that was ever staging/active/retired may not."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT status FROM documents WHERE company_id = ? AND version = ?",
                    (company_id, version),
                ).fetchone()
                if row and row["status"] != "failed":
                    raise RegistryError(f"{company_id} {version} already exists with status {row['status']}")
                conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(company_id, version, doc_name, effective_date, status, chunk_count, content_sha256, created_at) "
                    "VALUES (?, ?, ?, ?, 'staging', NULL, ?, ?)",
                    (company_id, version, doc_name, effective_date, sha256, _now()),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def mark_failed(self, company_id: str, version: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'failed' WHERE company_id = ? AND version = ? AND status = 'staging'",
                (company_id, version),
            )

    def activate(self, company_id: str, version: str, chunk_count: int) -> Optional[str]:
        """THE switch point. In one transaction: retire the current active version,
        mark the staged version active, and move the pointer. Returns the
        previously active version (or None). On any error nothing changes."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT version FROM active_version WHERE company_id = ?", (company_id,)
                ).fetchone()
                previous = row["version"] if row else None

                conn.execute(
                    "UPDATE documents SET status = 'retired' WHERE company_id = ? AND status = 'active'",
                    (company_id,),
                )
                cur = conn.execute(
                    "UPDATE documents SET status = 'active', chunk_count = ? "
                    "WHERE company_id = ? AND version = ? AND status = 'staging'",
                    (chunk_count, company_id, version),
                )
                if cur.rowcount != 1:
                    raise RegistryError(f"{company_id} {version} is not in staging; refusing to activate")
                conn.execute(
                    "INSERT INTO active_version (company_id, version, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(company_id) DO UPDATE SET version = excluded.version, updated_at = excluded.updated_at",
                    (company_id, version, _now()),
                )
                conn.execute("COMMIT")
                return previous
            except Exception:
                conn.execute("ROLLBACK")
                raise
