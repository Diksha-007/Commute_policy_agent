"""Thin wrapper around Chroma.

One collection per (company, version), named "<company>__<version>". There is no
shared collection holding several companies, so a query cannot reach another
company's chunks by forgetting a filter: it would have to open the wrong
collection by name, and names are only built from the authenticated company.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.chunking import Chunk
from app.embeddings import embed, embed_query

# Telemetry is disabled below, so nothing is ever sent. chromadb 0.6.3 still calls
# posthog with an outdated signature and logs a harmless TypeError; silence it.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def collection_name(company_id: str, version: str) -> str:
    return f"{company_id}__{version}"


class PolicyStore:
    def __init__(self, path: Path, embedding_function):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._ef = embedding_function

    def _get(self, name: str):
        return self._client.get_collection(name=name, embedding_function=self._ef)

    def list_collections(self) -> list[str]:
        cols = self._client.list_collections()
        # chromadb 0.6 returns names; older versions return objects.
        return [c if isinstance(c, str) else c.name for c in cols]

    def exists(self, name: str) -> bool:
        return name in self.list_collections()

    def create(self, name: str) -> None:
        # Cosine space: distance = 1 - cosine similarity.
        self._client.create_collection(
            name=name, embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, name: str, chunks: list[Chunk]) -> None:
        self._get(name).add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def count(self, name: str) -> int:
        return self._get(name).count()

    def all_metadata(self, name: str) -> list[dict]:
        return self._get(name).get(include=["metadatas"])["metadatas"]

    def rename(self, old: str, new: str) -> None:
        self._get(old).modify(name=new)

    def drop(self, name: str) -> None:
        """Delete a collection if it exists (idempotent)."""
        if self.exists(name):
            self._client.delete_collection(name=name)

    def query(self, name: str, text: str, n: int, *, as_query: bool = True) -> list[dict]:
        """Nearest chunks to `text`. Returns dicts with id, text, metadata, similarity
        and the chunk's embedding (needed by MMR re-ranking).

        as_query=True embeds `text` as a question (bge instruction prefix); False embeds it as a
        document, which is right for a HyDE passage (policy-style text)."""
        vector = (embed_query if as_query else embed)([text])[0].tolist()
        res = self._get(name).query(
            query_embeddings=[vector], n_results=n,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        embeddings = res["embeddings"][0]
        out = []
        for i, (cid, doc, meta, dist) in enumerate(zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )):
            out.append({"id": cid, "text": doc, "metadata": meta, "similarity": 1.0 - float(dist),
                        "embedding": np.asarray(embeddings[i], dtype=np.float32)})
        return out
