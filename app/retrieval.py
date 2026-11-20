"""Retrieval, locked to the caller's company and its ACTIVE policy version.

    retrieve(store, registry, principal, question, hyde_text=None)

1. company = principal.company_id (from the verified JWT; nothing else is accepted)
2. version = registry.active_version(company)
3. query ONLY the collection "<company>__<version>" with the question
   (and, when HyDE is on, also with the hypothetical passage)
4. choose the top k: Reciprocal Rank Fusion of the ranked lists (default), or MMR
   (use_mmr=True), which picks chunks that are relevant AND different from each other
5. assert every returned chunk belongs to (company, version); otherwise raise
   IsolationViolation. This guards against a routing bug or a tampered store.
   It runs over ALL candidates, before any re-ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from app.auth import Principal
from app.embeddings import embed, embed_query
from app.registry import Registry
from app.store import PolicyStore, collection_name


class IsolationViolation(Exception):
    """A chunk from another company or a stale version came back. Never answer."""


class NoActivePolicy(Exception):
    pass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    similarity: Optional[float]        # cosine similarity to the raw question (None if only HyDE found it)
    hyde_similarity: Optional[float]   # cosine similarity to the HyDE passage (None if HyDE off / not found)
    fused_score: float

    @property
    def body(self) -> str:
        """Chunk text without the breadcrumb line."""
        return self.text.split("\n", 1)[1] if "\n" in self.text else self.text


@dataclass(frozen=True)
class RetrievalResult:
    company_id: str
    version: str
    chunks: list[RetrievedChunk]       # top k, best first
    best_similarity: float             # best raw-question similarity (used by the not-covered gate)


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int) -> list[tuple[str, float]]:
    """score(chunk) = sum over lists of 1 / (k + rank). Ties keep first-seen order,
    so with a single list the order is unchanged."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit["id"], len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def mmr_select(relevance: list[float], embeddings: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance. Returns indices of up to k candidates, in pick order.

    Greedy: the first pick is the most relevant candidate. Every later pick maximises
        lam * relevance[i]  -  (1 - lam) * max cosine(embedding[i], embedding[j]) over picked j
    so a candidate that is almost a copy of one already chosen loses to a slightly
    less relevant candidate that adds something new. lam=1 is plain relevance order.
    `embeddings` must be L2-normalised, so a dot product is the cosine similarity.
    Ties go to the earlier candidate, so the result is deterministic.
    """
    n = len(relevance)
    if n == 0:
        return []
    sim = embeddings @ embeddings.T
    picked: list[int] = []
    remaining = list(range(n))
    while remaining and len(picked) < k:
        best_i, best_score = None, None
        for i in remaining:
            redundancy = max((float(sim[i, j]) for j in picked), default=0.0)
            score = lam * relevance[i] - (1.0 - lam) * redundancy
            if best_score is None or score > best_score + 1e-12:
                best_i, best_score = i, score
        picked.append(best_i)
        remaining.remove(best_i)
    return picked


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _check_isolation(hits: list[dict], company_id: str, version: str) -> None:
    for h in hits:
        meta = h.get("metadata") or {}
        if meta.get("company_id") != company_id or meta.get("version") != version:
            raise IsolationViolation(
                f"chunk {h.get('id')!r} ({meta.get('company_id')}/{meta.get('version')}) "
                f"returned for {company_id}/{version}"
            )


def retrieve(
    store: PolicyStore,
    registry: Registry,
    principal: Principal,
    question: str,
    *,
    hyde_text: Optional[str] = None,
    top_k: int = 3,
    candidates: int = 6,
    rrf_k: int = 60,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
    embed_fn: Callable[[list[str]], np.ndarray] = embed_query,
    embed_doc_fn: Callable[[list[str]], np.ndarray] = embed,
) -> RetrievalResult:
    company_id = principal.company_id
    version = registry.get_active_version(company_id)
    if version is None:
        raise NoActivePolicy(company_id)
    name = collection_name(company_id, version)

    question_hits = store.query(name, question, candidates)
    hyde_hits = store.query(name, hyde_text, candidates, as_query=False) if hyde_text else []
    # Check EVERY candidate, not only the ones we keep.
    _check_isolation(question_hits + hyde_hits, company_id, version)

    by_id = {h["id"]: h for h in hyde_hits}
    by_id.update({h["id"]: h for h in question_hits})
    q_sim = {h["id"]: h["similarity"] for h in question_hits}
    h_sim = {h["id"]: h["similarity"] for h in hyde_hits}

    if use_mmr and by_id:
        selected = _select_with_mmr(by_id, question, hyde_text, question_hits, top_k, mmr_lambda, embed_fn, embed_doc_fn)
    else:
        lists = [question_hits] + ([hyde_hits] if hyde_hits else [])
        selected = reciprocal_rank_fusion(lists, rrf_k)[:top_k]

    chunks = [
        RetrievedChunk(
            chunk_id=cid, text=by_id[cid]["text"], metadata=by_id[cid]["metadata"],
            similarity=q_sim.get(cid), hyde_similarity=h_sim.get(cid), fused_score=score,
        )
        for cid, score in selected
    ]
    best = max(q_sim.values(), default=0.0)
    return RetrievalResult(company_id, version, chunks, best)


def _select_with_mmr(by_id: dict, question: str, hyde_text: Optional[str], question_hits: list[dict],
                     top_k: int, lam: float, embed_fn, embed_doc_fn) -> list[tuple[str, float]]:
    """Run MMR over every candidate found (the question's and, if present, HyDE's).
    Relevance = cosine to the question (or to the HyDE passage, whichever is higher)."""
    ids = [h["id"] for h in question_hits] + [i for i in by_id if i not in {h["id"] for h in question_hits}]
    vectors = _unit(np.stack([by_id[i]["embedding"] for i in ids]))
    query_vectors = embed_fn([question])
    if hyde_text:                       # the passage is policy-style text: embedded as a document, as in the search
        query_vectors = np.vstack([query_vectors, embed_doc_fn([hyde_text])])
    relevance = (vectors @ query_vectors.T).max(axis=1)
    order = mmr_select(list(relevance), vectors, top_k, lam)
    return [(ids[i], float(relevance[i])) for i in order]
