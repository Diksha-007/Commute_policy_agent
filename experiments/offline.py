"""Shared pieces for the OFFLINE retrieval experiments (E2, E3): no app changes, no answer-LLM calls.

  * cached bge-base vectors for every chunk and question (computed once, reused)
  * the four ways of choosing which chunks the LLM will see (configs A-D), with or without HyDE
  * per-selection metrics, including the diversity ones (distinct sections, wasted slots)

Search is exact cosine over one company's chunks. The selection code reuses the app's own functions
(mmr_select, reciprocal_rank_fusion), so what is measured here is what the pipeline would do.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from app.retrieval import mmr_select, reciprocal_rank_fusion
from experiments import metrics as M
from experiments.matching import contains_phrase, norm
from experiments.run_e1 import COMPANIES, ROOT, load_company, section_of  # noqa: F401  (re-exported)

CACHE = ROOT / "experiments" / "cache"
RRF_K = 60        # same as app/config.py
PLAIN_POOL = 6    # candidates per ranked list for plain rank fusion (app/config.py: candidates_per_query)


# ------------------------------------------------------------------ configurations

@dataclass(frozen=True)
class Config:
    key: str
    label: str
    k: int            # chunks shown to the LLM
    mmr: bool         # choose them with MMR instead of plain relevance order
    pool: int         # MMR candidate pool
    lam: float = 1.0  # MMR lambda (1.0 = plain relevance)

    def with_lam(self, lam: float) -> "Config":
        return replace(self, lam=lam)


CONFIGS = {
    "A": Config("A", "A: top-3", 3, False, PLAIN_POOL),
    "B": Config("B", "B: top-3 + MMR", 3, True, 10, 0.7),
    "C": Config("C", "C: top-5", 5, False, PLAIN_POOL),
    "D": Config("D", "D: top-5 + MMR", 5, True, 15, 0.7),
}
LAMBDA_GRID = (0.5, 0.7, 0.85)


# ------------------------------------------------------------------ selection

def select(cfg: Config, D: np.ndarray, q: np.ndarray, h: np.ndarray | None = None, mode: str = "none") -> list[int]:
    """Indices of the chunks shown to the LLM, in the order shown.
    D: (n, d) unit chunk vectors. q: question vector. h: HyDE passage vector.
    mode: 'none' (question only), 'fused' (question + passage, as the app does), 'only' (passage only)."""
    if mode == "fused" and h is not None:
        sims_q, sims_h = D @ q, D @ h
        rq = np.argsort(-sims_q, kind="stable")
        rh = np.argsort(-sims_h, kind="stable")
        if not cfg.mmr:
            n = max(PLAIN_POOL, cfg.k)
            fused = reciprocal_rank_fusion(
                [[{"id": int(i)} for i in rq[:n]], [{"id": int(i)} for i in rh[:n]]], RRF_K)
            return [cid for cid, _ in fused[: cfg.k]]
        ids = list(dict.fromkeys([int(i) for i in rq[: cfg.pool]] + [int(i) for i in rh[: cfg.pool]]))
        relevance = [float(max(sims_q[i], sims_h[i])) for i in ids]   # as app/retrieval.py: best of the two vectors
        picks = mmr_select(relevance, D[ids], cfg.k, cfg.lam)
        return [ids[i] for i in picks]

    sims = D @ (h if (mode == "only" and h is not None) else q)
    order = np.argsort(-sims, kind="stable")
    if not cfg.mmr:
        return [int(i) for i in order[: cfg.k]]
    pool = order[: cfg.pool]
    picks = mmr_select([float(s) for s in sims[pool]], D[pool], cfg.k, cfg.lam)
    return [int(pool[i]) for i in picks]


# ------------------------------------------------------------------ metrics of one selection

def wasted_slots(shown: Sequence[str], rel_sets: Sequence[set[str]]) -> int:
    """Slots that add nothing: the chunk contains no needed fact, or only facts an earlier shown chunk already covered."""
    covered: set[int] = set()
    wasted = 0
    for c in shown:
        new = {g for g, s in enumerate(rel_sets) if c in s and g not in covered}
        if new:
            covered |= new
        else:
            wasted += 1
    return wasted


def score_selection(shown: Sequence[str], rel_sets: Sequence[set[str]], section_of_id: dict[str, str],
                    text_len: dict[str, int]) -> dict:
    k = len(shown)
    raw, vs_ceiling = M.precision_at_k(shown, rel_sets, k)
    return {
        "recall": M.recall_at_k(shown, rel_sets, k),
        "precision": raw, "precision_vs_ceiling": vs_ceiling,
        "rr": M.reciprocal_rank(shown, rel_sets, depth=k),
        "distinct_sections": len({section_of_id[c] for c in shown}),
        "wasted_slots": wasted_slots(shown, rel_sets),
        "context_chars": sum(text_len[c] for c in shown),
    }


# ------------------------------------------------------------------ vectors (cached)

def get_vectors(spec, corpora: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """({company: chunk vectors}, {query id: question vector}); computed once per model, then cached on disk."""
    signature = hashlib.sha256(json.dumps(
        [spec.key, [co["ids"] for co in corpora], [[q["id"], q["question"]] for co in corpora for q in co["queries"]]],
        ensure_ascii=False).encode("utf-8")).hexdigest()
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{spec.key}_{signature[:10]}_vectors.npz"     # one file per set of chunks and queries
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if str(data["signature"]) == signature:
            return ({co["company"]: data[f"D_{co['company']}"] for co in corpora},
                    {q["id"]: data[f"Q_{q['id']}"] for co in corpora for q in co["queries"]})
    D = {co["company"]: spec.docs(co["texts"]) for co in corpora}
    Q = {q["id"]: spec.query(q["question"]) for co in corpora for q in co["queries"]}
    np.savez_compressed(path, signature=np.array(signature),
                        **{f"D_{c}": v for c, v in D.items()}, **{f"Q_{i}": v for i, v in Q.items()})
    return D, Q


# ------------------------------------------------------------------ HyDE helpers

_NUMBER = re.compile(
    r"₹\s?\d[\d,]*(?:\.\d+)?"                                                    # amounts
    r"|\d{1,2}:\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm)\b"                                # times
    r"|\d+(?:\.\d+)?\s?(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|km|metres?|meters?|%|per ?cent|percent)\b",
    re.IGNORECASE)


def invented_numbers(passage: str, question: str, policy_norm: str) -> list[str]:
    """Numbers, times and amounts in a HyDE passage that appear neither in the question nor anywhere in the policy."""
    q = norm(question)
    out = []
    for m in _NUMBER.finditer(passage):
        token = m.group(0).strip()
        if not contains_phrase(q, token) and not contains_phrase(policy_norm, token):
            out.append(token)
    return out
