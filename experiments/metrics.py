"""Retrieval metrics for the experiments (pure functions, no models).

Vocabulary
  fact group   one thing a question needs. A single-fact question has one group; a two-part question has two.
               Taken from expected.required_sections, or else expected.sections as a single group.
  relevant set for each group, the chunks that CONTAIN the fact. A chunk contains it when
               (a) its section is one of the group's sections AND
               (b) its text contains (boundary-aware) a distinctive string of a ledger fact labelled with
                   one of those sections.
  ranked       chunk ids, best first (the full ranking of one company's chunks)

Metrics
  recall@k     facts found in the top-k / facts needed.  A fact is found when ANY chunk of its relevant set is in the top-k.
  precision@k  relevant chunks in the top-k / k. Reported raw, and against its ceiling min(k, #relevant chunks),
               because a question with one relevant chunk can never score above 1/k.
  MRR          1 / rank of the first relevant chunk (0 if none within DEPTH).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from experiments.matching import contains_phrase, norm

DEPTH = 20  # MRR looks this far down the ranking


# ------------------------------------------------------------------ what a query needs

def need_groups(query: dict) -> list[list[str]]:
    """Sections that can carry each needed fact (one list per fact group)."""
    exp = query["expected"]
    return exp.get("required_sections") or [exp["sections"]]


def relevant_chunk_sets(query: dict, chunks: Sequence, facts: Sequence[dict],
                        norm_text: dict[str, str] | None = None) -> list[set[str]]:
    """For each fact group, the ids of the chunks that contain the fact (see the module docstring).
    `chunks` need .chunk_id, .text and .metadata['section_id']."""
    norm_text = norm_text or {c.chunk_id: norm(c.text) for c in chunks}
    by_id = {f["fact_id"]: f for f in facts if "fact_id" in f}
    fact_ids = query["expected"].get("fact_ids")      # parallel to the groups when the query names its own facts
    sets: list[set[str]] = []
    for i, group in enumerate(need_groups(query)):
        sections = set(group)
        if fact_ids:
            # Only this question's own facts count. (Without fact_ids, every ledger fact sharing a section would,
            # which lets one fact's text make a chunk look relevant to a different fact in a shared table.)
            strings = [s for fid in fact_ids[i] for s in by_id[fid]["distinctive_strings"]]
        else:
            strings = [s for f in facts if sections & set(f["section_ids"]) for s in f["distinctive_strings"]]
        sets.append({
            c.chunk_id for c in chunks
            if c.metadata["section_id"] in sections
            and any(contains_phrase(norm_text[c.chunk_id], s) for s in strings)
        })
    return sets


# ------------------------------------------------------------------ ranking metrics

def recall_at_k(ranked: Sequence[str], rel_sets: Sequence[set[str]], k: int) -> float:
    top = set(ranked[:k])
    return sum(1 for s in rel_sets if top & s) / len(rel_sets)


def precision_at_k(ranked: Sequence[str], rel_sets: Sequence[set[str]], k: int) -> tuple[float, float]:
    """(raw, relative to the ceiling). Raw = hits / k; relative = hits / min(k, #relevant chunks)."""
    union = set().union(*rel_sets)
    hits = sum(1 for c in ranked[:k] if c in union)
    return hits / k, (hits / min(k, len(union)) if union else 0.0)


def reciprocal_rank(ranked: Sequence[str], rel_sets: Sequence[set[str]], depth: int = DEPTH) -> float:
    union = set().union(*rel_sets)
    for i, c in enumerate(ranked[:depth], start=1):
        if c in union:
            return 1.0 / i
    return 0.0


# ------------------------------------------------------------------ statistics

def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    if not len(values):
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float, float]:
    """Mean and a percentile-bootstrap confidence interval (resampling queries with replacement)."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(arr.mean()), float(lo), float(hi)


def cluster_bootstrap_mean_ci(values: Sequence[float], clusters: Sequence[str], n_boot: int = 10000,
                              seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Like bootstrap_mean_ci, but resamples whole CLUSTERS (e.g. all rewordings of one base question) with
    replacement. Rewordings of one question share their answer chunks, so they are not independent; resampling
    them one by one makes the interval too narrow."""
    vals = np.asarray(values, dtype=float)
    labels = list(dict.fromkeys(clusters))
    index = {c: i for i, c in enumerate(labels)}
    sums, counts = np.zeros(len(labels)), np.zeros(len(labels))
    for v, c in zip(vals, clusters):
        sums[index[c]] += v
        counts[index[c]] += 1
    draws = np.random.default_rng(seed).integers(0, len(labels), size=(n_boot, len(labels)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(vals.mean()), float(lo), float(hi)


def win_tie_loss(a: Iterable[float], b: Iterable[float], tol: float = 1e-9) -> tuple[int, int, int]:
    """How often b beats / ties / loses to a, query by query."""
    wins = ties = losses = 0
    for x, y in zip(a, b):
        d = y - x
        wins += d > tol
        losses += d < -tol
        ties += abs(d) <= tol
    return wins, ties, losses


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    total = 0.0
    for p in positives:
        for n in negatives:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(positives) * len(negatives))
