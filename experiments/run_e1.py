"""E1: which embedding model retrieves the answer chunk best, and at what cost?

    python experiments/run_e1.py --models minilm bge-base

Held fixed: the chunks (made once with the production chunker), the queries and the labels.
Varied: only the embedding model. Search is EXACT cosine over all of one company's chunks (no approximate
index), so differences come from the embedder alone.

Writes experiments/results/e1_results.json, e1_full_report.md and e1_per_query.jsonl.
(E1_results.md next to them is the short human-written summary.)
"""
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_document  # noqa: E402
from app.embeddings import minilm_embed as production_embed  # noqa: E402
from experiments import metrics as M  # noqa: E402
from experiments.matching import norm  # noqa: E402

DATA = ROOT / "experiments" / "data"
RESULTS = ROOT / "experiments" / "results"
COMPANIES = [("acme", "v2"), ("globex", "v1"), ("initech", "v1")]
FASTEMBED_CACHE = os.environ.get("FASTEMBED_CACHE_PATH", "/models")   # the app image has bge-base here
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "  # recommended for bge queries
STYLE = {"original": "original", "paraphrase-1": "keyword", "paraphrase-2": "chatty", "paraphrase-3": "synonyms"}
KS = (1, 3, 5)
PROJECTED_COMPANIES = 400  # the scale named in the case study
CURRENT_CUTOFF = 0.30      # the app's "not covered" similarity cutoff (app/config.py), tuned for MiniLM


# ------------------------------------------------------------------ models

def unit(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) / 1e6


@dataclass
class ModelSpec:
    key: str
    label: str
    dim: int
    docs: Callable[[list], np.ndarray]    # texts -> (n, dim) unit vectors
    query: Callable[[str], np.ndarray]    # text -> (dim,) unit vector
    load_seconds: float                   # cold start: construct + first embedding
    size_mb: float                        # model files on disk


def build_minilm() -> ModelSpec:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    start = time.perf_counter()
    ef = DefaultEmbeddingFunction()
    ef(["warm up"])
    load = time.perf_counter() - start
    return ModelSpec(
        "minilm", "all-MiniLM-L6-v2 (Chroma ONNX, used by the app today)", 384,
        docs=lambda texts: unit(ef(list(texts))),
        query=lambda text: unit(ef([text]))[0],
        load_seconds=load,
        # the weights only (the cache folder also keeps the downloaded archive)
        size_mb=sum(f.stat().st_size for f in (Path.home() / ".cache" / "chroma" / "onnx_models").rglob("model.onnx")) / 1e6,
    )


def build_fastembed(key: str, hf_name: str, dim: int, prefix: str, local_only: bool = True) -> ModelSpec:
    from fastembed import TextEmbedding

    def field(meta, name):
        return meta.get(name) if isinstance(meta, dict) else getattr(meta, name, None)

    size_gb = next((field(m, "size_in_GB") for m in TextEmbedding.list_supported_models() if field(m, "model") == hf_name), None)
    start = time.perf_counter()
    try:   # no network during the cold-start measurement, when this fastembed supports it
        model = TextEmbedding(model_name=hf_name, cache_dir=FASTEMBED_CACHE, local_files_only=local_only)
    except TypeError:
        model = TextEmbedding(model_name=hf_name, cache_dir=FASTEMBED_CACHE)
    list(model.embed(["warm up"]))
    load = time.perf_counter() - start
    return ModelSpec(
        key, f"{hf_name} (fastembed ONNX)", dim,
        docs=lambda texts: unit(list(model.embed(list(texts), batch_size=32))),
        query=lambda text: unit(list(model.embed([prefix + text])))[0],
        load_seconds=load, size_mb=(size_gb or 0) * 1000,
    )


BUILDERS = {
    "minilm": build_minilm,
    "bge-small": lambda: build_fastembed("bge-small", "BAAI/bge-small-en-v1.5", 384, BGE_QUERY_INSTRUCTION),
    "bge-base": lambda: build_fastembed("bge-base", "BAAI/bge-base-en-v1.5", 768, BGE_QUERY_INSTRUCTION),
}


# ------------------------------------------------------------------ data

def load_company(company: str, version: str, extended: bool = False) -> dict:
    """extended=True adds the extra (4x) queries and their facts to the core set."""
    stem = f"{company}_{version}"
    markdown = (DATA / f"{stem}_long.md").read_text(encoding="utf-8")
    _, chunks = chunk_document(markdown, expected_company=company, expected_version=version,
                               embed_fn=production_embed)
    queries = [json.loads(line) for line in (DATA / f"{stem}_queries.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    facts = json.loads((DATA / f"{stem}_facts.json").read_text(encoding="utf-8"))
    if extended:
        queries += [json.loads(line) for line in (DATA / f"{stem}_queries_extra.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        facts += json.loads((DATA / f"{stem}_facts_extra.json").read_text(encoding="utf-8"))
    norm_text = {c.chunk_id: norm(c.text) for c in chunks}
    for q in queries:
        if q["expected"]["status"] == "answered":
            q["_rel"] = M.relevant_chunk_sets(q, chunks, facts, norm_text)
            empty = [i for i, s in enumerate(q["_rel"]) if not s]
            if empty:
                raise SystemExit(f"{q['id']}: fact group(s) {empty} have no chunk containing them (label problem)")
    return {"company": company, "chunks": chunks, "ids": [c.chunk_id for c in chunks],
            "texts": [c.text for c in chunks], "queries": queries}


def section_of(chunk_id: str) -> str:
    return chunk_id.split("__")[2]


# ------------------------------------------------------------------ evaluation

def evaluate(spec: ModelSpec, corpora: list[dict]) -> tuple[list[dict], dict]:
    rows, embed_seconds, n_chunks = [], 0.0, 0
    for co in corpora:
        start = time.perf_counter()
        docs = spec.docs(co["texts"])
        embed_seconds += time.perf_counter() - start
        n_chunks += len(co["texts"])
        for q in co["queries"]:
            sims = docs @ spec.query(q["question"])
            order = np.argsort(-sims, kind="stable")
            ranked = [co["ids"][i] for i in order[: M.DEPTH + 5]]
            answerable = q["expected"]["status"] == "answered"
            row = {"model": spec.key, "id": q["id"], "company": co["company"], "base_id": q["base_id"],
                   "variant": q["variant"], "style": STYLE.get(q["variant"], q["variant"]),
                   "category": q["category"], "answerable": answerable, "best_sim": float(sims.max()),
                   "top5": [section_of(c) + ("*" if answerable and c in set().union(*q["_rel"]) else "") for c in ranked[:5]]}
            if answerable:
                rel = q["_rel"]
                row["n_groups"] = len(rel)
                for k in KS:
                    row[f"recall@{k}"] = M.recall_at_k(ranked, rel, k)
                row["precision@5"], row["precision@5_vs_ceiling"] = M.precision_at_k(ranked, rel, 5)
                row["rr"] = M.reciprocal_rank(ranked, rel)
            rows.append(row)
    return rows, {"embed_seconds": embed_seconds, "n_chunks": n_chunks}


def time_queries(spec: ModelSpec, questions: list[str], repeats: int = 5, warmup: int = 5) -> list[float]:
    for text in questions[:warmup]:
        spec.query(text)
    samples = []
    for _ in range(repeats):
        for text in questions:
            start = time.perf_counter()
            spec.query(text)
            samples.append((time.perf_counter() - start) * 1000)
    return samples


def mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if key in r]
    return float(np.mean(values)) if values else None


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    single = [r for r in ans if r["n_groups"] == 1]
    out = {
        "n_answerable": len(ans), "n_single_fact": len(single),
        "recall@1_single_fact": mean(single, "recall@1"),
        "recall@3": mean(ans, "recall@3"), "recall@5": mean(ans, "recall@5"),
        "mrr": mean(ans, "rr"),
        "precision@5": mean(ans, "precision@5"), "precision@5_vs_ceiling": mean(ans, "precision@5_vs_ceiling"),
    }
    by_style = {}
    for style in ("original", "keyword", "chatty", "synonyms"):
        sub = [r for r in ans if r["style"] == style]
        by_style[style] = {"n": len(sub),
                           "complete@3": float(np.mean([r["recall@3"] == 1.0 for r in sub])),
                           "complete@5": float(np.mean([r["recall@5"] == 1.0 for r in sub]))}
    para = [r for r in ans if r["style"] != "original"]
    by_style["all_paraphrases"] = {"n": len(para),
                                   "complete@3": float(np.mean([r["recall@3"] == 1.0 for r in para])),
                                   "complete@5": float(np.mean([r["recall@5"] == 1.0 for r in para]))}
    out["by_style"] = by_style
    out["paraphrase_drop@5"] = by_style["original"]["complete@5"] - by_style["all_paraphrases"]["complete@5"]
    out["by_category"] = {c: {"n": len(sub), "recall@3": mean(sub, "recall@3"), "recall@5": mean(sub, "recall@5")}
                          for c in sorted({r["category"] for r in ans}) for sub in [[r for r in ans if r["category"] == c]]}
    pos = [r["best_sim"] for r in ans]
    neg = [r["best_sim"] for r in rows if not r["answerable"]]
    out["gate"] = {"auc": M.auc(pos, neg), "mean_best_sim_answerable": float(np.mean(pos)),
                   "mean_best_sim_not_covered": float(np.mean(neg)), "min_answerable": min(pos), "max_not_covered": max(neg),
                   "min_not_covered": min(neg),
                   "not_covered_refused_by_current_cutoff": float(np.mean([x < CURRENT_CUTOFF for x in neg])),
                   "answerable_wrongly_refused_by_current_cutoff": float(np.mean([x < CURRENT_CUTOFF for x in pos]))}
    return out


def paired(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """b minus a, query by query, over the answerable queries."""
    a = {r["id"]: r for r in rows_a if r["answerable"]}
    b = {r["id"]: r for r in rows_b if r["answerable"]}
    ids = sorted(a)
    out = {}
    for key in ("recall@3", "recall@5", "rr", "precision@5_vs_ceiling"):
        xs, ys = [a[i][key] for i in ids], [b[i][key] for i in ids]
        mean_diff, lo, hi = M.bootstrap_mean_ci([y - x for x, y in zip(xs, ys)])
        wins, ties, losses = M.win_tie_loss(xs, ys)
        out[key] = {"mean_diff": mean_diff, "ci95": [lo, hi], "wins": wins, "ties": ties, "losses": losses}
    return out


# ------------------------------------------------------------------ report

def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render(results: dict, specs: list[ModelSpec]) -> str:
    keys = [s.key for s in specs]
    L = [f"# E1: embedding models ({', '.join(keys)})", ""]
    meta = results["meta"]
    L += [f"{meta['n_chunks']} chunks over 3 companies; {meta['n_queries']} queries "
          f"({meta['n_answerable']} answerable, {meta['n_not_covered']} not covered). Exact cosine search, chunks fixed.", ""]

    L += ["## Retrieval quality (answerable queries)", "",
          "| metric | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    s = results["summary"]
    for label, key in (("Recall@1 (single-fact queries)", "recall@1_single_fact"), ("Recall@3", "recall@3"),
                       ("Recall@5", "recall@5"), ("MRR", "mrr"), ("Precision@5 (raw)", "precision@5"),
                       ("Precision@5 (vs ceiling)", "precision@5_vs_ceiling")):
        L.append(f"| {label} | " + " | ".join(f"{s[k][key]:.3f}" for k in keys) + " |")

    L += ["", "## Paraphrase robustness (share of queries with ALL facts in the top 5 / top 3)", "",
          "| wording | n | " + " | ".join(f"{k} @5 | {k} @3" for k in keys) + " |", "|---|---|" + "---|---|" * len(keys)]
    for style in ("original", "keyword", "chatty", "synonyms", "all_paraphrases"):
        n = s[keys[0]]["by_style"][style]["n"]
        cells = " | ".join(f"{pct(s[k]['by_style'][style]['complete@5'])} | {pct(s[k]['by_style'][style]['complete@3'])}" for k in keys)
        L.append(f"| {style} | {n} | {cells} |")
    L.append("| drop, original -> paraphrases (@5) | | " + " | ".join(f"{100 * s[k]['paraphrase_drop@5']:+.1f} pts | " for k in keys) + " |")

    L += ["", "## By question type (Recall@3 / Recall@5)", "",
          "| type | n | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for cat in s[keys[0]]["by_category"]:
        n = s[keys[0]]["by_category"][cat]["n"]
        L.append(f"| {cat} | {n} | " + " | ".join(
            f"{s[k]['by_category'][cat]['recall@3']:.2f} / {s[k]['by_category'][cat]['recall@5']:.2f}" for k in keys) + " |")

    if results.get("paired"):
        a, b = keys[0], keys[1]
        L += ["", f"## Paired comparison: {b} minus {a} (per query, 95% bootstrap interval)", "",
              "| metric | mean difference | 95% interval | queries better / same / worse |", "|---|---|---|---|"]
        for key, v in results["paired"].items():
            lo, hi = v["ci95"]
            verdict = "" if lo <= 0 <= hi else " *"
            L.append(f"| {key} | {v['mean_diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}]{verdict} | {v['wins']} / {v['ties']} / {v['losses']} |")
        L.append("\n`*` the interval excludes zero. Otherwise the data cannot tell the models apart.")

    L += ["", "## Cost", "",
          "| | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    c = results["cost"]
    L.append("| dimensions | " + " | ".join(str(c[k]["dim"]) for k in keys) + " |")
    L.append("| query embedding p50 / p95 / mean (ms) | " + " | ".join(
        f"{c[k]['query_ms_p50']:.1f} / {c[k]['query_ms_p95']:.1f} / {c[k]['query_ms_mean']:.1f}" for k in keys) + " |")
    L.append("| cold start (s) | " + " | ".join(f"{c[k]['load_seconds']:.2f}" for k in keys) + " |")
    L.append(f"| embedding all {meta['n_chunks']} chunks (s) | " + " | ".join(f"{c[k]['embed_seconds']:.1f}" for k in keys) + " |")
    L.append("| model files on disk (MB) | " + " | ".join(f"{c[k]['size_mb']:.0f}" for k in keys) + " |")
    L.append("| vectors per company, average (MB) | " + " | ".join(f"{c[k]['index_mb_per_company']:.2f}" for k in keys) + " |")
    L.append(f"| projected vectors for {PROJECTED_COMPANIES} companies (MB) | " + " | ".join(f"{c[k]['projected_index_mb']:.0f}" for k in keys) + " |")

    L += ["", "## Not-covered cutoff: does the best similarity separate answerable from not-covered questions?", "",
          "| | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for label, key in (("AUC (1.0 = perfect, 0.5 = chance)", "auc"), ("mean best similarity, answerable", "mean_best_sim_answerable"),
                       ("mean best similarity, not covered", "mean_best_sim_not_covered"),
                       ("lowest answerable", "min_answerable"), ("lowest not covered", "min_not_covered"),
                       ("highest not covered", "max_not_covered")):
        L.append(f"| {label} | " + " | ".join(f"{s[k]['gate'][key]:.3f}" for k in keys) + " |")
    L.append(f"| not-covered questions the current {CURRENT_CUTOFF} cutoff refuses without an LLM call | " + " | ".join(
        pct(s[k]['gate']['not_covered_refused_by_current_cutoff']) for k in keys) + " |")
    L.append(f"| answerable questions the current {CURRENT_CUTOFF} cutoff wrongly refuses | " + " | ".join(
        pct(s[k]['gate']['answerable_wrongly_refused_by_current_cutoff']) for k in keys) + " |")
    L.append("\nSimilarity scales differ between models, so a cutoff tuned for MiniLM would need re-tuning for another model.")

    if results.get("latency_control"):
        lc = results["latency_control"]
        L += ["", "## Latency control: MiniLM run through the same runtime as bge-base", "",
              "Chroma's wrapper (the app today) and fastembed are different runtimes, so the table above mixes library and model.",
              "", "| | p50 / p95 / mean (ms) |", "|---|---|"]
        L.append(f"| MiniLM via Chroma (the app today) | {c[keys[0]]['query_ms_p50']:.1f} / {c[keys[0]]['query_ms_p95']:.1f} / {c[keys[0]]['query_ms_mean']:.1f} |")
        L.append(f"| MiniLM via fastembed (control) | {lc['query_ms_p50']:.1f} / {lc['query_ms_p95']:.1f} / {lc['query_ms_mean']:.1f} |")

    L += ["", "## Misses (Recall@5 below 1): top-5 sections, `*` marks a chunk that contains a needed fact", ""]
    for k in keys:
        misses = [r for r in results["rows"] if r["model"] == k and r["answerable"] and r["recall@5"] < 1.0]
        L.append(f"**{k}**: {len(misses)} of {meta['n_answerable']}")
        for r in misses:
            L.append(f"- `{r['id']}` ({r['style']}, recall@5 {r['recall@5']:.2f}): {', '.join(r['top5'])}")
        L.append("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["minilm", "bge-base"], choices=sorted(BUILDERS))
    args = ap.parse_args()

    print("loading policies and chunking (production chunker)...", flush=True)
    corpora = [load_company(c, v) for c, v in COMPANIES]
    all_q = [q for co in corpora for q in co["queries"]]
    n_answerable = sum(q["expected"]["status"] == "answered" for q in all_q)
    n_chunks = sum(len(co["ids"]) for co in corpora)
    print(f"  {n_chunks} chunks, {len(all_q)} queries ({n_answerable} answerable)", flush=True)

    specs, rows_by_model, results = [], {}, {"summary": {}, "cost": {}, "rows": []}
    questions = [q["question"] for q in all_q]
    for key in args.models:
        print(f"[{key}] loading...", flush=True)
        spec = BUILDERS[key]()
        specs.append(spec)
        print(f"[{key}] retrieval on all queries...", flush=True)
        rows, info = evaluate(spec, corpora)
        rows_by_model[key] = rows
        results["rows"] += rows
        results["summary"][key] = summarise(rows)
        print(f"[{key}] timing query embedding...", flush=True)
        ms = time_queries(spec, questions)
        mean_chunks = n_chunks / len(corpora)
        index_mb = mean_chunks * spec.dim * 4 / 1e6
        results["cost"][key] = {
            "label": spec.label, "dim": spec.dim, "load_seconds": spec.load_seconds, "size_mb": spec.size_mb,
            "embed_seconds": info["embed_seconds"], "query_ms_p50": M.percentile(ms, 50), "query_ms_p95": M.percentile(ms, 95),
            "query_ms_mean": float(np.mean(ms)), "n_timing_samples": len(ms),
            "index_mb_per_company": index_mb, "projected_index_mb": index_mb * PROJECTED_COMPANIES,
        }
    if len(specs) >= 2:
        results["paired"] = paired(rows_by_model[specs[0].key], rows_by_model[specs[1].key])
    if "minilm" in args.models and any(k != "minilm" for k in args.models):
        # Latency only: the same MiniLM weights through fastembed, to separate runtime from model size.
        print("[control] MiniLM via fastembed, timing only...", flush=True)
        control = build_fastembed("minilm-fastembed", "sentence-transformers/all-MiniLM-L6-v2", 384, "", local_only=False)
        ms = time_queries(control, questions)
        results["latency_control"] = {"query_ms_p50": M.percentile(ms, 50), "query_ms_p95": M.percentile(ms, 95),
                                      "query_ms_mean": float(np.mean(ms)), "n_timing_samples": len(ms)}
    results["meta"] = {"n_chunks": n_chunks, "n_queries": len(all_q), "n_answerable": n_answerable,
                       "n_not_covered": len(all_q) - n_answerable, "models": [s.label for s in specs],
                       "machine": {"cpus": os.cpu_count(), "platform": platform.platform()}}

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "e1_per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in results["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS / "e1_results.json").write_text(
        json.dumps({k: v for k, v in results.items() if k != "rows"}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(results, specs)
    (RESULTS / "e1_full_report.md").write_text(report, encoding="utf-8")   # E1_results.md is the hand-written summary
    print("\n" + report)


if __name__ == "__main__":
    main()
