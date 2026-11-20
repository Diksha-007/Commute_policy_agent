"""E2 (offline part): how many chunks should the LLM see, and how should they be chosen?

    python experiments/run_e2.py

Configs, all on bge-base vectors and exact cosine search:
    A  top-3 by relevance                      (what the app does today)
    B  top-3 chosen with MMR (pool 10)
    C  top-5 by relevance (the LLM then cites at most 3)
    D  top-5 chosen with MMR (pool 15)
This measures RETRIEVAL only: what reaches the LLM. It cannot show whether the LLM is confused by extra chunks,
or how long the LLM call takes. MMR's lambda is chosen on Globex and reported on Acme and Initech (hold-out).

Writes experiments/results/e2_full_report.md, e2_results.json and e2_per_query.jsonl.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import metrics as M  # noqa: E402
from experiments.offline import (  # noqa: E402
    CONFIGS, COMPANIES, LAMBDA_GRID, Config, get_vectors, load_company, score_selection, section_of, select,
)
from experiments.run_e1 import BUILDERS, STYLE, time_queries  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
TUNE_ON = "globex"
STYLES = ("original", "keyword", "chatty", "synonyms")
METRICS = ("recall", "recall@3", "recall@5", "precision", "precision_vs_ceiling", "rr",
           "distinct_sections", "wasted_slots", "context_chars")


def evaluate(cfg, corpora, D, Q, companies=None, H=None, mode="none") -> list[dict]:
    """H: {query id: HyDE passage vector}; mode: 'none' | 'fused' | 'only' (see offline.select)."""
    rows = []
    for co in corpora:
        if companies and co["company"] not in companies:
            continue
        sect = {cid: section_of(cid) for cid in co["ids"]}
        length = {c.chunk_id: len(c.text) for c in co["chunks"]}
        for q in co["queries"]:
            if q["expected"]["status"] != "answered":
                continue
            h = H.get(q["id"]) if H else None
            shown = [co["ids"][i] for i in select(cfg, D[co["company"]], Q[q["id"]], h, mode)]
            union = set().union(*q["_rel"])
            row = {"config": cfg.key, "id": q["id"], "company": co["company"], "style": STYLE[q["variant"]],
                   "category": q["category"], "n_groups": len(q["_rel"]),
                   "shown": [sect[c] + ("*" if c in union else "") for c in shown],
                   "recall@3": M.recall_at_k(shown, q["_rel"], 3), "recall@5": M.recall_at_k(shown, q["_rel"], 5)}
            row.update(score_selection(shown, q["_rel"], sect, length))
            rows.append(row)
    return rows


def mean_of(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def summarise(rows) -> dict:
    out = {"n": len(rows)} | {k: mean_of(rows, k) for k in METRICS}
    out["complete"] = float(np.mean([r["recall"] == 1.0 for r in rows]))
    out["by_category"] = {c: mean_of([r for r in rows if r["category"] == c], "recall")
                          for c in sorted({r["category"] for r in rows})}
    out["by_style"] = {s: float(np.mean([r["recall"] == 1.0 for r in rows if r["style"] == s])) for s in STYLES}
    return out


def tune_lambda(base, corpora, D, Q) -> list[dict]:
    table = []
    for lam in LAMBDA_GRID:
        rows = evaluate(base.with_lam(lam), corpora, D, Q, companies={TUNE_ON})
        table.append({"lambda": lam, "recall": mean_of(rows, "recall"), "wasted_slots": mean_of(rows, "wasted_slots"),
                      "distinct_sections": mean_of(rows, "distinct_sections"), "n": len(rows)})
    return table


def pick_lambda(table: list[dict]) -> float:
    """Best recall on the tuning company; ties go to the larger lambda (the less aggressive MMR)."""
    return max(table, key=lambda t: (round(t["recall"], 9), t["lambda"]))["lambda"]


def paired(rows_a, rows_b, key) -> dict:
    a = {r["id"]: r[key] for r in rows_a}
    b = {r["id"]: r[key] for r in rows_b}
    ids = sorted(a)
    xs, ys = [a[i] for i in ids], [b[i] for i in ids]
    mean_diff, lo, hi = M.bootstrap_mean_ci([y - x for x, y in zip(xs, ys)])
    wins, ties, losses = M.win_tie_loss(xs, ys)
    return {"mean_diff": mean_diff, "ci95": [lo, hi], "higher": wins, "same": ties, "lower": losses}


def time_select(cfg, corpora, D, Q, repeats=20) -> list[float]:
    samples = []
    for co in corpora:
        for q in co["queries"]:
            if q["expected"]["status"] != "answered":
                continue
            for _ in range(repeats):
                start = time.perf_counter()
                select(cfg, D[co["company"]], Q[q["id"]])
                samples.append((time.perf_counter() - start) * 1000)
    return samples


# ------------------------------------------------------------------ report

def md_table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(r) + " |" for r in rows]


def render(res: dict) -> str:
    cfgs = res["configs"]
    keys = list(cfgs)
    L = ["# E2 (offline): how many chunks, and how to choose them", "",
         f"bge-base vectors, exact search, {res['meta']['n_answerable']} answerable queries over 3 companies. "
         "RETRIEVAL ONLY: what reaches the LLM. Not measured here: whether extra chunks confuse the LLM, "
         "and the LLM call's latency.", "",
         "Configs: " + "; ".join(f"**{k}** {cfgs[k]['label']}" + (f" (pool {cfgs[k]['pool']}, lambda {cfgs[k]['lam']})" if cfgs[k]['mmr'] else "")
                                 for k in keys), ""]

    L += ["## Main results (all queries)", "",
          "`recall` = facts found among the chunks the LLM sees (3 for A/B, 5 for C/D). `recall@3` / `recall@5` = "
          "the same measured on the first 3 / 5 chunks shown.", ""]
    L += md_table(["config", "shown", "recall", "recall@3", "recall@5", "all facts found", "precision (vs ceiling)", "MRR",
                   "distinct sections", "wasted slots", "context chars"],
                  [[k, str(cfgs[k]["k"]), f"{res['summary'][k]['recall']:.3f}", f"{res['summary'][k]['recall@3']:.3f}",
                    f"{res['summary'][k]['recall@5']:.3f}", f"{100 * res['summary'][k]['complete']:.0f}%",
                    f"{res['summary'][k]['precision']:.2f} ({res['summary'][k]['precision_vs_ceiling']:.2f})",
                    f"{res['summary'][k]['rr']:.3f}", f"{res['summary'][k]['distinct_sections']:.2f}",
                    f"{res['summary'][k]['wasted_slots']:.2f}", f"{res['summary'][k]['context_chars']:.0f}"] for k in keys])

    L += ["", f"## Lambda chosen on {TUNE_ON.capitalize()} ({res['tuning']['B'][0]['n']} queries), then fixed", ""]
    for k in ("B", "D"):
        L += [f"**{k}** (chosen lambda = {cfgs[k]['lam']})", ""]
        L += md_table(["lambda", "recall", "distinct sections", "wasted slots"],
                      [[f"{t['lambda']}", f"{t['recall']:.3f}", f"{t['distinct_sections']:.2f}", f"{t['wasted_slots']:.2f}"]
                       for t in res["tuning"][k]]) + [""]

    L += ["## Hold-out: Acme and Initech only (lambda was not tuned on these)", ""]
    L += md_table(["config", "recall", "all facts found", "wasted slots", "distinct sections"],
                  [[k, f"{res['holdout'][k]['recall']:.3f}", f"{100 * res['holdout'][k]['complete']:.0f}%",
                    f"{res['holdout'][k]['wasted_slots']:.2f}", f"{res['holdout'][k]['distinct_sections']:.2f}"] for k in keys])

    L += ["", "## Paired differences (second minus first, per query, 95% bootstrap interval)", "",
          "`*` = the interval excludes zero. For `wasted slots` lower is better.", ""]
    rows = []
    for (a, b), per_metric in res["paired"].items() if False else []:
        pass
    for pair, per_metric in res["paired"].items():
        for metric, v in per_metric.items():
            lo, hi = v["ci95"]
            star = "" if lo <= 0 <= hi else " *"
            rows.append([pair.replace("-", " minus "), metric, f"{v['mean_diff']:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]{star}",
                         f"{v['higher']} / {v['same']} / {v['lower']}"])
    L += md_table(["comparison", "metric", "mean difference", "95% interval", "higher / same / lower"], rows)
    L += ["", "- **B minus A** = effect of MMR at 3 chunks. **C minus A** = effect of showing 5 instead of 3. "
          "**D minus C** = effect of MMR at 5. **C minus B** = your story's turning point: retrieve 5, or diversify 3?"]

    L += ["", "## By question type (recall)", ""]
    cats = sorted(res["summary"][keys[0]]["by_category"])
    L += md_table(["type"] + keys, [[c] + [f"{res['summary'][k]['by_category'][c]:.2f}" for k in keys] for c in cats])

    L += ["", "## By wording (share of queries with all facts found)", ""]
    L += md_table(["wording"] + keys, [[s] + [f"{100 * res['summary'][k]['by_style'][s]:.0f}%" for k in keys] for s in STYLES])

    L += ["", "## Step timing (selection only; embedding is the same for every config)", "",
          f"Embedding one question with bge-base: p50 {res['timing']['embed_query_ms']['p50']:.1f} ms, "
          f"p95 {res['timing']['embed_query_ms']['p95']:.1f} ms (57 questions x 5).", ""]
    L += md_table(["config", "choose chunks p50 (ms)", "p95 (ms)", "approx. prompt tokens from chunks (chars/4)"],
                  [[k, f"{res['timing']['select_ms'][k]['p50']:.3f}", f"{res['timing']['select_ms'][k]['p95']:.3f}",
                    f"{res['summary'][k]['context_chars'] / 4:.0f}"] for k in keys])
    L += ["", "The LLM call itself is not measured offline. Its cost grows with the number of chunks in the prompt."]

    L += ["", "## Misses: queries where not every fact reached the LLM (`*` = chunk contains a needed fact)", ""]
    for k in keys:
        misses = [r for r in res["rows"] if r["config"] == k and r["recall"] < 1.0]
        L.append(f"**{k}**: {len(misses)} of {res['summary'][k]['n']}")
        L += [f"- `{r['id']}` ({r['style']}, recall {r['recall']:.2f}): {', '.join(r['shown'])}" for r in misses] + [""]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ robustness sweep (exploratory)

def sweep(corpora, D, Q) -> str:
    """Is MMR's result an artifact of the lambda or pool I picked? Mean recall over ALL queries, for a wide grid.
    Exploratory: it looks at every cell on the same queries, so no cell is a tuned result."""
    lams, pools = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0), (6, 10, 15, 20)
    plain = {k: mean_of(evaluate(Config("p", "p", k, False, 6), corpora, D, Q), "recall") for k in (3, 5)}
    L = ["# E2 sweep (exploratory): MMR recall for many lambda and pool values", "",
         "Mean recall over all 48 queries. lambda = 1.0 is plain relevance order. No cell here was tuned on held-out data.", ""]
    for k in (3, 5):
        L += [f"## {k} chunks shown (plain top-{k}: {plain[k]:.3f})", "",
              "| pool \\ lambda | " + " | ".join(str(x) for x in lams) + " |", "|---|" + "---|" * len(lams)]
        best = (-1.0, None)
        for pool in pools:
            cells = []
            for lam in lams:
                r = mean_of(evaluate(Config("s", "s", k, True, pool, lam), corpora, D, Q), "recall")
                cells.append(f"{r:.3f}")
                best = max(best, (r, (pool, lam)))
            L.append(f"| {pool} | " + " | ".join(cells) + " |")
        L += ["", f"Best cell: {best[0]:.3f} at pool {best[1][0]}, lambda {best[1][1]} (plain: {plain[k]:.3f}, difference {best[0] - plain[k]:+.3f}).", ""]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main() -> None:
    sweep_only = "--sweep" in sys.argv
    print("loading policies and chunking...", flush=True)
    corpora = [load_company(c, v) for c, v in COMPANIES]
    print("loading bge-base and vectors (cached after the first run)...", flush=True)
    spec = BUILDERS["bge-base"]()
    D, Q = get_vectors(spec, corpora)
    if sweep_only:
        text = sweep(corpora, D, Q)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "e2_sweep.md").write_text(text, encoding="utf-8")
        print(text)
        return

    print("tuning lambda on globex...", flush=True)
    tuning = {k: tune_lambda(CONFIGS[k], corpora, D, Q) for k in ("B", "D")}
    configs = dict(CONFIGS)
    for k in ("B", "D"):
        configs[k] = CONFIGS[k].with_lam(pick_lambda(tuning[k]))

    print("evaluating configs...", flush=True)
    rows = {k: evaluate(cfg, corpora, D, Q) for k, cfg in configs.items()}
    holdout = {k: summarise([r for r in rs if r["company"] != TUNE_ON]) for k, rs in rows.items()}
    summary = {k: summarise(rs) for k, rs in rows.items()}
    pairs = {"B-A": ("A", "B"), "C-A": ("A", "C"), "D-C": ("C", "D"), "C-B": ("B", "C")}
    paired_res = {name: {m: paired(rows[a], rows[b], m) for m in ("recall", "precision_vs_ceiling", "wasted_slots", "distinct_sections")}
                  for name, (a, b) in pairs.items()}

    print("timing...", flush=True)
    questions = [q["question"] for co in corpora for q in co["queries"]]
    embed_ms = time_queries(spec, questions)
    timing = {"embed_query_ms": {"p50": M.percentile(embed_ms, 50), "p95": M.percentile(embed_ms, 95)},
              "select_ms": {k: {"p50": M.percentile(s, 50), "p95": M.percentile(s, 95)}
                            for k, s in ((k, time_select(c, corpora, D, Q)) for k, c in configs.items())}}

    res = {"meta": {"n_answerable": sum(len(v) for v in [rows["A"]]), "tuned_on": TUNE_ON},
           "configs": {k: {"label": c.label, "k": c.k, "mmr": c.mmr, "pool": c.pool, "lam": c.lam} for k, c in configs.items()},
           "tuning": tuning, "summary": summary, "holdout": holdout, "paired": paired_res, "timing": timing,
           "rows": [r for rs in rows.values() for r in rs]}

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "e2_per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in res["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS / "e2_results.json").write_text(
        json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(res)
    (RESULTS / "e2_full_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
