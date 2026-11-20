"""E4 (offline): does HyDE help when the embedder is MiniLM (the other E1 model)?

    python experiments/run_e4.py

Same 192 questions, same labels, same chunks, same 201 cached HyDE passages as E3 (so no LLM calls), same four setups
(A-D), same modes (none / fused / only) and the same pre-committed rule. Only the embedder changes: MiniLM, with no query
instruction. bge-base is run in the same script, so the embedder x HyDE grid is computed on identical data.

Writes experiments/results/e4_full_report.md, e4_results.json and e4_per_query.jsonl.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from experiments import metrics as M  # noqa: E402
from experiments.offline import CONFIGS, COMPANIES, get_vectors, load_company  # noqa: E402
from experiments.run_e1 import BUILDERS, STYLE  # noqa: E402
from experiments.run_e2 import STYLES, evaluate, md_table, summarise  # noqa: E402
from experiments.run_e3 import ADOPT_MIN_GAIN, MODES, base_of, generate, paired_clustered  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
EMBEDDERS = ("minilm", "bge-base")


def cluster_diff(a: dict, b: dict) -> dict:
    """b minus a per query id, interval resampling whole base questions."""
    ids = sorted(a)
    diffs = [b[i] - a[i] for i in ids]
    mean, lo, hi = M.cluster_bootstrap_mean_ci(diffs, [base_of(i) for i in ids])
    return {"mean_diff": mean, "ci95": [lo, hi], "n": len(ids)}


def main() -> None:
    settings = get_settings()
    print("loading policies and chunking...", flush=True)
    corpora = [load_company(c, v, extended=True) for c, v in COMPANIES]
    all_q = [q for co in corpora for q in co["queries"]]
    passages = generate(all_q, settings)            # all cached: no LLM calls
    ids = [q["id"] for q in all_q]
    e2 = RESULTS / "e2_results.json"
    lam = json.loads(e2.read_text(encoding="utf-8"))["configs"]["B"]["lam"] if e2.exists() else 0.85
    configs = {k: (c.with_lam(lam) if c.mmr else c) for k, c in CONFIGS.items()}

    rows, summary, vectors = {}, {}, {}
    for emb in EMBEDDERS:
        print(f"loading {emb} and vectors...", flush=True)
        spec = BUILDERS[emb]()
        D, Q = get_vectors(spec, corpora)
        H = dict(zip(ids, spec.docs([passages[i]["passage"] for i in ids])))
        vectors[emb] = (D, Q, H)
        for k, cfg in configs.items():
            for m in MODES:
                rows[(emb, k, m)] = evaluate(cfg, corpora, D, Q, H=H, mode=m)
        summary[emb] = {k: {m: summarise(rows[(emb, k, m)]) for m in MODES} for k in configs}

    # ---- paired HyDE effect inside each embedder
    paired = {emb: {k: {m: {met: paired_clustered(rows[(emb, k, "none")], rows[(emb, k, m)], met) for met in ("recall", "rr")}
                        for m in ("fused", "only")} for k in configs} for emb in EMBEDDERS}

    rec = {key: {r["id"]: r["recall"] for r in rs} for key, rs in rows.items()}
    sty = {r["id"]: r["style"] for r in rows[("minilm", "C", "none")]}

    # ---- the grid for C and its comparisons
    grid = {emb: {m: float(np.mean(list(rec[(emb, "C", m)].values()))) for m in ("none", "fused")} for emb in EMBEDDERS}
    gain = {emb: {i: rec[(emb, "C", "fused")][i] - rec[(emb, "C", "none")][i] for i in rec[(emb, "C", "none")]} for emb in EMBEDDERS}
    interaction = cluster_diff(gain["bge-base"], gain["minilm"])               # MiniLM gain minus bge-base gain
    gap = {"minilm_none_vs_bge_none": cluster_diff(rec[("minilm", "C", "none")], rec[("bge-base", "C", "none")]),
           "minilm_hyde_vs_bge_none": cluster_diff(rec[("bge-base", "C", "none")], rec[("minilm", "C", "fused")])}

    def subset(emb, keep):
        d = [gain[emb][i] for i in gain[emb] if keep(i)]
        return {"mean_diff": float(np.mean(d)) if d else float("nan"), "n": len(d)}

    is_new = lambda i: "-x" in i
    by_source = {emb: {"original": subset(emb, lambda i: not is_new(i)), "new": subset(emb, is_new)} for emb in EMBEDDERS}
    by_style = {emb: {s: {"none": float(np.mean([v for i, v in rec[(emb, "C", "none")].items() if sty[i] == s])),
                          "fused": float(np.mean([v for i, v in rec[(emb, "C", "fused")].items() if sty[i] == s]))}
                      for s in STYLES} for emb in EMBEDDERS}
    wtl = {}
    for emb in EMBEDDERS:
        a, b = rec[(emb, "C", "none")], rec[(emb, "C", "fused")]
        wtl[emb] = M.win_tie_loss([a[i] for i in sorted(a)], [b[i] for i in sorted(a)])

    # ---- gate: top-1 similarity, covered vs not covered (raw query, as the app's gate; and best of query/passage)
    gate = {}
    for emb in EMBEDDERS:
        D, Q, H = vectors[emb]
        cov_q, unc_q, cov_b, unc_b = [], [], [], []
        for co in corpora:
            for q in co["queries"]:
                sq = float(max(D[co["company"]] @ Q[q["id"]]))
                sb = max(sq, float(max(D[co["company"]] @ H[q["id"]])))
                (cov_q if q["expected"]["status"] == "answered" else unc_q).append(sq)
                (cov_b if q["expected"]["status"] == "answered" else unc_b).append(sb)
        gate[emb] = {"n_covered": len(cov_q), "n_not_covered": len(unc_q),
                     "auc_query": M.auc(cov_q, unc_q), "auc_best_of_query_and_passage": M.auc(cov_b, unc_b),
                     "not_covered_max_query": max(unc_q), "not_covered_max_with_passage": max(unc_b)}

    res = {"meta": {"n_answerable": len(rec[("minilm", "C", "none")]), "hyde_model": settings.hyde_model, "lambda": lam,
                    "n_bases": len({base_of(i) for i in rec[("minilm", "C", "none")]})},
           "summary": summary, "paired": paired, "grid": grid, "interaction": interaction, "gap": gap,
           "by_source": by_source, "by_style": by_style, "win_tie_loss": wtl, "gate": gate}

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "e4_per_query.jsonl").open("w", encoding="utf-8") as f:
        for (emb, k, m), rs in rows.items():
            for r in rs:
                f.write(json.dumps({**r, "embedder": emb, "mode": m}, ensure_ascii=False) + "\n")
    (RESULTS / "e4_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(res, configs)
    (RESULTS / "e4_full_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


def ci_txt(v: dict) -> str:
    lo, hi = v["ci95"]
    return f"{v['mean_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]{'' if lo <= 0 <= hi else ' *'}"


def render(res: dict, configs: dict) -> str:
    m = res["meta"]
    L = ["# E4 (offline): does HyDE help with MiniLM? (4x set)", "",
         f"{m['n_answerable']} answerable queries from {m['n_bases']} base questions, exact search, same chunks and the same cached "
         f"HyDE passages (`{m['hyde_model']}`) as E3. RETRIEVAL ONLY. Intervals resample whole base questions. `*` = excludes zero.", ""]
    L += ["## Facts found, by embedder, config and way of searching", ""]
    rows = []
    for emb in EMBEDDERS:
        for k in configs:
            rows.append([emb, k] + [f"{res['summary'][emb][k][mo]['recall']:.3f}" for mo in MODES])
    L += md_table(["embedder", "config", "none", "fused", "only"], rows)

    L += ["", "## HyDE minus none, paired (fused, then only)", ""]
    rows = []
    for emb in EMBEDDERS:
        for k in configs:
            rows.append([emb, k, ci_txt(res["paired"][emb][k]["fused"]["recall"]), ci_txt(res["paired"][emb][k]["only"]["recall"])])
    L += md_table(["embedder", "config", "fused - none (recall)", "only - none (recall)"], rows)

    v = res["paired"]["minilm"]["C"]["fused"]["recall"]
    adopt = v["mean_diff"] >= ADOPT_MIN_GAIN and v["ci95"][0] > 0
    L += ["", "## Decision rule (same as E3), MiniLM, config C", "",
          f"Adopt only if recall rises by at least +{ADOPT_MIN_GAIN:.2f} and the interval excludes zero. Result: "
          f"**{v['mean_diff']:+.3f}**, interval [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] -> **{'ADOPT' if adopt else 'DO NOT ADOPT'}**.", ""]

    g = res["grid"]
    L += ["## Embedder x HyDE grid (config C, facts found)", ""]
    L += md_table(["", "no HyDE", "HyDE fused", "HyDE effect"],
                  [[emb, f"{g[emb]['none']:.3f}", f"{g[emb]['fused']:.3f}", f"{g[emb]['fused'] - g[emb]['none']:+.3f}"] for emb in EMBEDDERS])
    L += ["", f"- **Interaction** (MiniLM's HyDE effect minus bge-base's): {ci_txt(res['interaction'])}",
          f"- **bge-base minus MiniLM, neither using HyDE:** {ci_txt(res['gap']['minilm_none_vs_bge_none'])}",
          f"- **MiniLM + HyDE minus bge-base without HyDE** (does HyDE close the gap?): {ci_txt(res['gap']['minilm_hyde_vs_bge_none'])}", ""]

    L += ["## Queries where HyDE (fused, C) found more / the same / fewer facts", ""]
    L += md_table(["embedder", "better", "same", "worse"], [[emb] + [str(x) for x in res["win_tie_loss"][emb]] for emb in EMBEDDERS])

    L += ["", "## HyDE effect (fused - none, C) by question source", ""]
    L += md_table(["embedder", "original queries", "new queries"],
                  [[emb] + [f"{res['by_source'][emb][s]['mean_diff']:+.3f} (n={res['by_source'][emb][s]['n']})" for s in ("original", "new")]
                   for emb in EMBEDDERS])

    L += ["", "## Facts found by wording (C): none -> fused", ""]
    L += md_table(["wording"] + list(EMBEDDERS),
                  [[s] + [f"{res['by_style'][emb][s]['none']:.3f} -> {res['by_style'][emb][s]['fused']:.3f}" for emb in EMBEDDERS] for s in STYLES])

    L += ["", "## Not-covered gate: top-1 similarity, covered vs not covered", "",
          "The app's gate uses the raw question's similarity only, so HyDE does not change it. The last columns show what it would be "
          "if the passage were also used.", ""]
    L += md_table(["embedder", "n covered / not covered", "AUC (question)", "AUC (best of question, passage)",
                   "highest not-covered score (question)", "(with passage)"],
                  [[emb, f"{x['n_covered']} / {x['n_not_covered']}", f"{x['auc_query']:.3f}", f"{x['auc_best_of_query_and_passage']:.3f}",
                    f"{x['not_covered_max_query']:.3f}", f"{x['not_covered_max_with_passage']:.3f}"] for emb, x in res["gate"].items()])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
