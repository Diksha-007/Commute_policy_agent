"""E3 (offline part): does HyDE help retrieval?

    python experiments/run_e3.py

HyDE: a small LLM (Qwen) writes a short policy-style passage that would answer the question; we search with that
passage as well as with the question. Here that is measured on retrieval only, for all four E2 configs (A-D):
    none    the question alone                      (E2)
    fused   question + passage, merged              (what the app does when HyDE is on)
    only    the passage alone                       (ablation)
The 57 passages are generated ONCE (real Groq calls, about 15,000 Qwen tokens), cached in
experiments/data/hyde_passages.jsonl, and reused, so every variant costs no further LLM calls. The generation
calls also give the HyDE step's latency. The passage is embedded as a document (no query instruction).

Writes experiments/results/e3_full_report.md, e3_results.json and e3_per_query.jsonl.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import GroqLLM, LLMTimeout, LLMUnavailable  # noqa: E402
from app.prompts import HYDE_SYSTEM, build_hyde_messages  # noqa: E402
from experiments import metrics as M  # noqa: E402
from experiments.matching import norm  # noqa: E402
from experiments.offline import CONFIGS, COMPANIES, get_vectors, invented_numbers, load_company  # noqa: E402
from experiments.run_e1 import BUILDERS, STYLE  # noqa: E402
from experiments.run_e2 import STYLES, evaluate, mean_of, md_table, summarise  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
PASSAGES = ROOT / "experiments" / "data" / "hyde_passages.jsonl"
MODES = ("none", "fused", "only")
APP_HYDE_TIMEOUT_MS = 1000   # app/config.py hyde_timeout_s: slower than this and the app skips HyDE


# ------------------------------------------------------------------ passages

def generate(queries: list[dict], settings) -> dict[str, dict]:
    signature = hashlib.sha256((HYDE_SYSTEM + settings.hyde_model + settings.hyde_reasoning_effort).encode()).hexdigest()[:12]
    cache: dict[str, dict] = {}
    if PASSAGES.exists():
        for line in PASSAGES.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("signature") == signature:
                cache[r["id"]] = r
    todo = [q for q in queries if q["id"] not in cache or cache[q["id"]]["question"] != q["question"]]
    print(f"HyDE passages: {len(cache)} cached, {len(todo)} to generate with {settings.hyde_model}", flush=True)
    if not todo:
        return cache
    llm = GroqLLM(settings.groq_key, settings.groq_base_url)
    PASSAGES.parent.mkdir(parents=True, exist_ok=True)
    with PASSAGES.open("a", encoding="utf-8") as f:
        for n, q in enumerate(todo, start=1):
            attempts = 0
            while True:
                attempts += 1
                start = time.perf_counter()
                try:
                    text = llm.complete(model=settings.hyde_model, messages=build_hyde_messages(q["question"]),
                                        timeout=10.0, max_tokens=150, reasoning_effort=settings.hyde_reasoning_effort)
                    ms = (time.perf_counter() - start) * 1000
                    break
                except (LLMUnavailable, LLMTimeout) as e:
                    if attempts >= 5:
                        raise SystemExit(f"giving up on {q['id']} after {attempts} attempts: {e}")
                    print(f"  {q['id']}: {type(e).__name__}: {str(e)[:160]}; waiting 20 s (attempt {attempts})", flush=True)
                    time.sleep(20)
            row = {"id": q["id"], "question": q["question"], "passage": text.strip(), "latency_ms": ms,
                   "attempts": attempts, "model": settings.hyde_model, "signature": signature}
            cache[q["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {q['id']:<20} {ms:6.0f} ms  {len(text.split()):3d} words  attempts={attempts}", flush=True)
            time.sleep(2.6)   # Qwen's free tier allows about 30 requests a minute; 1.5 s tripped it twice
    return cache


# ------------------------------------------------------------------ analysis

def closeness(co, D, Q, H, q) -> tuple[float, float]:
    """Cosine of the best chunk that contains the answer, to the question and to the HyDE passage."""
    idx = {cid: i for i, cid in enumerate(co["ids"])}
    rel = [idx[c] for c in set().union(*q["_rel"])]
    return float(max(D[co["company"]][rel] @ Q[q["id"]])), float(max(D[co["company"]][rel] @ H[q["id"]]))


ADOPT_MIN_GAIN = 0.05   # fixed BEFORE the bigger run: HyDE counts as a win only if recall rises by at least this
                        # for config C AND the interval (resampling whole base questions) excludes zero


def base_of(query_id: str) -> str:
    """'acme-x04-v2' -> 'acme-x04': the rewordings of one question share a base."""
    return query_id.rsplit("-v", 1)[0]


def paired_clustered(rows_a, rows_b, key) -> dict:
    """b minus a per query, with an interval that resamples whole BASE QUESTIONS (rewordings are not independent)."""
    a = {r["id"]: r[key] for r in rows_a}
    b = {r["id"]: r[key] for r in rows_b}
    ids = sorted(a)
    diffs = [b[i] - a[i] for i in ids]
    mean_diff, lo, hi = M.cluster_bootstrap_mean_ci(diffs, [base_of(i) for i in ids])
    wins, ties, losses = M.win_tie_loss([a[i] for i in ids], [b[i] for i in ids])
    per_base: dict[str, list[float]] = {}
    for i, d in zip(ids, diffs):
        per_base.setdefault(base_of(i), []).append(d)
    return {"mean_diff": mean_diff, "ci95": [lo, hi], "higher": wins, "same": ties, "lower": losses,
            "bases": len(per_base),
            "bases_up": sum(1 for v in per_base.values() if np.mean(v) > 1e-9),
            "bases_down": sum(1 for v in per_base.values() if np.mean(v) < -1e-9)}


def render(res: dict) -> str:
    keys = list(res["configs"])
    L = [f"# E3 (offline): does HyDE help retrieval? ({res['meta']['set']})", "",
         f"{res['meta']['n_answerable']} answerable queries from {res['meta']['n_bases']} base questions, bge-base, exact search. "
         f"Passages written by `{res['meta']['hyde_model']}`. RETRIEVAL ONLY: the extra LLM call's effect on final answers "
         "is not measured here. Intervals resample whole base questions, because the rewordings of one question are not independent.", ""]

    L += ["## Facts found, by config and way of searching", "",
          "`none` = question only, `fused` = question + HyDE passage (as the app does), `only` = passage alone.", ""]
    rows = []
    for k in keys:
        for mode in MODES:
            s = res["summary"][k][mode]
            rows.append([k, mode, f"{s['recall']:.3f}", f"{100 * s['complete']:.0f}%", f"{s['rr']:.3f}", f"{s['wasted_slots']:.2f}"])
    L += md_table(["config", "search with", "facts found", "all facts found", "MRR", "wasted slots"], rows)

    L += ["", "## Paired differences against `none` (95% bootstrap interval over base questions)", "",
          "`*` = interval excludes zero.", ""]
    rows = []
    for k in keys:
        for mode in ("fused", "only"):
            for metric in ("recall", "rr"):
                v = res["paired"][k][mode][metric]
                lo, hi = v["ci95"]
                rows.append([k, mode, metric, f"{v['mean_diff']:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]{'' if lo <= 0 <= hi else ' *'}",
                             f"{v['higher']} / {v['same']} / {v['lower']}",
                             f"{v['bases_up']} / {v['bases'] - v['bases_up'] - v['bases_down']} / {v['bases_down']}"])
    L += md_table(["config", "search with", "metric", "mean difference", "95% interval", "queries better / same / worse",
                   "base questions better / same / worse"], rows)

    v = res["paired"]["C"]["fused"]["recall"]
    adopt = v["mean_diff"] >= ADOPT_MIN_GAIN and v["ci95"][0] > 0
    L += ["", "## The decision rule, fixed before this run", "",
          f"HyDE is adopted only if, for config C (top 5), recall rises by at least **+{ADOPT_MIN_GAIN:.2f}** and the interval "
          f"excludes zero. Result: mean difference **{v['mean_diff']:+.3f}**, interval [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
          f"-> **{'ADOPT' if adopt else 'DO NOT ADOPT'}**.", "",
          "Same comparison (fused minus none, recall) for the original queries, the new ones and all of them:", ""]
    L += md_table(["config", "original queries", "new queries", "all"],
                  [[k] + [f"{res['subsets'][k][s]['mean_diff']:+.3f} (n={res['subsets'][k][s]['n']})" for s in ("core", "new", "all")]
                   for k in keys])

    L += ["", "## By wording (share of queries with all facts found): none vs fused", ""]
    L += md_table(["wording"] + [h for k in keys for h in (f"{k} none", f"{k} fused")],
                  [[s] + [f"{100 * res['summary'][k][m]['by_style'][s]:.0f}%" for k in keys for m in ("none", "fused")]
                   for s in STYLES])

    L += ["", "## Why it does or does not help: how close is the answer chunk?", "",
          "Cosine similarity between the best answer-bearing chunk and (a) the question, (b) the HyDE passage. "
          "Higher means easier to find.", ""]
    L += md_table(["wording", "n", "question to answer chunk", "passage to answer chunk"],
                  [[s, str(v["n"]), f"{v['question']:.3f}", f"{v['passage']:.3f}"] for s, v in res["closeness"].items()])

    inv, lat = res["invented"], res["latency"]
    L += ["", "## The HyDE passages themselves", "",
          f"- Passages with at least one invented number, time or amount (not in the question, not anywhere in the policy): "
          f"**{inv['passages_with_invented']} of {inv['n']}** ({100 * inv['rate']:.0f}%).",
          f"- Typical length: {res['passage_words']['median']:.0f} words (max {res['passage_words']['max']}).",
          f"- HyDE call latency: p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms, mean {lat['mean']:.0f} ms over {lat['n']} calls. "
          f"**{lat['over_app_timeout']} of {lat['n']}** took longer than the app's {APP_HYDE_TIMEOUT_MS} ms limit and would have been skipped.",
          f"- Calls that needed a retry because of a rate limit: {lat['retried']}.", ""]

    L += ["## Rescued and broken queries (fused vs none)", ""]
    for k in ("A", "C"):
        r, b = res["rescued"][k], res["broken"][k]
        L.append(f"**{k}**: {len(r)} rescued, {len(b)} broken" + (" (first 12 of each shown)" if max(len(r), len(b)) > 12 else ""))
        L += [f"- rescued `{x['id']}` ({x['style']}): none {x['none']} -> fused {x['fused']}" for x in r[:12]]
        L += [f"- broken `{x['id']}` ({x['style']}): none {x['none']} -> fused {x['fused']}" for x in b[:12]] + [""]
    L += ["## Example passages", ""]
    for ex in res["examples"]:
        L += [f"**{ex['kind']}** `{ex['id']}`: *{ex['question']}*", "", f"> {ex['passage']}", ""]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main() -> None:
    settings = get_settings()
    if not settings.groq_key and not PASSAGES.exists():
        sys.exit("groq_key is not set and no cached passages exist")
    extended = "--extended" in sys.argv       # the 4x query set (core + extra)
    prefix = "e3_ext" if extended else "e3"
    print("loading policies and chunking...", flush=True)
    corpora = [load_company(c, v, extended=extended) for c, v in COMPANIES]
    all_q = [q for co in corpora for q in co["queries"]]
    passages = generate(all_q, settings)

    print("loading bge-base and vectors...", flush=True)
    spec = BUILDERS["bge-base"]()
    D, Q = get_vectors(spec, corpora)
    ids = [q["id"] for q in all_q]
    H = dict(zip(ids, spec.docs([passages[i]["passage"] for i in ids])))   # the passage is embedded as a document

    e2 = RESULTS / "e2_results.json"
    lam = json.loads(e2.read_text(encoding="utf-8"))["configs"]["B"]["lam"] if e2.exists() else 0.85
    configs = {k: (c.with_lam(lam) if c.mmr else c) for k, c in CONFIGS.items()}

    rows = {(k, m): evaluate(cfg, corpora, D, Q, H=H, mode=m) for k, cfg in configs.items() for m in MODES}
    summary = {k: {m: summarise(rows[(k, m)]) for m in MODES} for k in configs}
    paired_res = {k: {m: {met: paired_clustered(rows[(k, "none")], rows[(k, m)], met) for met in ("recall", "rr")}
                      for m in ("fused", "only")} for k in configs}

    def subset_diff(k, keep):
        none = {r["id"]: r["recall"] for r in rows[(k, "none")] if keep(r["id"])}
        fused = {r["id"]: r["recall"] for r in rows[(k, "fused")] if keep(r["id"])}
        return {"mean_diff": float(np.mean([fused[i] - none[i] for i in none])) if none else float("nan"), "n": len(none)}

    is_new = lambda i: "-x" in i          # ids of the extra queries look like acme-x04-v2
    subsets = {k: {"core": subset_diff(k, lambda i: not is_new(i)), "new": subset_diff(k, is_new),
                   "all": subset_diff(k, lambda i: True)} for k in configs}

    def detail(k):
        none = {r["id"]: r for r in rows[(k, "none")]}
        fused = {r["id"]: r for r in rows[(k, "fused")]}
        up = [{"id": i, "style": none[i]["style"], "none": none[i]["shown"], "fused": fused[i]["shown"]} for i in none if fused[i]["recall"] > none[i]["recall"]]
        down = [{"id": i, "style": none[i]["style"], "none": none[i]["shown"], "fused": fused[i]["shown"]} for i in none if fused[i]["recall"] < none[i]["recall"]]
        return up, down

    rescued, broken = {}, {}
    for k in configs:
        rescued[k], broken[k] = detail(k)

    clos = {s: [] for s in STYLES}
    for co in corpora:
        for q in co["queries"]:
            if q["expected"]["status"] == "answered":
                clos[STYLE[q["variant"]]].append(closeness(co, D, Q, H, q))
    closeness_res = {s: {"n": len(v), "question": float(np.mean([a for a, _ in v])), "passage": float(np.mean([b for _, b in v]))}
                     for s, v in clos.items()}

    policy_norm = {co["company"]: norm(" ".join(c.text for c in co["chunks"])) for co in corpora}
    invented = {}
    for co in corpora:
        for q in co["queries"]:
            bad = invented_numbers(passages[q["id"]]["passage"], q["question"], policy_norm[co["company"]])
            if bad:
                invented[q["id"]] = bad
    lat_ms = [passages[i]["latency_ms"] for i in ids]
    words = [len(passages[i]["passage"].split()) for i in ids]

    by_id = {q["id"]: q for q in all_q}
    examples = []
    for kind, pool in (("rescued", rescued["A"][:2]), ("broken", broken["A"][:2])):
        for x in pool:
            examples.append({"kind": f"{kind} (config A)", "id": x["id"], "question": by_id[x["id"]]["question"],
                             "passage": passages[x["id"]]["passage"][:420]})
    for i, bad in list(invented.items())[:2]:
        examples.append({"kind": f"invented {bad}", "id": i, "question": by_id[i]["question"], "passage": passages[i]["passage"][:420]})

    res = {"meta": {"n_answerable": len(rows[("A", "none")]), "hyde_model": settings.hyde_model, "lambda": lam,
                    "set": "4x set: original + new queries" if extended else "original 48 queries",
                    "n_bases": len({base_of(r["id"]) for r in rows[("A", "none")]})},
           "configs": {k: {"label": c.label, "k": c.k, "mmr": c.mmr, "pool": c.pool, "lam": c.lam} for k, c in configs.items()},
           "summary": summary, "paired": paired_res, "subsets": subsets, "closeness": closeness_res, "rescued": rescued, "broken": broken,
           "invented": {"n": len(ids), "passages_with_invented": len(invented), "rate": len(invented) / len(ids), "details": invented},
           "latency": {"n": len(lat_ms), "p50": M.percentile(lat_ms, 50), "p95": M.percentile(lat_ms, 95), "mean": float(np.mean(lat_ms)),
                       "over_app_timeout": sum(x > APP_HYDE_TIMEOUT_MS for x in lat_ms),
                       "retried": sum(passages[i]["attempts"] > 1 for i in ids)},
           "passage_words": {"median": float(np.median(words)), "max": max(words)}, "examples": examples}

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"{prefix}_per_query.jsonl").open("w", encoding="utf-8") as f:
        for (k, m), rs in rows.items():
            for r in rs:
                f.write(json.dumps({**r, "mode": m}, ensure_ascii=False) + "\n")
    (RESULTS / f"{prefix}_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(res)
    (RESULTS / f"{prefix}_full_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
