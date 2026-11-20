"""Evaluation runner: one fixed question set, scored the same way every time.

    python -m eval.run_eval --llm groq --hyde off              # real model (demo/report)
    python -m eval.run_eval --llm groq --hyde on --repeats 3
    python -m eval.run_eval --llm fake                          # offline wiring/format check
    python -m eval.run_eval --sweep                             # similarity-threshold sweep (no LLM)

Setup (fresh temporary storage every run): ingest acme v1, globex v1, initech v1,
then acme v2, so every Acme case is answered by v2 and must show no v1 value.

Scoring per run of a case (see score_run). With --judge strings (the default for --llm fake) "content" is a
substring check against must_contain_*/must_not_contain. With --judge llm (the default for --llm groq) a judge model
that is given the case's REFERENCE ANSWER decides whether the reply is right (eval/judge.py), replacing the status,
citation and content checks; the leak check stays in code and vetoes a pass either way.

Checks of the string scorer:
    hit@5      an expected section is among the 5 retrieved chunks (the ones the model sees) (n/a if none expected)
    status     actual status is one of the expected statuses
    citation   answered: cites an expected section, all citations from the caller's
               company and active version; not answered: no citations
    content    answered: contains one of must_contain_any and none of must_not_contain
    leak       a citation from another company/version, or the answer names another
               company that the question did not mention
    pass       status AND citation AND content AND NOT leak
A case passes only if EVERY repeat passes (the model is not deterministic).

Output: eval/results/results_<tag>.jsonl (fields: id, input, expected, actual, pass, notes)
        eval/results/summary_<tag>.json and .md
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.auth import Principal  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import FakeLLM, GroqLLM  # noqa: E402
from app.retrieval import retrieve  # noqa: E402
from eval.judge import LLMJudge  # noqa: E402
from app.services import Services, build_services  # noqa: E402

CASES_PATH = ROOT / "eval" / "cases.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULT_FIELDS = ("id", "input", "expected", "actual", "pass", "notes")
SETUP = [("acme", "v1"), ("globex", "v1"), ("initech", "v1"), ("acme", "v2")]
COMPANY_NAMES = {"acme": "acme", "globex": "globex", "initech": "initech"}

log = logging.getLogger("eval")


# ------------------------------------------------------------ setup

def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            case = json.loads(line)
            for key in ("id", "company", "question", "expected"):
                if key not in case:
                    raise ValueError(f"{path}:{n} missing {key!r}")
            cases.append(case)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


def extractive_fake_llm() -> FakeLLM:
    """Offline stand-in: HyDE echoes a generic passage; the answer quotes the top
    excerpt and cites it. Checks wiring and format, not answer quality."""
    def respond(req: dict) -> str:
        if not req["json_mode"]:
            return "Employees are eligible for transport subject to the policy rules."
        content = req["messages"][-1]["content"]
        ids = re.findall(r'<excerpt id="([^"]+)"', content)
        bodies = re.findall(r'<excerpt id="[^"]+" section="[^"]*">\n(.*?)\n</excerpt>', content, flags=re.S)
        return json.dumps({"status": "answered", "answer": bodies[0][:300], "cited_chunk_ids": ids[:1]})
    return FakeLLM(responder=respond)


def build_eval_services(llm_kind: str, data_dir: Path, **overrides) -> Services:
    settings = get_settings().model_copy(update={"data_dir": data_dir, **overrides})
    llm = extractive_fake_llm() if llm_kind == "fake" else None
    services = build_services(settings, llm=llm)
    services.ingestor.gc_delay_s = 0
    for company, version in SETUP:
        text = (settings.policies_dir / f"{company}_{version}.md").read_text(encoding="utf-8")
        services.ingestor.ingest(company, version, text)
    return services


def principal_for(case: dict) -> Principal:
    p = case.get("profile", {})
    return Principal(employee_id=f"eval-{case['id']}", company_id=case["company"], role="employee",
                     grade=p.get("grade"), shift=p.get("shift"), gender=p.get("gender"))


# ------------------------------------------------------------ scoring

def norm(text: str) -> str:
    """Lowercase, unify typographic quotes, dashes and hyphens (the model writes non-breaking hyphens), collapse whitespace."""
    for fancy, plain in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                         ("–", "-"), ("—", "-"), ("‑", "-"), ("‐", "-"),     # non-breaking and plain hyphen
                         (" ", " "), (" ", " "), ("₹ ", "₹")):
        text = text.replace(fancy, plain)
    return " ".join(text.lower().split())


def allowed_statuses(expected: dict) -> list[str]:
    s = expected["status"]
    return s if isinstance(s, list) else [s]


def score_run(case: dict, result, doc_names: dict[str, str], versions: dict[str, str]) -> dict:
    exp = case["expected"]
    company = case["company"]
    sections = exp.get("sections", [])
    answer, question = norm(result.answer), norm(case["question"])

    retrieved_sections = [r["section_id"] for r in result.retrieved]
    hit = any(s in sections for s in retrieved_sections) if sections else None
    # Multi-part questions: every GROUP of required sections must be covered by the 5 retrieved chunks
    # (a group is covered when any one of its sections is retrieved).
    required = exp.get("required_sections")
    coverage = (sum(any(s in retrieved_sections for s in g) for g in required) / len(required)
                if required else None)

    status_ok = result.status in allowed_statuses(exp)

    foreign_citations = [c for c in result.citations
                         if c["doc_name"] != doc_names[company] or c["version"] != versions[company]]
    foreign_names = [n for co, n in COMPANY_NAMES.items()
                     if co != company and n in answer and n not in question]
    leak = bool(foreign_citations or foreign_names)

    cited_sections = [c["section_id"] for c in result.citations]
    if result.status == "answered":
        citation_ok = bool(sections) and any(s in sections for s in cited_sections) and not foreign_citations
        must = [norm(x) for x in exp.get("must_contain_any", [])]
        must_not = [norm(x) for x in exp.get("must_not_contain", [])]
        # must_contain_each: a list of groups; EVERY group needs at least one of its phrases.
        each = [[norm(x) for x in group] for group in exp.get("must_contain_each", [])]
        missing_parts = [group[0] for group in each if not any(m in answer for m in group)]
        missing = (bool(must) and not any(m in answer for m in must)) or bool(missing_parts)
        forbidden = [m for m in must_not if m in answer]
        content_ok = not missing and not forbidden
    else:
        citation_ok = not result.citations
        missing, missing_parts, forbidden, content_ok = False, [], [], True

    passed = status_ok and citation_ok and content_ok and not leak

    notes = [
        f"hit@5={'n/a' if hit is None else ('yes' if hit else 'no')}",
        *([f"coverage={coverage:.0%}"] if coverage is not None else []),
        f"status={result.status}{'' if status_ok else ' (expected ' + '|'.join(allowed_statuses(exp)) + ')'}",
        f"cited={','.join(cited_sections) or '-'}{'' if citation_ok else ' (bad)'}",
    ]
    if missing:
        notes.append(f"missing expected fact{' ' + str(missing_parts) if missing_parts else ''}")
    if forbidden:
        notes.append(f"contains forbidden {forbidden}")
    if leak:
        notes.append(f"LEAK citations={len(foreign_citations)} names={foreign_names}")
    if result.status == "try_again":
        notes.append(f"infra: {result.reason}")
    elif result.reason not in ("ok",):
        notes.append(f"reason={result.reason}")

    return {"hit": hit, "coverage": coverage, "status_ok": status_ok, "citation_ok": citation_ok,
            "content_ok": content_ok, "leak": leak, "pass": passed,
            "infra_error": result.status == "try_again", "notes": "; ".join(notes)}


# ------------------------------------------------------------ running

def apply_judge(judge: LLMJudge, case: dict, result, scored: dict) -> None:
    """Replace the string-based pass with the judge's verdict. The leak check stays in code and vetoes a pass."""
    scored["strings_pass"] = scored["pass"]
    scored["judge_verdict"], scored["judge_reason"] = "skipped", ""
    if scored["infra_error"]:
        return                                          # the assistant never produced a reply: nothing to judge
    verdict = judge.judge(case, result)
    if verdict.error:
        scored.update(judge_verdict="error", judge_reason=verdict.reason, infra_error=True)
        scored["pass"] = False
        scored["notes"] += f"; judge unavailable ({verdict.reason})"
        return
    scored["judge_verdict"], scored["judge_reason"] = ("pass" if verdict.passed else "fail"), verdict.reason
    scored["pass"] = bool(verdict.passed) and not scored["leak"]
    agree = "agrees with" if scored["pass"] == scored["strings_pass"] else "DISAGREES with"
    scored["notes"] = (f"judge={scored['judge_verdict']}: {verdict.reason}; leak={'YES' if scored['leak'] else 'no'}; "
                       f"string check {agree} the judge (strings_pass={scored['strings_pass']}); " + scored["notes"])


def run_case_once(services: Services, case: dict, use_hyde: bool, use_mmr: bool,
                  rate_limit_wait: float, max_infra_retries: int):
    """Ask once. If the provider is rate-limiting (try_again with LLMUnavailable),
    the HARNESS waits and asks again; that wait is never part of the latency."""
    for attempt in range(max_infra_retries + 1):
        result = services.answerer.answer(principal_for(case), case["question"],
                                          use_hyde=use_hyde, use_mmr=use_mmr)
        rate_limited = result.status == "try_again" and "LLMUnavailable" in result.reason
        if not rate_limited:
            return result      # HyDE fallbacks are NOT retried: they are real behaviour, counted in the summary
        if attempt < max_infra_retries:
            log.warning("%s: provider unavailable (%s), waiting %.0fs", case["id"], result.reason, rate_limit_wait)
            time.sleep(rate_limit_wait)
    return result


def percentile(values: list[float], p: float) -> Optional[float]:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def run(cases: list[dict], services: Services, *, use_hyde: bool, use_mmr: bool = False, repeats: int = 1,
        pace_s: float = 0.0, rate_limit_wait: float = 30.0, max_infra_retries: int = 2,
        judge: Optional[LLMJudge] = None) -> tuple[list[dict], dict]:
    doc_names = {c: services.registry.get_document(c, v)["doc_name"]
                 for c, v in services.registry.active_versions().items()}
    versions = services.registry.active_versions()

    rows, all_runs = [], []
    last_calls = 0     # LLM calls the previous run made: pace by calls actually made, so questions
                       # refused by the similarity cutoff (0 calls) cost no waiting
    for i, case in enumerate(cases, start=1):
        runs = []
        for r in range(repeats):
            if pace_s and last_calls:
                time.sleep(pace_s * last_calls)
            result = run_case_once(services, case, use_hyde, use_mmr, rate_limit_wait, max_infra_retries)
            last_calls = result.llm_calls
            scored = score_run(case, result, doc_names, versions)
            if judge is not None:
                apply_judge(judge, case, result, scored)
            runs.append((result, scored))
            all_runs.append((case, result, scored))
        last_result, last_scored = runs[-1]
        n_pass = sum(s["pass"] for _, s in runs)
        statuses = [res.status for res, _ in runs]
        notes = last_scored["notes"] + f"; {n_pass}/{repeats} runs passed"
        if len(set(statuses)) > 1:
            notes += f"; INCONSISTENT statuses={statuses}"
        rows.append({
            "id": case["id"],
            "input": {"company": case["company"], "profile": case.get("profile", {}),
                      "question": case["question"], "category": case.get("category", "")},
            "expected": case["expected"],
            "actual": {
                "status": last_result.status,
                "answer": last_result.answer,
                "citations": last_result.citations,
                "retrieved_sections": [r["section_id"] for r in last_result.retrieved],
                "retrieved_chunks": last_result.retrieved,
                "latency_ms": last_result.latency_ms,
                "llm_calls": last_result.llm_calls,
                "hyde_used": last_result.hyde_used,
                "runs": [{"status": res.status, "pass": s["pass"], "latency_ms": res.latency_ms,
                          "llm_calls": res.llm_calls} for res, s in runs],
                **({"judge": {"model": judge.model, "verdict": last_scored["judge_verdict"],
                              "reason": last_scored["judge_reason"], "strings_pass": last_scored["strings_pass"]}}
                   if judge is not None else {}),
            },
            "pass": n_pass == repeats,
            "notes": notes,
        })
        print(f"[{i:2}/{len(cases)}] {'PASS' if n_pass == repeats else 'FAIL'} {case['id']:38} {notes}", flush=True)

    return rows, summarise(rows, all_runs, use_hyde, repeats, use_mmr, judge.model if judge is not None else None)


def summarise(rows: list[dict], all_runs: list[tuple], use_hyde: bool, repeats: int,
              use_mmr: bool = False, judge_model: Optional[str] = None) -> dict:
    def rate(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    measured = [(c, r, s) for c, r, s in all_runs if not s["infra_error"]]
    latencies = [r.latency_ms for _, r, _ in measured]
    llm_latencies = [r.latency_ms for _, r, _ in measured if r.llm_calls > 0]

    by_cat: dict[str, list] = defaultdict(list)
    for row in rows:
        by_cat[row["input"]["category"]].append(row["pass"])

    multi = [(c, r, s) for c, r, s in all_runs if s["coverage"] is not None]
    judged = [s for _, _, s in all_runs if s.get("judge_verdict") in ("pass", "fail")]
    return {
        "scoring": (f"LLM judge with reference answers ({judge_model}); leak check in code" if judge_model
                    else "string matching"),
        **({"judge_vs_strings_agree": f"{sum(s['pass'] == s['strings_pass'] for s in judged)}/{len(judged)}",
            "judge_errors": sum(s.get("judge_verdict") == "error" for _, _, s in all_runs)} if judge_model else {}),
        "use_hyde": use_hyde,
        "use_mmr": use_mmr,
        "repeats": repeats,
        "cases": len(rows),
        "cases_passed": sum(r["pass"] for r in rows),
        "case_pass_rate": rate([r["pass"] for r in rows]),
        "run_pass_rate": rate([s["pass"] for _, _, s in all_runs]),
        "retrieval_hit_at_5": rate([s["hit"] for _, _, s in all_runs]),
        "multi_section_coverage_at_5": rate([s["coverage"] for _, _, s in multi]),
        "multi_section_full_coverage": rate([s["coverage"] == 1.0 for _, _, s in multi]),
        "multi_section_run_pass_rate": rate([s["pass"] for _, _, s in multi]),
        "status_accuracy": rate([s["status_ok"] for _, _, s in all_runs]),
        "citation_accuracy": rate([s["citation_ok"] for _, _, s in all_runs]),
        "content_accuracy": rate([s["content_ok"] for _, _, s in all_runs]),
        "leaks": sum(s["leak"] for _, _, s in all_runs),
        "infra_errors": sum(s["infra_error"] for _, _, s in all_runs),
        "hyde_fallbacks": sum(1 for _, r, _ in all_runs if use_hyde and not r.hyde_used),
        "inconsistent_cases": sum("INCONSISTENT" in r["notes"] for r in rows),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_p95_llm_answered_path": percentile(llm_latencies, 95),
        "latency_ms_max": max(latencies) if latencies else None,
        "llm_calls_avg": rate([r.llm_calls for _, r, _ in all_runs]),
        "llm_calls_max": max((r.llm_calls for _, r, _ in all_runs), default=0),
        "max_citations": max((len(r.citations) for _, r, _ in all_runs), default=0),
        "status_counts": dict(Counter(r.status for _, r, _ in all_runs)),
        "pass_by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "failed_cases": [r["id"] for r in rows if not r["pass"]],
    }


def write_outputs(rows: list[dict], summary: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results_{tag}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            assert tuple(row) == RESULT_FIELDS
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [f"# Evaluation summary: {tag}", "", "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    (out_dir / f"summary_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------ threshold sweep (no LLM)

def sweep(cases: list[dict], services: Services, out_dir: Path) -> str:
    """Best raw-question similarity per case vs. what the gate SHOULD do.
    'must answer'  = only 'answered' is acceptable (gating it is an error)
    'must refuse'  = 'answered' is not acceptable (gating it is correct and saves a call)"""
    rows = []
    for case in cases:
        best = retrieve(services.store, services.registry, principal_for(case), case["question"]).best_similarity
        allowed = allowed_statuses(case["expected"])
        kind = "must answer" if allowed == ["answered"] else ("must refuse" if "answered" not in allowed else "either")
        rows.append((best, kind, case["id"]))
    rows.sort()

    lines = ["# Similarity threshold sweep", "", "Best raw-question similarity per case:", "",
             "| best similarity | gate should | case |", "|---|---|---|"]
    lines += [f"| {b:.3f} | {k} | {cid} |" for b, k, cid in rows]
    answer_sims = [b for b, k, _ in rows if k == "must answer"]
    refuse_sims = [b for b, k, _ in rows if k == "must refuse"]
    lines += ["", f"Lowest 'must answer' similarity: {min(answer_sims):.3f}",
              f"Highest 'must refuse' similarity: {max(refuse_sims):.3f}", "",
              "| threshold | must-answer cases wrongly gated | must-refuse cases caught by gate |", "|---|---|---|"]
    for t in [x / 100 for x in range(20, 52, 2)]:
        wrongly = sum(b < t for b in answer_sims)
        caught = sum(b < t for b in refuse_sims)
        lines.append(f"| {t:.2f} | {wrongly}/{len(answer_sims)} | {caught}/{len(refuse_sims)} |")
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "threshold_sweep.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------ MMR comparison (retrieval only, no LLM)

def mmr_compare(cases: list[dict], services: Services, out_dir: Path, lam: float, pool: int) -> str:
    """Deterministic before/after of MMR re-ranking on what the model would be shown.
    Two groups of cases:
      single  = one expected section (the 'original' style cases): MMR must not make these worse
      multi   = needs facts from two sections ('required_sections'): where crowding can hurt
    Metrics: hit@3 for single; group coverage and full coverage for multi; distinct sections in top 3."""
    single = [c for c in cases if c["expected"].get("sections") and not c["expected"].get("required_sections")]
    multi = [c for c in cases if c["expected"].get("required_sections")]

    def top3(case: dict, mmr_lam: Optional[float], mmr_pool: Optional[int]):
        r = retrieve(services.store, services.registry, principal_for(case), case["question"],
                     use_mmr=mmr_lam is not None, mmr_lambda=mmr_lam or 1.0,
                     candidates=mmr_pool or services.settings.candidates_per_query, top_k=3)
        return [c.chunk_id.split("__", 2)[2].replace("__s0", "") for c in r.chunks], [c.metadata["section_id"] for c in r.chunks]

    def covered(case: dict, secs: list[str]) -> int:
        return sum(any(s in secs for s in g) for g in case["expected"]["required_sections"])

    grid: list[tuple[str, Optional[float], Optional[int]]] = [("baseline (plain top-3)", None, None)]
    grid += [(f"MMR lambda={l} pool={p}", l, p) for p in (6, 10, 15) for l in (0.9, 0.8, 0.7, 0.6, 0.5)]

    total_groups = sum(len(c["expected"]["required_sections"]) for c in multi)
    base_single = {c["id"]: any(s in top3(c, None, None)[1] for s in c["expected"]["sections"]) for c in single}
    base_multi = {c["id"]: covered(c, top3(c, None, None)[1]) for c in multi}

    lines = ["# MMR re-ranking: retrieval-only comparison", "",
             f"Same questions, same chunks; only the choice of the 3 chunks shown to the model changes. No LLM involved.",
             f"- single-section cases (must not get worse): {len(single)}",
             f"- two-section cases ('multi_section'): {len(multi)}, {total_groups} required section groups", "",
             "| setting | single: hit@3 | single: cases worse / better | multi: groups covered | multi: cases fully covered | multi: worse / better | avg distinct sections in top 3 |",
             "|---|---|---|---|---|---|---|"]
    for name, mmr_lam, mmr_pool in grid:
        s_hits, s_worse, s_better, m_cov, m_full, m_worse, m_better, distinct = 0, 0, 0, 0, 0, 0, 0, []
        for c in single:
            _, secs = top3(c, mmr_lam, mmr_pool)
            hit = any(s in secs for s in c["expected"]["sections"])
            s_hits += hit
            s_worse += base_single[c["id"]] and not hit
            s_better += hit and not base_single[c["id"]]
            distinct.append(len(set(secs)))
        for c in multi:
            _, secs = top3(c, mmr_lam, mmr_pool)
            n = covered(c, secs)
            m_cov += n
            m_full += n == len(c["expected"]["required_sections"])
            m_worse += n < base_multi[c["id"]]
            m_better += n > base_multi[c["id"]]
            distinct.append(len(set(secs)))
        lines.append(f"| {name} | {s_hits}/{len(single)} | {s_worse} / {s_better} | {m_cov}/{total_groups} | "
                     f"{m_full}/{len(multi)} | {m_worse} / {m_better} | {sum(distinct) / len(distinct):.2f} |")

    lines += ["", f"## Per-question detail at the chosen setting (lambda={lam}, pool={pool})", "",
              "| case | needs | baseline top-3 | MMR top-3 | coverage before -> after |", "|---|---|---|---|---|"]
    for c in multi:
        ids_b, secs_b = top3(c, None, None)
        ids_m, secs_m = top3(c, lam, pool)
        need = " AND ".join("/".join(g) for g in c["expected"]["required_sections"])
        n = len(c["expected"]["required_sections"])
        before, after = covered(c, secs_b), covered(c, secs_m)
        mark = " **better**" if after > before else (" **WORSE**" if after < before else "")
        lines.append(f"| {c['id']} | {need} | {', '.join(ids_b)} | {', '.join(ids_m)} | {before}/{n} -> {after}/{n}{mark} |")

    changed = []
    for c in single:
        ids_b, secs_b = top3(c, None, None)
        ids_m, secs_m = top3(c, lam, pool)
        if set(ids_b) != set(ids_m):
            hb = any(s in secs_b for s in c["expected"]["sections"])
            ha = any(s in secs_m for s in c["expected"]["sections"])
            changed.append(f"| {c['id']} | {', '.join(ids_b)} | {', '.join(ids_m)} | {'yes' if hb else 'no'} -> {'yes' if ha else 'no'} |")
    lines += ["", f"## Single-section cases whose top 3 changed under MMR ({len(changed)} of {len(single)})", "",
              "| case | baseline top-3 | MMR top-3 | expected section found |", "|---|---|---|---|"] + changed
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mmr_retrieval.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["groq", "fake"], default="groq")
    ap.add_argument("--hyde", choices=["auto", "on", "off"], default="auto",
                    help="auto = whatever the app does (USE_HYDE, default on); on/off force it, for before/after runs")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--pace", type=float, default=None,
                    help="seconds to wait after each LLM call made (default 15 for groq, 0 for fake). "
                         "Groq's free tier allows 8000 tokens/minute; an answer call with 5 excerpts is about 2000 tokens, "
                         "so about 4 calls a minute.")
    ap.add_argument("--rate-limit-wait", type=float, default=30.0)
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--mmr", choices=["on", "off"], default="off", help="MMR re-ranking of the top 3")
    ap.add_argument("--mmr-lambda", type=float, default=None)
    ap.add_argument("--mmr-pool", type=int, default=None)
    ap.add_argument("--min-similarity", type=float, default=None)
    ap.add_argument("--judge", choices=["auto", "llm", "strings"], default="auto",
                    help="how to decide pass/fail: llm = a judge model that knows the reference answer; strings = "
                         "substring checks; auto = llm for --llm groq, strings for --llm fake")
    ap.add_argument("--only", default=None, help="comma-separated case ids to run (default: all)")
    ap.add_argument("--sweep", action="store_true", help="threshold sweep only (no LLM calls)")
    ap.add_argument("--mmr-compare", action="store_true", help="MMR before/after on retrieval only (no LLM calls)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cases = load_cases(args.cases)
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        missing = [x for x in wanted if x not in {c["id"] for c in cases}]
        if missing:
            sys.exit(f"unknown case id(s): {missing}")
        cases = [c for c in cases if c["id"] in wanted]
    overrides = {} if args.min_similarity is None else {"min_similarity": args.min_similarity}
    if args.mmr_lambda is not None:
        overrides["mmr_lambda"] = args.mmr_lambda
    if args.mmr_pool is not None:
        overrides["mmr_pool"] = args.mmr_pool

    with tempfile.TemporaryDirectory() as tmp:
        services = build_eval_services("fake" if (args.sweep or args.mmr_compare) else args.llm, Path(tmp), **overrides)
        if args.sweep:
            print(sweep(cases, services, args.out))
            return
        if args.mmr_compare:
            print(mmr_compare(cases, services, args.out, services.settings.mmr_lambda, services.settings.mmr_pool))
            return
        if args.llm == "groq" and not services.settings.groq_key:
            sys.exit("groq_key is not set; use --llm fake for an offline run")
        pace = args.pace if args.pace is not None else (15.0 if args.llm == "groq" else 0.0)
        use_judge = args.judge == "llm" or (args.judge == "auto" and args.llm == "groq")
        judge = None
        if use_judge:
            if args.llm == "fake":
                sys.exit("--judge llm needs a real model; use --judge strings with --llm fake")
            missing_ref = [c["id"] for c in cases if not c.get("reference")]
            if missing_ref:
                sys.exit(f"--judge llm needs a reference answer in every case; missing: {missing_ref[:5]}")
            cfg = services.settings
            judge_llm = GroqLLM(cfg.groq_judge_key, cfg.groq_base_url) if cfg.groq_judge_key else services.llm
            judge = LLMJudge(judge_llm, cfg.judge_model, cfg.judge_reasoning_effort)
        use_hyde = services.settings.use_hyde if args.hyde == "auto" else args.hyde == "on"
        rows, summary = run(cases, services, use_hyde=use_hyde, use_mmr=args.mmr == "on",
                            repeats=args.repeats, pace_s=pace, rate_limit_wait=args.rate_limit_wait, judge=judge)

    tag = f"{args.llm}_hyde_{'on' if use_hyde else 'off'}" + ("_mmr" if args.mmr == "on" else "")
    path = write_outputs(rows, summary, args.out, tag)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
