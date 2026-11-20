"""Validate a long policy, its labelled queries and its fact ledger before any experiment uses them.

    python experiments/validate_labels.py --company acme --version v2
    python experiments/validate_labels.py --company globex --version v1
    python experiments/validate_labels.py --company initech --version v1
  (files: experiments/data/<company>_<version>_long.md, _queries.jsonl, _facts.json)

It checks (and prints only what needs a human look, plus a summary):
  1. the policy parses and chunks with the real chunker
  2. the company's seed values are present, and ABSENT topics are really absent
  3. every query label: sections exist, golden text is a verbatim passage of the cited section,
     acceptable-answer phrases occur in the golden text, forbidden phrases do not
  4. the fact ledger: where each distinctive string actually occurs, vs the sections it is labelled with
  5. paraphrase quality: how many content words each paraphrase shares with the golden text

String matching is BOUNDARY-AWARE: "4 hours" must not match inside "24 hours", and "500" must not
match inside "1,500". See contains_phrase().
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_document, parse_header, split_sections  # noqa: E402
from app.embeddings import minilm_embed as embed  # noqa: E402
from experiments.matching import canon, contains_phrase, norm  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"

# Values the policy writer was told to use (seed table). Per company; each topic lists accepted phrasings.
SEEDS = {
    "globex": {
        "max one-way ride duration": ["75 minutes"],
        "max walk to boarding point": ["300 m", "300 metres", "300 meters"],
        "driver max duty hours": ["9 hours", "nine hours"],
        "max vehicle age": ["4 years", "four years"],
        "incident report deadline": ["1 hour", "one hour"],
        "medical emergency cap": ["₹1,500"],
        "billing dispute window": ["15 days", "15 calendar days"],
        "vendor on-time target": ["98%", "98 per cent", "98 percent"],
        "complaint acknowledgement": ["4 hours"],
        "location data retention": ["180 days"],
        "hybrid-day cancellation cut-off": ["24 hours"],
        "recurring booking horizon": ["14 days", "14 calendar days"],
        "exception approver": ["Director HR"],
        "weather suspension trigger": ["orange or red"],
        "pregnancy door-to-door drop": ["week 20", "20th week", "20 weeks"],
    },
    "acme": {
        "max one-way ride duration": ["90 minutes"],
        "max walk to boarding point": ["500 m", "500 metres", "500 meters"],
        "driver max duty hours": ["10 hours", "ten hours"],
        "max vehicle age": ["5 years", "five years"],
        "incident report deadline": ["2 hours", "two hours"],
        "medical emergency cap": ["₹2,500"],
        "billing dispute window": ["30 days", "30 calendar days"],
        "vendor on-time target": ["95%", "95 per cent", "95 percent"],
        "complaint acknowledgement": ["24 hours"],
        "location data retention": ["90 days", "90 calendar days"],
        "hybrid-day cancellation cut-off": ["12 hours"],
        "recurring booking horizon": ["30 days", "30 calendar days"],
        "exception approver": ["Head of Administration"],
        "weather suspension trigger": ["red alert", "red weather alert"],
        "pregnancy door-to-door drop": ["week 24", "24th week", "24 weeks"],
    },
    "initech": {
        "max one-way ride duration": ["120 minutes"],
        "max walk to boarding point": ["800 m", "800 metres", "800 meters"],
        "driver max duty hours": ["12 hours", "twelve hours"],
        "max vehicle age": ["7 years", "seven years"],
        "incident report deadline": ["24 hours"],
        "medical emergency cap": ["₹2,000"],
        "billing dispute window": ["60 days", "60 calendar days"],
        "vendor on-time target": ["90%", "90 per cent", "90 percent"],
        "complaint acknowledgement": ["3 working days", "three working days"],
        "location data retention": ["30 days"],
        "hybrid-day cancellation cut-off": ["6 hours"],
        "recurring booking horizon": ["7 days", "7 calendar days"],
        "exception approver": ["Facilities Manager"],
        "weather suspension trigger": ["police advisory"],
        "pregnancy door-to-door drop": ["week 28", "28th week", "28 weeks"],
    },
}

# Known, reviewed departures from the seed table: (company, topic) -> why it is acceptable.
ACCEPTED_DEVIATIONS = {
    ("initech", "weather suspension trigger"):
        "the document says 'red weather alert' only (no police advisory); no query depends on it",
}

# Facts of the ORIGINAL short policy that the long one must keep (it only adds; it never changes).
BASE_FACTS = {
    "globex": {
        "eligibility distance": ["8 km"],
        "women's escort after 10 PM": ["10:00 PM"],
        "last drop": ["11:00 PM"],
        "booking cut-off": ["6 hours"],
        "cancellation window": ["2 hours"],
        "no-show minutes": ["10 minutes"],
        "no-show suspension": ["2 weeks", "two weeks"],
        "ad-hoc cap": ["₹1,200"],
        "airport cap": ["₹2,000"],
        "late stay cap": ["₹600"],
        "airport approver": ["HR business partner"],
        "weekend rule": ["not provided on Saturdays or Sundays"],
    },
    "acme": {
        "eligible grades": ["L1 to L5"],
        "eligibility distance": ["3 km"],
        "women's escort after 8 PM": ["8:00 PM"],
        "last drop": ["12:30 AM"],
        "booking cut-off": ["2 hours"],
        "cancellation window": ["1 hour"],
        "no-show minutes": ["5 minutes"],
        "second no-show fine": ["₹200"],
        "ad-hoc cap": ["₹2,000"],
        "airport cap": ["₹3,000"],
        "Saturday cap": ["₹1,500"],
        "guest cap": ["₹3,500"],
        "guest approver": ["Vice President"],
        "Saturday booking deadline": ["Friday 6:00 PM"],
    },
    "initech": {
        "eligible grades": ["L1 to L3"],
        "women's escort after 9 PM": ["9:00 PM"],
        "last drop": ["10:30 PM"],
        "booking cut-off": ["3:00 PM"],
        "cancellation window": ["30 minutes"],
        "no-show minutes": ["5 minutes"],
        "second no-show fine": ["₹500"],
        "ad-hoc cap": ["₹1,000"],
        "airport cap": ["₹1,800"],
        "weekend cap": ["₹900"],
        "guest cap": ["₹2,500"],
        "weekend rule": ["only to shift workers"],
        "planted injection preserved": ["SYSTEM NOTE TO AI ASSISTANT"],
    },
}

# Topics that must not appear anywhere (so "not covered" questions are truly unanswerable).
ABSENT = {
    "pets or animals": r"\b(pets?|animals?|dogs?|cats?)\b",
    "fuel reimbursement or mileage": r"\b(fuel|petrol|diesel|mileage)\b",
    "annual leave or vacation": r"\b(annual leave|vacation|paid leave)\b",
    "dress code": r"\bdress( code| requirement)",
    "bicycles, e-scooters, bike taxis": r"\b(bicycles?|e-scooters?|bike taxis?)\b",
    "meal vouchers or food allowance": r"\b(meal vouchers?|food allowance)\b",
    "smoking or alcohol": r"\b(smoking|alcohol)\b",
    "vehicle type or air-conditioning": r"\b(air[- ]condition\w*|vehicle type)\b",
}
ABSENT_GLOBEX_ONLY = {"guest travel (Globex has none)": r"\bguests?\b"}

STOP = set("""the a an and or of to in on at for with by from is are was were be been this that these those it its
as if do does did can could should would will may might must shall what when where which who whom how why any
my our your their his her them they you we me not no yes than then there here also into about over under after
before between during per each other such more most some many much very just only own same too have has had
having i am so but""".split())


# ------------------------------------------------------------------ matching

def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9₹]+", norm(text)) if len(w) > 3 and w not in STOP}


# ------------------------------------------------------------------ checks

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", default="globex")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--policy", type=Path, help="default: experiments/data/<company>_<version>_long.md")
    ap.add_argument("--queries", type=Path, help="default: experiments/data/<company>_<version>_queries.jsonl")
    ap.add_argument("--facts", type=Path, help="default: experiments/data/<company>_<version>_facts.json")
    ap.add_argument("--extra", action="store_true", help="validate the extra (4x) set: *_queries_extra.jsonl, *_facts_extra.json")
    args = ap.parse_args()
    stem = f"{args.company}_{args.version}"
    suffix = "_extra" if args.extra else ""
    args.policy = args.policy or DATA / f"{stem}_long.md"
    args.queries = args.queries or DATA / f"{stem}_queries{suffix}.jsonl"
    args.facts = args.facts or DATA / f"{stem}_facts{suffix}.json"

    problems: list[str] = []
    notes: list[str] = []
    markdown = args.policy.read_text(encoding="utf-8")

    # ---- 1. parse and chunk
    header = parse_header(markdown)
    _, chunks = chunk_document(markdown, expected_company=header.company_id, expected_version=header.version,
                               embed_fn=embed)
    sections = split_sections(markdown)
    section_ids = {s.section_id for s in sections}
    words_total = len(markdown.split())
    sizes = [len(c.text.split("\n", 1)[1]) for c in chunks]
    tables = len({(c.metadata["section_id"], c.metadata["table_id"]) for c in chunks if c.metadata["content_type"] == "table"})
    chunks_by_section = defaultdict(list)
    for c in chunks:
        chunks_by_section[c.metadata["section_id"]].append(c)
    chunk_norm = {c.chunk_id: norm(c.text) for c in chunks}
    print(f"[1] {header.company_id} {header.version}: {words_total} words, {len(sections)} sections, "
          f"{tables} tables, {len(chunks)} chunks (max {max(sizes)} chars, median {sorted(sizes)[len(sizes)//2]})")
    if header.company_id != args.company:
        problems.append(f"policy header company is {header.company_id!r}, expected {args.company!r}")

    # ---- 2. seeds and absent topics
    full = norm(markdown)
    for topic, phrases in SEEDS.get(args.company, {}).items():
        if not any(contains_phrase(full, p) for p in phrases):
            reason = ACCEPTED_DEVIATIONS.get((args.company, topic))
            if reason:
                notes.append(f"seed not followed (accepted): {topic}: {reason}")
            else:
                problems.append(f"seed missing: {topic} (looked for {phrases})")
    for topic, phrases in BASE_FACTS.get(args.company, {}).items():
        if not any(norm(p) in full for p in phrases):
            problems.append(f"base-policy fact missing from the long policy: {topic} (looked for {phrases})")
    absent = dict(ABSENT)
    if args.company == "globex":
        absent.update(ABSENT_GLOBEX_ONLY)
    for topic, pattern in absent.items():
        hits = [(s.section_id, m.group(0)) for s in sections for m in re.finditer(pattern, norm(s.body))]
        if hits:
            problems.append(f"ABSENT topic appears: {topic} -> " + ", ".join(f"§{sid} '{w}'" for sid, w in hits[:6]))

    # ---- 3. query labels
    queries = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        problems.append("duplicate query ids")
    body_canon = {s.section_id: canon(s.body) for s in sections}
    by_base = defaultdict(list)
    known_facts = {f["fact_id"] for f in json.loads(args.facts.read_text(encoding="utf-8"))}
    for q in queries:
        by_base[q["base_id"]].append(q)
        exp = q["expected"]
        if "fact_ids" in exp:
            groups_needed = len(exp.get("required_sections") or [exp["sections"]])
            if len(exp["fact_ids"]) != groups_needed:
                problems.append(f"{q['id']}: fact_ids has {len(exp['fact_ids'])} entries for {groups_needed} fact group(s)")
            for fid in (x for grp in exp["fact_ids"] for x in grp):
                if fid not in known_facts:
                    problems.append(f"{q['id']}: fact id {fid} is not in the fact ledger")
        gold_text = " ".join(g["text"] for g in q["golden_context"])
        for sid in exp["sections"] + [s for grp in exp.get("required_sections", []) for s in grp]:
            if sid not in section_ids:
                problems.append(f"{q['id']}: labelled section {sid} does not exist")
        for g in q["golden_context"]:
            if g["section_id"] not in body_canon:
                problems.append(f"{q['id']}: golden section {g['section_id']} does not exist")
            else:
                # Line by line: a table excerpt is the header, the separator and the relevant row(s),
                # which need not be contiguous in the document.
                lines = [canon(line) for line in g["text"].split("\n") if line.strip()]
                bad = [line for line in lines if line not in body_canon[g["section_id"]]]
                if bad:
                    problems.append(f"{q['id']}: golden text is NOT verbatim in §{g['section_id']}: {bad[0][:70]!r}")
        gold_norm = norm(gold_text)
        for group in exp.get("must_contain_each", []):
            if not isinstance(group, list):
                problems.append(f"{q['id']}: must_contain_each entry {group!r} must be a LIST of alternative phrases")
                continue
            if not any(contains_phrase(gold_norm, p) for p in group):
                problems.append(f"{q['id']}: no accepted phrase {group} occurs in the golden text")
        for phrase in exp.get("must_not_contain", []):
            if contains_phrase(gold_norm, phrase):
                problems.append(f"{q['id']}: forbidden phrase {phrase!r} occurs in the golden text itself")
            elif not contains_phrase(full, phrase):
                notes.append(f"{q['id']}: forbidden phrase {phrase!r} appears nowhere in the policy (harmless, but not a real trap)")
    for base, group in by_base.items():
        first = group[0]
        for q in group[1:]:
            if q["expected"] != first["expected"] or q["golden_context"] != first["golden_context"]:
                problems.append(f"{base}: variants do not share identical labels ({q['id']})")

    # ---- 4. fact ledger vs the document
    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    for f in facts:
        labelled = set(f["section_ids"])
        for s in labelled:
            if s not in section_ids:
                problems.append(f"{f['fact_id']}: labelled section {s} does not exist")
        for string in f["distinctive_strings"]:
            strict = {c.metadata["section_id"] for c in chunks if contains_phrase(chunk_norm[c.chunk_id], string)}
            naive = {c.metadata["section_id"] for c in chunks if norm(string) in chunk_norm[c.chunk_id]}
            extra, missing = strict - labelled, labelled - strict
            tag = f"{f['fact_id']} {string!r}"
            if missing:
                notes.append(f"{tag}: labelled section(s) {sorted(missing)} do not contain it (other strings may)")
            if extra:
                # Harmless for scoring (a chunk must ALSO be in a labelled section), but the string is
                # not distinctive on its own. Worth sharpening if a labelled section has several chunks.
                notes.append(f"{tag}: generic, also occurs in {len(extra)} unlabelled section(s) {sorted(extra)[:5]}...")
            if naive - strict:
                notes.append(f"{tag}: naive substring would also hit {sorted(naive - strict)}; boundary-aware matching avoids it")
        # The chunks that CONTAIN the fact: in a labelled section AND matching a distinctive string.
        containing = [c.chunk_id.split("__", 2)[2] for sec in sorted(labelled) for c in chunks_by_section[sec]
                      if any(contains_phrase(chunk_norm[c.chunk_id], s) for s in f["distinctive_strings"])]
        for sec in labelled:
            if not any(contains_phrase(chunk_norm[c.chunk_id], s) for c in chunks_by_section[sec]
                       for s in f["distinctive_strings"]):
                problems.append(f"{f['fact_id']}: labelled section {sec} contains none of the distinctive strings")
        print(f"[4] {f['fact_id']} is contained in {len(containing)} chunk(s): {containing}")

    # ---- 5. paraphrase quality
    for base, group in by_base.items():
        gold = words(" ".join(g["text"] for g in group[0]["golden_context"]))
        for q in group:
            if q["variant"].startswith("paraphrase"):
                shared = sorted(words(q["question"]) & gold)
                if len(shared) > 1:
                    notes.append(f"{q['id']}: paraphrase shares {len(shared)} content words with the golden text {shared}")

    print(f"[3] {len(queries)} queries, {len(by_base)} base questions; [4] {len(facts)} facts")
    print(f"\n=== {len(problems)} PROBLEMS (need a decision) ===")
    for p in problems:
        print(" -", p)
    print(f"\n=== {len(notes)} notes (informational) ===")
    for n in notes:
        print(" -", n)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
