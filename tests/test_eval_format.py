"""The evaluation set runs offline and writes the required JSONL format."""
import json

from eval.run_eval import (
    RESULT_FIELDS, build_eval_services, load_cases, percentile, run, write_outputs,
)


def test_cases_file_is_valid():
    cases = load_cases()
    assert len(cases) >= 10
    categories = {c["category"] for c in cases}
    assert {"paraphrase", "cross_company", "not_covered", "injection", "version", "table",
            "multi_section"} <= categories


def test_extended_set_contains_the_main_set():
    from eval.run_eval import ROOT
    main = {c["id"]: c for c in load_cases()}
    extended = {c["id"]: c for c in load_cases(ROOT / "eval" / "cases_extended.jsonl")}
    assert len(main) == 36 and len(extended) == 61
    assert all(extended[i] == c for i, c in main.items())      # identical definitions, no drift


def test_multi_section_cases_declare_required_section_groups():
    from eval.run_eval import ROOT
    main_multi = [c for c in load_cases() if c["category"] == "multi_section"]
    ext_multi = [c for c in load_cases(ROOT / "eval" / "cases_extended.jsonl") if c["category"] == "multi_section"]
    assert len(main_multi) >= 8 and len(ext_multi) == 20
    for c in ext_multi:
        groups = c["expected"]["required_sections"]
        assert len(groups) >= 2 and all(isinstance(g, list) and g for g in groups), c["id"]
        assert {s for g in groups for s in g} <= set(c["expected"]["sections"]), c["id"]
        assert c["expected"]["must_contain_each"], c["id"]


def _stub_result(answer, retrieved_sections, cited_sections=("4.2",), status="answered"):
    from types import SimpleNamespace
    return SimpleNamespace(
        status=status, answer=answer, reason="ok",
        retrieved=[{"section_id": s} for s in retrieved_sections],
        citations=[{"doc_name": "Acme Corp Employee Transport Policy", "version": "v2", "section_id": s}
                   for s in cited_sections] if status == "answered" else [],
    )


def test_score_run_needs_every_fact_group_and_reports_coverage():
    from eval.run_eval import score_run
    case = {"id": "x", "company": "acme", "question": "q", "expected": {
        "status": "answered", "sections": ["4.2", "7.1"], "required_sections": [["4.2"], ["7.1", "7.2"]],
        "must_contain_each": [["no-show"], ["200"]]}}
    docs, versions = {"acme": "Acme Corp Employee Transport Policy"}, {"acme": "v2"}

    both = score_run(case, _stub_result("It counts as a no-show and costs 200.", ["4.2", "7.1", "3.1"]), docs, versions)
    assert both["pass"] and both["coverage"] == 1.0

    one_fact = score_run(case, _stub_result("It counts as a no-show.", ["4.2", "7.1", "3.1"]), docs, versions)
    assert not one_fact["content_ok"] and not one_fact["pass"]
    assert "missing expected fact ['200']" in one_fact["notes"]

    half = score_run(case, _stub_result("It counts as a no-show and costs 200.", ["4.2", "3.1", "5.1"]), docs, versions)
    assert half["coverage"] == 0.5          # retrieval covered only one of the two required groups


def test_offline_eval_writes_required_fields(tmp_path):
    cases = [c for c in load_cases() if c["id"] in
             {"direct-acme-cutoff", "notcov-acme-pets", "version-acme-second-noshow"}]
    services = build_eval_services("fake", tmp_path / "data", min_similarity=0.35)
    rows, summary = run(cases, services, use_hyde=True, repeats=2)
    path = write_outputs(rows, summary, tmp_path / "out", "fake_test")

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    for line in lines:
        assert tuple(line) == RESULT_FIELDS
        assert isinstance(line["pass"], bool) and isinstance(line["notes"], str)
        assert len(line["actual"]["runs"]) == 2
    assert summary["llm_calls_max"] <= 2 and summary["max_citations"] <= 3
    assert summary["leaks"] == 0
    # Acme answers come from v2 after the upgrade.
    noshow = next(l for l in lines if l["id"] == "version-acme-second-noshow")
    assert all(c["version"] == "v2" for c in noshow["actual"]["citations"])


def test_scorer_treats_non_breaking_hyphens_and_spaces_as_ordinary_ones():
    """The answer model writes "no‑show" (non-breaking hyphen); that must still match the expected "no-show"."""
    from eval.run_eval import norm
    assert norm("A second no‑show costs ₹200") == norm("a second no-show costs ₹200")
    assert norm("2 weeks") == "2 weeks"
