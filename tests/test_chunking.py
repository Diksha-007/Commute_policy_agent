"""Chunking tests: structure, semantic text splitting, table splitting, validation.

Most tests use a fake keyword embedder so the expected split points are exact.
Tests that run whole documents use the shared deterministic test embedder (tests/fake_embedder.py). The real-model
checks live in tests/test_real_embedder.py (opt-in: pytest -m real_model).
"""
from pathlib import Path

import numpy as np
import pytest

from app.chunking import (
    IngestValidationError,
    TableBlock,
    TextBlock,
    chunk_document,
    consecutive_distances,
    parse_header,
    semantic_split,
    split_blocks,
    split_sections,
    split_sentences,
    table_chunks,
)
from app.embeddings import embed

POLICIES = Path(__file__).resolve().parent.parent / "policies"
ALL_POLICIES = [("acme", "v1"), ("acme", "v2"), ("globex", "v1"), ("initech", "v1")]


def load(company: str, version: str) -> str:
    return (POLICIES / f"{company}_{version}.md").read_text(encoding="utf-8")


# A deterministic fake embedder: one dimension per topic, counted by keywords.
TOPICS = {
    "escort": ["guard", "escort", "security", "night"],
    "fines": ["fine", "penalty", "payroll", "deducted"],
    "booking": ["book", "portal", "cut-off"],
}


def fake_embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        low = t.lower()
        v = [sum(low.count(w) for w in words) for words in TOPICS.values()] + [0.01]
        vecs.append(v)
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def doc(body: str, company: str = "testco", version: str = "v1") -> str:
    return (
        f"# Test Policy\n<!-- company_id: {company} | version: {version} | effective_date: 2026-01-01 -->\n\n"
        + body
    )


# ------------------------------------------------------------- pass 1: structure

def test_sections_parsed_with_ids_titles_and_parents():
    sections = split_sections(load("acme", "v1"))
    by_id = {s.section_id: s for s in sections}
    assert len(sections) == 17
    assert by_id["3.2"].title == "Security escort for women employees"
    assert by_id["3.2"].parent_title == "Night Shift"
    assert "8:00 PM" in by_id["3.2"].body
    assert "9" not in {s.section_id for s in sections}  # "## 9." only groups 9.1


@pytest.mark.parametrize("company,version", ALL_POLICIES)
def test_no_chunk_crosses_a_section_boundary(company, version):
    md = load(company, version)
    sections = {s.section_id: s for s in split_sections(md)}
    _, chunks = chunk_document(md, expected_company=company, expected_version=version, embed_fn=fake_embed)
    for c in chunks:
        section = sections[c.metadata["section_id"]]
        if c.metadata["content_type"] == "text":
            body = c.text.split("\n", 1)[1]
            for sentence in split_sentences(body):
                assert sentence in " ".join(section.body.split()), (c.chunk_id, sentence)


# ------------------------------------------------------------- pass 2: blocks

def test_table_is_separated_from_text_and_short_lead_in_becomes_caption():
    body = (
        "Long introductory paragraph about penalties that is clearly prose and more than the caption limit "
        "so it stays a separate text block and is not treated as a caption at all.\n\n"
        "Counted per month.\n"
        "| Count | Penalty |\n|---|---|\n| 1 | Warning |\n| 2 | Fine |\n\n"
        "Text after the table."
    )
    blocks = split_blocks(body, caption_max_chars=150)
    assert [type(b) for b in blocks] == [TextBlock, TableBlock, TextBlock]
    table = blocks[1]
    assert table.caption == "Counted per month."
    assert table.header == ("Count", "Penalty")
    assert table.rows == (("1", "Warning"), ("2", "Fine"))


def test_ragged_table_row_is_rejected():
    body = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 |\n"
    with pytest.raises(IngestValidationError, match="cells"):
        split_blocks(body)


# ------------------------------------------------------------- pass 3a: semantic

def test_semantic_split_breaks_between_two_topics():
    sentences = [
        "A security guard rides in the cab at night.",
        "The guard sits in the front seat as escort.",
        "Security staff verify the night escort identity.",
        "A fine of 200 rupees is deducted through payroll.",
        "The penalty doubles for a repeat fine.",
        "Payroll deducted penalty amounts appear on the payslip.",
    ]
    vecs = fake_embed(sentences)
    dists = consecutive_distances(vecs)
    threshold = float(np.percentile(dists, 85))
    chunks = semantic_split(sentences, dists, threshold, min_chars=50, max_chars=900)
    assert chunks == [" ".join(sentences[:3]), " ".join(sentences[3:])]


# ------------------------------------------------------------- pass 3b: tables

def test_table_rows_are_grouped_with_header_repeated_and_rows_linearised():
    table = TableBlock(
        header=("Shift", "Login"),
        rows=tuple((f"S{i}", f"{i}:00 AM") for i in range(1, 6)),
        caption="Shift slots.",
    )
    pieces = table_chunks(table, rows_per_chunk=4, max_chars=900)
    assert [(r0, r1) for _, r0, r1 in pieces] == [(1, 4), (5, 5)]
    for body, _, _ in pieces:
        assert body.startswith("Shift slots.\n| Shift | Login |")  # caption + header in every chunk
    assert "- Shift: S5; Login: 5:00 AM." in pieces[1][0]
    assert "S1" not in pieces[1][0]


def test_policy_table_chunks_have_row_metadata_and_ids():
    _, chunks = chunk_document(load("acme", "v1"), expected_company="acme", expected_version="v1",
                               embed_fn=fake_embed)
    tables = [c for c in chunks if c.metadata["section_id"] == "3.1" and c.metadata["content_type"] == "table"]
    assert [c.chunk_id for c in tables] == ["acme__v1__3.1__t0__r1-4", "acme__v1__3.1__t0__r5-5"]
    assert tables[1].metadata["row_start"] == 5
    assert "| Shift | Login | Logout | Drop departs |" in tables[1].text
    assert "Shift: Night" in tables[1].text


# ------------------------------------------------------------- whole documents

@pytest.mark.parametrize("company,version", ALL_POLICIES)
def test_real_policies_chunk_cleanly(company, version):
    header, chunks = chunk_document(load(company, version), expected_company=company,
                                    expected_version=version, embed_fn=embed)
    assert header.company_id == company
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.chunk_id.startswith(f"{company}__{version}__")
        assert c.metadata["company_id"] == company and c.metadata["version"] == version
        assert all(isinstance(v, (str, int, float, bool)) for v in c.metadata.values())
        body = c.text.split("\n", 1)[1]
        assert body.strip() and len(body) <= 900


def test_chunking_is_deterministic():
    md = load("acme", "v2")
    a = chunk_document(md, expected_company="acme", expected_version="v2", embed_fn=embed)[1]
    b = chunk_document(md, expected_company="acme", expected_version="v2", embed_fn=embed)[1]
    assert [(c.chunk_id, c.text) for c in a] == [(c.chunk_id, c.text) for c in b]


def test_breadcrumb_prefixes_every_chunk():
    _, chunks = chunk_document(load("globex", "v1"), expected_company="globex", expected_version="v1",
                               embed_fn=fake_embed)
    c = next(c for c in chunks if c.metadata["section_id"] == "3.2")
    assert c.text.startswith("Night Shift > 3.2 Security escort for women employees\n")


# ------------------------------------------------------------- validation

def test_missing_header_comment_is_rejected():
    with pytest.raises(IngestValidationError, match="header comment"):
        chunk_document("# Policy\n## 1. A\nText.", expected_company="x1", expected_version="v1",
                       embed_fn=fake_embed)


def test_company_mismatch_is_rejected():
    with pytest.raises(IngestValidationError, match="labelled company 'globex'"):
        chunk_document(load("globex", "v1"), expected_company="acme", expected_version="v1",
                       embed_fn=fake_embed)


def test_version_mismatch_is_rejected():
    with pytest.raises(IngestValidationError, match="labelled version 'v2'"):
        chunk_document(load("acme", "v2"), expected_company="acme", expected_version="v3",
                       embed_fn=fake_embed)


def test_duplicate_section_id_is_rejected():
    md = doc("## 1. A\n### 1.1 One\nText one.\n### 1.1 Again\nText two.\n")
    with pytest.raises(IngestValidationError, match="duplicate section"):
        split_sections(md)


def test_empty_section_is_rejected():
    md = doc("## 1. A\n### 1.1 Empty\n### 1.2 Full\nText.\n")
    with pytest.raises(IngestValidationError, match="empty"):
        split_sections(md)


