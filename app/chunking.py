"""Policy Markdown -> validated, citable chunks.

Three passes, each a small pure function so it can be tested on its own:

  Pass 1  split_sections()   Structural split on headings. Sections are HARD
                             boundaries: no chunk ever spans two sections, which is
                             what keeps "cite document and section" exact.
  Pass 2  split_blocks()     Inside a section, separate prose (TextBlock) from
                             Markdown tables (TableBlock).
  Pass 3a semantic_split()   Prose: split into sentences, embed them, and break
                             where consecutive sentences are far apart in meaning.
  Pass 3b table_chunks()     Tables: groups of rows, each chunk repeating the header
                             row, plus every row written out as a plain sentence.

chunk_document() runs the passes and validates the result. Any problem raises
IngestValidationError BEFORE anything is written to storage.

Expected Markdown shape:

    # Acme Corp Employee Transport Policy
    <!-- company_id: acme | version: v2 | effective_date: 2026-10-01 -->
    ## 3. Night Shift
    ### 3.2 Security escort for women employees
    Prose...
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from app.embeddings import EmbedFn

# ---------------------------------------------------------------- patterns

HEADER_RE = re.compile(
    r"<!--\s*company_id:\s*([^|]+?)\s*\|\s*version:\s*([^|]+?)\s*\|"
    r"\s*effective_date:\s*([^|]+?)\s*-->"
)
H1_RE = re.compile(r"^#\s+(.+?)\s*$")
H2_RE = re.compile(r"^##\s+(\d+)\.?\s+(.+?)\s*$")
H3_RE = re.compile(r"^###\s+(\d+\.\d+)\s+(.+?)\s*$")
ANY_HEADING_RE = re.compile(r"^#{1,6}\s")
TABLE_SEP_RE = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+$")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# IDs end up in Chroma collection names ("acme__v2"), so keep them strict.
# "_" is not allowed in a company id because "__" is our separator.
COMPANY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,30}$")
VERSION_RE = re.compile(r"^v\d{1,4}$")

# Abbreviations that end with "." but do not end a sentence.
ABBREVIATIONS = {"e.g.", "i.e.", "no.", "mr.", "mrs.", "ms.", "dr.", "etc.", "vs.", "approx.", "st."}

# Absolute safety cap on any chunk body (a single huge table row is the only
# way to get past max_chars; beyond this we refuse the document).
HARD_MAX_CHUNK_CHARS = 2000


class IngestValidationError(ValueError):
    """The document is malformed. Raised before any storage write."""


# ---------------------------------------------------------------- data types

@dataclass(frozen=True)
class DocHeader:
    company_id: str
    version: str
    effective_date: str
    doc_name: str


@dataclass(frozen=True)
class Section:
    section_id: str      # "3.2" (or "3" for a top-level section with no subsections)
    title: str           # "Security escort for women employees"
    parent_title: str    # "Night Shift"
    body: str


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class TableBlock:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""


Block = Union[TextBlock, TableBlock]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str            # breadcrumb line + body; this is what gets embedded and shown to the LLM
    metadata: dict       # scalar values only (Chroma requirement)


# ---------------------------------------------------------------- pass 0: header

def parse_header(markdown: str) -> DocHeader:
    doc_name = None
    for line in markdown.splitlines():
        if line.strip():
            m = H1_RE.match(line.strip())
            if m and not line.startswith("##"):
                doc_name = m.group(1)
            break
    if not doc_name:
        raise IngestValidationError("document must start with a '# <document name>' title line")

    m = HEADER_RE.search(markdown)
    if not m:
        raise IngestValidationError(
            "missing header comment: <!-- company_id: ... | version: ... | effective_date: ... -->"
        )
    company_id, version, effective_date = (g.strip() for g in m.groups())
    if not COMPANY_ID_RE.match(company_id):
        raise IngestValidationError(f"invalid company_id {company_id!r}")
    if not VERSION_RE.match(version):
        raise IngestValidationError(f"invalid version {version!r} (expected v1, v2, ...)")
    if not DATE_RE.match(effective_date):
        raise IngestValidationError(f"invalid effective_date {effective_date!r} (expected YYYY-MM-DD)")
    return DocHeader(company_id, version, effective_date, doc_name)


# ---------------------------------------------------------------- pass 1: sections

def split_sections(markdown: str) -> list[Section]:
    """Split on ## / ### headings. Each ### becomes a Section; a ## with no ###
    children becomes a Section itself. Text under a ## before its first ### is
    kept as its own Section (id "3")."""
    raw: list[dict] = []          # in document order
    current: Optional[dict] = None
    h2: Optional[tuple[str, str]] = None   # (number, title)
    seen_h1 = False

    for lineno, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        m3 = H3_RE.match(stripped)
        m2 = H2_RE.match(stripped)
        if m3:
            sid, title = m3.groups()
            if h2 is None:
                raise IngestValidationError(f"line {lineno}: subsection {sid} appears before any '##' section")
            if not sid.startswith(h2[0] + "."):
                raise IngestValidationError(
                    f"line {lineno}: subsection {sid} is numbered outside its section {h2[0]}"
                )
            current = {"id": sid, "title": title, "parent": h2[1], "lines": [], "intro": False}
            raw.append(current)
        elif m2:
            h2 = m2.groups()
            current = {"id": h2[0], "title": h2[1], "parent": h2[1], "lines": [], "intro": True}
            raw.append(current)
        elif ANY_HEADING_RE.match(stripped):
            if H1_RE.match(stripped) and not seen_h1 and current is None:
                seen_h1 = True
                continue
            raise IngestValidationError(f"line {lineno}: unsupported or malformed heading {stripped!r}")
        elif current is None:
            if stripped and not HEADER_RE.search(stripped):
                raise IngestValidationError(f"line {lineno}: content before the first '##' section")
        else:
            current["lines"].append(line)

    sections: list[Section] = []
    for i, r in enumerate(raw):
        body = "\n".join(r["lines"]).strip()
        if not body:
            has_children = (
                r["intro"] and i + 1 < len(raw) and raw[i + 1]["id"].startswith(r["id"] + ".")
            )
            if has_children:
                continue  # a "## 3. Night Shift" heading that only groups subsections
            raise IngestValidationError(f"section {r['id']} ({r['title']}) is empty")
        sections.append(Section(r["id"], r["title"], r["parent"], body))

    if not sections:
        raise IngestValidationError("document has no sections")
    ids = [s.section_id for s in sections]
    dups = sorted({x for x in ids if ids.count(x) > 1})
    if dups:
        raise IngestValidationError(f"duplicate section ids: {dups}")
    return sections


# ---------------------------------------------------------------- pass 2: blocks

def _cells(line: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in line.strip().strip("|").split("|"))


def split_blocks(body: str, caption_max_chars: int = 150) -> list[Block]:
    """Separate Markdown tables from prose. A short paragraph directly before a
    table (e.g. "No-shows are counted per calendar month.") becomes the table's
    caption, so it is repeated in every chunk of that table."""
    lines = body.splitlines()
    blocks: list[Block] = []
    buf: list[str] = []

    def flush_text():
        text = "\n".join(buf).strip()
        if text:
            blocks.append(TextBlock(text))
        buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1].strip()):
            # The last paragraph right before the table becomes its caption if short.
            paragraphs = re.split(r"\n\s*\n", "\n".join(buf).strip())
            caption = ""
            if paragraphs[-1] and len(paragraphs[-1]) <= caption_max_chars:
                caption = paragraphs.pop().strip()
                buf[:] = ["\n\n".join(paragraphs)]
            flush_text()
            header = _cells(line)
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                row = _cells(lines[j])
                if len(row) != len(header):
                    raise IngestValidationError(
                        f"table row has {len(row)} cells but header has {len(header)}: {lines[j].strip()!r}"
                    )
                rows.append(row)
                j += 1
            if not rows:
                raise IngestValidationError(f"table with header {header} has no rows")
            blocks.append(TableBlock(header, tuple(rows), caption))
            i = j
        else:
            buf.append(lines[i])
            i += 1
    flush_text()
    return blocks


# ---------------------------------------------------------------- pass 3a: semantic text

def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter. Bullet / numbered list items count as one
    sentence each. Does not split after abbreviations like 'e.g.' or 'No.'."""
    sentences: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph():
        para = " ".join(paragraph).strip()
        paragraph.clear()
        if not para:
            return
        start = 0
        for m in re.finditer(r"[.!?]\s+(?=[A-Z0-9₹\"'(])", para):
            candidate = para[start:m.start() + 1]
            words = candidate.split()
            if words and words[-1].lower() in ABBREVIATIONS:
                continue
            sentences.append(candidate.strip())
            start = m.end()
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)

    for line in text.splitlines():
        if not line.strip():
            flush_paragraph()
        elif BULLET_RE.match(line):
            flush_paragraph()
            sentences.append(BULLET_RE.sub("", line).strip())
        else:
            paragraph.append(line.strip())
    flush_paragraph()
    return [s for s in sentences if s]


def consecutive_distances(vectors: np.ndarray) -> list[float]:
    """Cosine distance between sentence i and i+1 (vectors are L2-normalised)."""
    return [float(1.0 - np.dot(vectors[i], vectors[i + 1])) for i in range(len(vectors) - 1)]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last resort for a single sentence longer than max_chars: split on words."""
    parts, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        parts.append(cur)
    return parts


def semantic_split(
    sentences: list[str],
    distances: list[float],
    threshold: float,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Group consecutive sentences into chunks.

    1. Break wherever the distance to the next sentence exceeds `threshold`.
    2. A group longer than max_chars is split at its largest internal distance
       (recursively).
    3. A group shorter than min_chars is merged into the neighbour it is MORE
       similar to (smaller boundary distance), if the result fits max_chars.
    4. A single sentence longer than max_chars is split on words.

    `distances[i]` is the distance between sentences[i] and sentences[i+1].
    """
    n = len(sentences)
    if n == 0:
        return []
    assert len(distances) == n - 1

    def text(g: tuple[int, int]) -> str:
        return " ".join(sentences[g[0]:g[1]])

    # 1. breakpoints
    groups: list[tuple[int, int]] = []
    start = 0
    for i, d in enumerate(distances):
        if d > threshold:
            groups.append((start, i + 1))
            start = i + 1
    groups.append((start, n))

    # 2. split oversized groups at the weakest link
    def split_big(g: tuple[int, int]) -> list[tuple[int, int]]:
        if len(text(g)) <= max_chars or g[1] - g[0] == 1:
            return [g]
        k = max(range(g[0], g[1] - 1), key=lambda i: distances[i])
        return split_big((g[0], k + 1)) + split_big((k + 1, g[1]))

    groups = [piece for g in groups for piece in split_big(g)]

    # 3. merge undersized groups into the more similar neighbour
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for idx, g in enumerate(groups):
            if len(text(g)) >= min_chars:
                continue
            options = []
            if idx > 0:
                options.append((distances[g[0] - 1], idx - 1))
            if idx < len(groups) - 1:
                options.append((distances[g[1] - 1], idx + 1))
            for _, nb in sorted(options):
                merged = (min(g[0], groups[nb][0]), max(g[1], groups[nb][1]))
                if len(text(merged)) <= max_chars:
                    lo = min(idx, nb)
                    groups[lo:lo + 2] = [merged]
                    changed = True
                    break
            if changed:
                break

    # 4. hard-split any single sentence still over the limit
    out: list[str] = []
    for g in groups:
        t = text(g)
        out.extend([t] if len(t) <= max_chars else _hard_split(t, max_chars))
    return out


# ---------------------------------------------------------------- pass 3b: tables

def _render_table(table: TableBlock, rows: list[tuple[str, ...]]) -> str:
    lines = []
    if table.caption:
        lines.append(table.caption)
    lines.append("| " + " | ".join(table.header) + " |")
    lines.append("|" + "---|" * len(table.header))
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    lines.append("")
    lines.append("Rows:")
    for r in rows:
        lines.append("- " + "; ".join(f"{h}: {v}" for h, v in zip(table.header, r)) + ".")
    return "\n".join(lines)


def table_chunks(table: TableBlock, rows_per_chunk: int, max_chars: int) -> list[tuple[str, int, int]]:
    """Group rows (at most rows_per_chunk, and within max_chars when possible).
    Each chunk repeats the caption and header row, and lists every row as a
    sentence ("Shift: Early; Login: 6:00 AM; ...") because the embedding model
    understands sentences better than pipe-separated cells.

    Returns (chunk_body, row_start, row_end) with 1-based inclusive row numbers."""
    out = []
    i = 0
    rows = list(table.rows)
    while i < len(rows):
        group = [rows[i]]
        j = i + 1
        while j < len(rows) and len(group) < rows_per_chunk:
            if len(_render_table(table, group + [rows[j]])) > max_chars:
                break
            group.append(rows[j])
            j += 1
        out.append((_render_table(table, group), i + 1, j))
        i = j
    return out


# ---------------------------------------------------------------- orchestration

def _breadcrumb(section: Section) -> str:
    if section.parent_title == section.title:
        return f"{section.section_id} {section.title}"
    return f"{section.parent_title} > {section.section_id} {section.title}"


def chunk_document(
    markdown: str,
    *,
    expected_company: str,
    expected_version: str,
    embed_fn: EmbedFn,
    breakpoint_percentile: float = 85.0,
    min_chars: int = 150,
    max_chars: int = 900,
    rows_per_chunk: int = 4,
) -> tuple[DocHeader, list[Chunk]]:
    """Parse, validate and chunk one policy document. Pure apart from embed_fn:
    nothing is written anywhere. Raises IngestValidationError on any problem."""
    header = parse_header(markdown)
    if header.company_id != expected_company:
        raise IngestValidationError(
            f"document is labelled company {header.company_id!r}, but is being ingested for {expected_company!r}"
        )
    if header.version != expected_version:
        raise IngestValidationError(
            f"document is labelled version {header.version!r}, but is being ingested as {expected_version!r}"
        )

    sections = split_sections(markdown)
    section_blocks = [(s, split_blocks(s.body, caption_max_chars=min_chars)) for s in sections]

    # Embed every prose sentence of the whole document in ONE batch, then pick the
    # breakpoint threshold as a percentile of all distances in this document, so
    # it adapts to the document's writing style.
    sentence_lists: dict[tuple[int, int], list[str]] = {}
    for si, (_, blocks) in enumerate(section_blocks):
        for bi, block in enumerate(blocks):
            if isinstance(block, TextBlock):
                sentence_lists[(si, bi)] = split_sentences(block.text)
    all_sentences = [s for sents in sentence_lists.values() for s in sents]
    vectors = embed_fn(all_sentences) if all_sentences else np.zeros((0, 0))

    distance_lists: dict[tuple[int, int], list[float]] = {}
    offset = 0
    for key, sents in sentence_lists.items():
        distance_lists[key] = consecutive_distances(vectors[offset:offset + len(sents)])
        offset += len(sents)
    all_distances = [d for ds in distance_lists.values() for d in ds]
    threshold = float(np.percentile(all_distances, breakpoint_percentile)) if all_distances else float("inf")

    chunks: list[Chunk] = []
    prefix = f"{header.company_id}__{header.version}"
    for si, (section, blocks) in enumerate(section_blocks):
        crumb = _breadcrumb(section)
        text_part = 0
        table_index = 0
        for bi, block in enumerate(blocks):
            base_meta = {
                "company_id": header.company_id,
                "version": header.version,
                "doc_name": header.doc_name,
                "effective_date": header.effective_date,
                "section_id": section.section_id,
                "section_title": section.title,
                "parent_title": section.parent_title,
            }
            if isinstance(block, TextBlock):
                pieces = semantic_split(
                    sentence_lists[(si, bi)], distance_lists[(si, bi)], threshold, min_chars, max_chars
                )
                for body in pieces:
                    chunks.append(Chunk(
                        chunk_id=f"{prefix}__{section.section_id}__s{text_part}",
                        text=f"{crumb}\n{body}",
                        metadata={**base_meta, "content_type": "text", "table_id": "",
                                  "row_start": 0, "row_end": 0, "chunk_index": len(chunks)},
                    ))
                    text_part += 1
            else:
                table_id = f"t{table_index}"
                for body, r0, r1 in table_chunks(block, rows_per_chunk, max_chars):
                    chunks.append(Chunk(
                        chunk_id=f"{prefix}__{section.section_id}__{table_id}__r{r0}-{r1}",
                        text=f"{crumb}\n{body}",
                        metadata={**base_meta, "content_type": "table", "table_id": table_id,
                                  "row_start": r0, "row_end": r1, "chunk_index": len(chunks)},
                    ))
                table_index += 1

    _validate_chunks(chunks)
    return header, chunks


def _validate_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        raise IngestValidationError("document produced no chunks")
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise IngestValidationError("duplicate chunk ids produced")
    for c in chunks:
        body = c.text.split("\n", 1)[1] if "\n" in c.text else ""
        if not body.strip():
            raise IngestValidationError(f"chunk {c.chunk_id} is empty")
        if len(body) > HARD_MAX_CHUNK_CHARS:
            raise IngestValidationError(f"chunk {c.chunk_id} is {len(body)} chars (limit {HARD_MAX_CHUNK_CHARS})")
        for k, v in c.metadata.items():
            if not isinstance(v, (str, int, float, bool)):
                raise IngestValidationError(f"chunk {c.chunk_id} metadata {k!r} is not a scalar")
