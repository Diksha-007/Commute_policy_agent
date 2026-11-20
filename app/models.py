"""API request/response schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    # Only the question. The company is never read from the body; any other field is ignored.
    question: str

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = " ".join(v.split())          # collapse whitespace and control characters
        if len(v) < MIN_QUESTION_CHARS:
            raise ValueError("please type a question")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"please keep the question under {MAX_QUESTION_CHARS} characters")
        return v


class Citation(BaseModel):
    doc_name: str
    version: str
    effective_date: str
    section_id: str
    section_title: str
    content_type: str
    excerpt: str
    rows: Optional[str] = None


class AskResponse(BaseModel):
    status: str                 # answered | not_covered | cannot_answer | try_again
    answer: str
    citations: list[Citation]
    request_id: str
    llm_calls: int
    latency_ms: int
