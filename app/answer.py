"""Question -> grounded, cited answer (or an honest refusal), within budget.

    1. [LLM call 1, optional] HyDE passage. Any failure: carry on without it.
    2. Retrieve (own company + active version only), under a timeout.
    3. Gate: best similarity below threshold -> "not covered", no LLM call.
    4. [LLM call 2] Answer as JSON, with whatever time is left.
    5. Validate the JSON in code: shape, status, citations subset of retrieved ids.
    6. Build citations from chunk METADATA (never from model text).

Statuses:
    answered       grounded answer + 1-3 citations
    not_covered    policy is silent (gate or model)              -> desk message
    cannot_answer  model output unusable / isolation check failed -> desk message
    try_again      LLM or retrieval slow/unavailable              -> HTTP 503
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app import messages
from app.auth import Principal
from app.config import Settings
from app.llm import CallBudget, Deadline, LLMBadResponse, LLMClient, LLMError
from app.prompts import build_answer_messages, build_hyde_messages
from app.registry import Registry
from app.retrieval import IsolationViolation, NoActivePolicy, RetrievalResult, RetrievedChunk, retrieve
from app.store import PolicyStore

log = logging.getLogger(__name__)

MAX_ANSWER_CHARS = 1200
MAX_HYDE_CHARS = 800
EXCERPT_CHARS = 200

_RETRIEVAL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval")


class ModelOutputError(ValueError):
    """The model's JSON is unusable. The message says why (logged, not shown)."""


@dataclass
class AnswerResult:
    status: str
    answer: str
    citations: list[dict]
    llm_calls: int
    latency_ms: int
    request_id: str
    # Diagnostics for logs and the evaluation; not returned by the API.
    reason: str = ""
    retrieved: list[dict] = field(default_factory=list)
    hyde_used: bool = False
    cited_chunk_ids: list[str] = field(default_factory=list)


# ------------------------------------------------------------ validation (pure)

def _strip_code_fence(raw: str) -> str:
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw, flags=re.S)
    return m.group(1) if m else raw


def parse_model_output(raw: str, retrieved_ids: list[str], max_citations: int) -> tuple[str, str, list[str]]:
    """Return (status, answer, cited_ids) or raise ModelOutputError."""
    try:
        data = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError) as e:
        raise ModelOutputError("not valid JSON") from e
    if not isinstance(data, dict):
        raise ModelOutputError("JSON is not an object")

    status = data.get("status")
    if status not in ("answered", "not_covered"):
        raise ModelOutputError(f"unknown status {status!r}")
    if status == "not_covered":
        return "not_covered", "", []

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ModelOutputError("answered without answer text")
    if len(answer) > MAX_ANSWER_CHARS:
        raise ModelOutputError("answer too long")

    cited = data.get("cited_chunk_ids")
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        raise ModelOutputError("cited_chunk_ids is not a list of strings")
    cited = list(dict.fromkeys(cited))            # dedupe, keep order
    if not cited:
        raise ModelOutputError("answered without citations")
    unknown = [c for c in cited if c not in retrieved_ids]
    if unknown:
        raise ModelOutputError(f"cited ids that were not retrieved: {unknown}")
    if len(cited) > max_citations:
        # The model saw more excerpts than it may cite. Keep the best-ranked ones it cited.
        cited = sorted(cited, key=retrieved_ids.index)[:max_citations]
    return "answered", answer.strip(), cited


def build_citation(chunk: RetrievedChunk) -> dict:
    m = chunk.metadata
    citation = {
        "doc_name": m["doc_name"],
        "version": m["version"],
        "effective_date": m["effective_date"],
        "section_id": m["section_id"],
        "section_title": m["section_title"],
        "content_type": m["content_type"],
        "excerpt": chunk.body[:EXCERPT_CHARS] + ("…" if len(chunk.body) > EXCERPT_CHARS else ""),
    }
    if m["content_type"] == "table":
        citation["rows"] = f"{m['row_start']}-{m['row_end']}"
    return citation


# ------------------------------------------------------------ service

class AnswerService:
    def __init__(self, store: PolicyStore, registry: Registry, llm: LLMClient, settings: Settings):
        self.store = store
        self.registry = registry
        self.llm = llm
        self.settings = settings

    def answer(self, principal: Principal, question: str, *, use_hyde: Optional[bool] = None,
               use_mmr: Optional[bool] = None) -> AnswerResult:
        s = self.settings
        start = time.monotonic()
        request_id = uuid.uuid4().hex[:12]
        deadline = Deadline(s.total_budget_s)
        budget = CallBudget(s.max_llm_calls)
        use_hyde = s.use_hyde if use_hyde is None else use_hyde
        use_mmr = s.use_mmr if use_mmr is None else use_mmr
        retrieval: Optional[RetrievalResult] = None
        retrieval_ms: Optional[int] = None
        hyde_text: Optional[str] = None

        def done(status: str, answer: str, reason: str, citations=None, cited_ids=None) -> AnswerResult:
            result = AnswerResult(
                status=status, answer=answer, citations=citations or [], llm_calls=budget.used,
                latency_ms=round(1000 * (time.monotonic() - start)), request_id=request_id, reason=reason,
                retrieved=[
                    {"chunk_id": c.chunk_id, "section_id": c.metadata["section_id"],
                     "similarity": c.similarity, "hyde_similarity": c.hyde_similarity}
                    for c in (retrieval.chunks if retrieval else [])
                ],
                hyde_used=hyde_text is not None, cited_chunk_ids=list(cited_ids or []),
            )
            log.info("ask request_id=%s company=%s status=%s reason=%s llm_calls=%d latency_ms=%d retrieval_ms=%s retrieved=%s",
                     request_id, principal.company_id, status, reason, result.llm_calls, result.latency_ms, retrieval_ms,
                     [r["chunk_id"] for r in result.retrieved])
            return result

        # 1. HyDE (optional). Its output only steers retrieval inside this company's own collection.
        if use_hyde:
            try:
                text = budget.complete(
                    self.llm, model=s.hyde_model, messages=build_hyde_messages(question),
                    timeout=deadline.timeout_for(s.hyde_timeout_s), max_tokens=150,
                    reasoning_effort=s.hyde_reasoning_effort,
                ).strip()
                hyde_text = text[:MAX_HYDE_CHARS] or None
            except LLMError as e:
                log.warning("request_id=%s hyde_fallback: %s", request_id, e)   # never retried

        # 2. Retrieval, under its own timeout.
        retrieval_started = time.monotonic()
        future = _RETRIEVAL_POOL.submit(
            retrieve, self.store, self.registry, principal, question,
            hyde_text=hyde_text, top_k=max(s.retrieve_k, s.top_k),
            candidates=s.mmr_pool if use_mmr else s.candidates_per_query, rrf_k=s.rrf_k,
            use_mmr=use_mmr, mmr_lambda=s.mmr_lambda,
        )
        try:
            retrieval = future.result(timeout=deadline.timeout_for(s.retrieval_timeout_s))
            retrieval_ms = round(1000 * (time.monotonic() - retrieval_started))
        except concurrent.futures.TimeoutError:
            return done("try_again", messages.TRY_AGAIN, "retrieval_timeout")
        except IsolationViolation as e:
            log.error("request_id=%s ISOLATION VIOLATION: %s", request_id, e)
            return done("cannot_answer", messages.CANNOT_ANSWER, "isolation_violation")
        except NoActivePolicy:
            return done("cannot_answer", messages.CANNOT_ANSWER, "no_active_policy")
        except Exception as e:
            log.error("request_id=%s retrieval_error: %s", request_id, e)
            return done("try_again", messages.TRY_AGAIN, "retrieval_error")

        # 3. Not-covered gate: no relevant evidence -> honest answer, no LLM call.
        if not retrieval.chunks or retrieval.best_similarity < s.min_similarity:
            return done("not_covered", messages.NOT_COVERED, "below_similarity_threshold")

        # 4. Answer call with whatever time is left (never start a call we cannot finish).
        if deadline.remaining() < s.min_answer_budget_s:
            return done("try_again", messages.TRY_AGAIN, "time_budget_exhausted")
        try:
            raw = budget.complete(
                self.llm, model=s.answer_model,
                messages=build_answer_messages(principal, question, retrieval.chunks),
                timeout=deadline.timeout_for(s.answer_timeout_s), json_mode=True, max_tokens=400,
                reasoning_effort=s.answer_reasoning_effort,
            )
        except LLMBadResponse as e:
            log.warning("request_id=%s bad_llm_response: %s", request_id, e)
            return done("cannot_answer", messages.CANNOT_ANSWER, "bad_llm_response")
        except LLMError as e:
            log.warning("request_id=%s llm_failure: %s", request_id, e)
            return done("try_again", messages.TRY_AGAIN, f"llm_{type(e).__name__}")

        # 5. Validate the model's output in code.
        ids = [c.chunk_id for c in retrieval.chunks]
        try:
            status, answer, cited = parse_model_output(raw, ids, s.top_k)
        except ModelOutputError as e:
            log.warning("request_id=%s invalid_model_output: %s | raw=%r", request_id, e, raw[:300])
            return done("cannot_answer", messages.CANNOT_ANSWER, f"invalid_model_output: {e}")
        if status == "not_covered":
            return done("not_covered", messages.NOT_COVERED, "model_not_covered")

        # 6. Citations come from our metadata, in the order the model cited them.
        by_id = {c.chunk_id: c for c in retrieval.chunks}
        return done("answered", answer, "ok", [build_citation(by_id[c]) for c in cited], cited_ids=cited)
