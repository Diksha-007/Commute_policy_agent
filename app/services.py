"""Builds the long-lived objects the app needs, in one place.

Tests build their own Services pointing at temporary directories and pass them
to create_app(), so they never touch ./data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.answer import AnswerService
from app.config import Settings
from app.embeddings import embed, get_embedding_function
from app.ingest import Ingestor, IngestResult
from app.llm import LLMClient, build_llm
from app.registry import Registry
from app.store import PolicyStore

log = logging.getLogger(__name__)

# Seeded on first start. acme v2 is deliberately NOT here: it is uploaded live
# in the demo to show the version switch.
DEFAULT_SEED = [("acme", "v1"), ("globex", "v1"), ("initech", "v1")]


@dataclass
class Services:
    settings: Settings
    registry: Registry
    store: PolicyStore
    ingestor: Ingestor
    llm: LLMClient
    answerer: AnswerService


def build_services(settings: Settings, llm: Optional[LLMClient] = None) -> Services:
    """`llm` lets tests and the offline eval inject a FakeLLM."""
    registry = Registry(settings.registry_path)
    store = PolicyStore(settings.chroma_dir, get_embedding_function())
    llm = llm or build_llm(settings)
    ingestor = Ingestor(
        registry, store, embed,
        gc_delay_s=settings.gc_grace_seconds,
        breakpoint_percentile=settings.semantic_breakpoint_percentile,
        min_chars=settings.min_chunk_chars,
        max_chars=settings.max_chunk_chars,
        rows_per_chunk=settings.table_rows_per_chunk,
    )
    answerer = AnswerService(store, registry, llm, settings)
    return Services(settings, registry, store, ingestor, llm, answerer)


def seed_defaults(services: Services) -> list[IngestResult]:
    """Ingest the default policies for companies that have no active version yet."""
    results = []
    for company, version in DEFAULT_SEED:
        if services.registry.get_active_version(company) is None:
            path = services.settings.policies_dir / f"{company}_{version}.md"
            results.append(services.ingestor.ingest(company, version, path.read_text(encoding="utf-8")))
            log.info("seeded %s %s", company, version)
    return results
