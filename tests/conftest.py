"""Shared fixtures: every test gets throwaway storage, so tests never touch ./data.

- registry / store / ingestor: empty, per test (for ingestion tests)
- seeded_dir: acme v1, globex v1, initech v1 ingested ONCE per session (read-only use)
- make_services(): Services over seeded_dir with a FakeLLM and setting overrides
"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import Principal, issue_token
from app.config import Settings
from app import embeddings
from app.embeddings import embed, get_embedding_function
from app.ingest import Ingestor
from app.main import create_app
from app.registry import Registry
from app.services import build_services
from app.store import PolicyStore
from tests.fake_embedder import HashingEmbedder

# Every test uses the small deterministic embedder, not the embedding model, unless it asks for `real_embedder`.
embeddings.set_embedder(HashingEmbedder())


@pytest.fixture
def real_embedder():
    """Switch to the real bge-base model for one test (those tests are marked `real_model`)."""
    embeddings.set_embedder(None)
    yield
    embeddings.set_embedder(HashingEmbedder())


POLICIES = Path(__file__).resolve().parent.parent / "policies"


def policy_text(company: str, version: str) -> str:
    return (POLICIES / f"{company}_{version}.md").read_text(encoding="utf-8")


# ------------------------------------------------------------- empty storage

@pytest.fixture
def registry(tmp_path) -> Registry:
    return Registry(tmp_path / "registry.db")


@pytest.fixture
def store(tmp_path) -> PolicyStore:
    return PolicyStore(tmp_path / "chroma", get_embedding_function())


@pytest.fixture
def ingestor(registry, store) -> Ingestor:
    # gc_delay_s=0: drop old versions immediately so tests are deterministic.
    return Ingestor(registry, store, embed, gc_delay_s=0)


# ------------------------------------------------------------- seeded storage

@pytest.fixture(scope="session")
def seeded_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("seeded")
    settings = Settings(data_dir=path, _env_file=None)
    ing = Ingestor(Registry(settings.registry_path), PolicyStore(settings.chroma_dir, get_embedding_function()),
                   embed, gc_delay_s=0)
    for company in ("acme", "globex", "initech"):
        ing.ingest(company, "v1", policy_text(company, "v1"))
    return path


def make_services(data_dir: Path, llm, **overrides):
    """Services over existing storage, with an injected LLM and settings overrides."""
    overrides.setdefault("use_hyde", False)        # HyDE is on by default in the app; tests script one answer call
    settings = Settings(data_dir=data_dir, _env_file=None, **overrides)
    services = build_services(settings, llm=llm)
    services.ingestor.gc_delay_s = 0
    return services


def make_client(services) -> TestClient:
    # raise_server_exceptions=False: let the app's own 500 handler respond, like in production.
    return TestClient(create_app(services, seed=False), raise_server_exceptions=False)


def principal(company: str = "acme", **profile) -> Principal:
    return Principal(employee_id="emp-1", company_id=company, role="employee", **profile)


def bearer_for(services, company: str = "acme", **claims) -> dict:
    token = issue_token(services.settings, employee_id="emp-1", company_id=company, **claims)
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------- fake LLM responders

def excerpt_ids(request: dict) -> list[str]:
    """The excerpt ids the answer prompt offered to the model."""
    return re.findall(r'<excerpt id="([^"]+)"', request["messages"][-1]["content"])


def cite_first(answer: str = "Grounded answer."):
    """A well-behaved model: answers and cites the top retrieved excerpt."""
    return lambda req: json.dumps({"status": "answered", "answer": answer, "cited_chunk_ids": excerpt_ids(req)[:1]})


def cite_section(section: str, answer: str = "Grounded answer."):
    """A well-behaved model that cites the excerpt of one section among those it was shown. (Which chunk ranks first
    depends on the embedder, so tests pick by section instead of relying on the ranking.)"""
    def respond(req):
        ids = [i for i in excerpt_ids(req) if f"__{section}__" in i]
        assert ids, f"section {section} was not among the excerpts shown: {excerpt_ids(req)}"
        return json.dumps({"status": "answered", "answer": answer, "cited_chunk_ids": ids[:1]})
    return respond


def reply(status: str = "answered", answer: str = "text", cited=None) -> str:
    return json.dumps({"status": status, "answer": answer, "cited_chunk_ids": cited or []})
