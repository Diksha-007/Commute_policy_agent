"""MMR re-ranking: the pure function, and its use inside retrieve()."""
import numpy as np
import pytest

from app.llm import FakeLLM
from app.retrieval import IsolationViolation, mmr_select, retrieve
from tests.conftest import cite_first, make_services, principal


def unit(rows):
    arr = np.asarray(rows, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


# A is the most relevant; B is an almost-exact copy of A; C is different and a bit less relevant.
EMB = unit([[1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.0, 1.0, 0.0]])
REL = [0.80, 0.75, 0.60]


def test_a_near_duplicate_loses_to_a_different_chunk():
    # plain top-2 would be [A, B]; MMR prefers C because B only repeats A.
    assert mmr_select(REL, EMB, k=2, lam=1.0) == [0, 1]
    assert mmr_select(REL, EMB, k=2, lam=0.7) == [0, 2]


# ------------------------------------------------------------- inside retrieve()

@pytest.fixture
def services(seeded_dir):
    return make_services(seeded_dir, FakeLLM(responder=cite_first()), min_similarity=0.0)


def test_isolation_check_runs_over_the_whole_pool_before_mmr(services, monkeypatch):
    """A foreign chunk deep in the pool (rank 9, which MMR would never pick) must still be caught."""
    real_query = services.store.query

    def leaky(name, text, n, **kw):
        hits = real_query(name, text, n, **kw)
        hits[-1] = {**hits[-1], "metadata": {**hits[-1]["metadata"], "company_id": "acme"}}
        return hits

    monkeypatch.setattr(services.store, "query", leaky)
    with pytest.raises(IsolationViolation):
        retrieve(services.store, services.registry, principal("globex"), "escort", use_mmr=True, candidates=10)


