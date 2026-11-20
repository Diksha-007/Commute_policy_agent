"""Opt-in checks of the REAL embedding model (bge-base). Run with:   pytest -m real_model

The default suite does not use the model at all (see tests/fake_embedder.py), so it needs no model files and passes the
same on any machine. These few tests make sure the real model is wired in correctly: shape, normalisation, the query
instruction, and that a question really finds its section through the whole stack."""
import numpy as np
import pytest

from app.chunking import consecutive_distances, semantic_split
from app.embeddings import DIMENSIONS, embed, embed_query
from app.llm import FakeLLM
from app.retrieval import retrieve
from tests.conftest import cite_first, make_services, policy_text, principal

pytestmark = [pytest.mark.real_model, pytest.mark.usefixtures("real_embedder")]


def test_vectors_have_the_right_shape_and_are_unit_length():
    vecs = embed(["A security escort travels on night drops.", "Cost cap for an airport trip."])
    assert vecs.shape == (2, DIMENSIONS)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_semantic_split_sees_the_topic_shift():
    """The topic shift (escort -> no-show fines) is one of the two largest gaps. bge-base's distances are compressed
    (the shift is only just behind a gap inside the escort topic), which is stated in the README."""
    sentences = [
        "A trained security guard travels in the cab on night drops for women employees.",
        "The guard sits in the front seat and stays until the last woman employee is dropped.",
        "The transport desk checks the guard's identity card before the trip starts.",
        "An employee who misses a pickup twice in a month is fined 200 rupees.",
        "The fine is recovered from the employee's salary through payroll.",
        "A third missed pickup suspends the employee's transport for one week.",
    ]
    dists = consecutive_distances(embed(sentences))
    assert 2 in set(np.argsort(dists)[-2:].tolist())
    assert len(semantic_split(sentences, dists, threshold=max(dists) - 1e-6, min_chars=50, max_chars=900)) == 2


def test_the_whole_stack_finds_the_escort_section_with_the_real_model(tmp_path):
    services = make_services(tmp_path, FakeLLM(responder=cite_first()), min_similarity=0.0)
    services.ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    result = retrieve(services.store, services.registry, principal("acme"),
                      "Will a guard travel with me if my cab leaves the office at 9 at night?", top_k=3)
    assert "3.2" in [c.metadata["section_id"] for c in result.chunks]
