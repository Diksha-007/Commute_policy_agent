from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # The key must never be exposed, only whether one is configured.
    assert set(body) == {"status", "llm_configured", "use_hyde"}


def test_hyde_is_on_by_default_and_the_call_cap_still_holds():
    from app.config import Settings
    s = Settings(_env_file=None)
    assert s.use_hyde is True                     # project choice; see app/config.py and docs/experiments.md
    assert s.max_llm_calls == 2                   # HyDE + answer is exactly the cap
    assert s.hyde_timeout_s + s.retrieval_timeout_s + s.answer_timeout_s > s.total_budget_s    # the shared deadline, not these, bounds a question


def test_startup_loads_the_embedder_and_opens_each_live_collection(seeded_dir, monkeypatch):
    """The first question after a restart must not pay for loading the model (it could exceed the retrieval limit)."""
    import app.main as main_module
    from tests.conftest import make_client, make_services
    from app.llm import FakeLLM

    calls = {"query_embeds": 0}
    real = main_module.embed_query
    monkeypatch.setattr(main_module, "embed_query", lambda texts: calls.__setitem__("query_embeds", calls["query_embeds"] + 1) or real(texts))
    services = make_services(seeded_dir, FakeLLM())
    searched = []
    real_query = services.store.query
    monkeypatch.setattr(services.store, "query", lambda name, text, n, **kw: searched.append(name) or real_query(name, text, n, **kw))
    with make_client(services):
        pass
    assert calls["query_embeds"] == 1
    assert sorted(searched) == ["acme__v1", "globex__v1", "initech__v1"]


def test_a_failing_warm_up_does_not_stop_the_app(seeded_dir, monkeypatch):
    import app.main as main_module
    from tests.conftest import make_client, make_services
    from app.llm import FakeLLM

    monkeypatch.setattr(main_module, "embed_query", lambda texts: (_ for _ in ()).throw(RuntimeError("model missing")))
    with make_client(make_services(seeded_dir, FakeLLM())) as client:
        assert client.get("/health").status_code == 200
