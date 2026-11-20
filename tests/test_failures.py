"""Failure handling through the HTTP API: slow/failing LLM, slow/failing retrieval,
budgets, bad input, unexpected errors. Users get "try again" or a desk message,
never a hang, a stack trace or half an answer."""
import time

import pytest

from app import messages
from app.llm import FakeLLM, LLMBadResponse, LLMTimeout, LLMUnavailable, Slow
from tests.conftest import bearer_for, cite_first, cite_section, make_client, make_services

Q = {"question": "Do women get a security escort on late night drops?"}


def api(seeded_dir, llm, **overrides):
    overrides.setdefault("min_similarity", 0.0)
    services = make_services(seeded_dir, llm, **overrides)
    client = make_client(services)
    client.__enter__()                     # run startup (recover, no seeding)
    return client, services


def ask(client, services, body=Q, company="acme"):
    return client.post("/ask", json=body, headers=bearer_for(services, company))


# ------------------------------------------------------------- happy path over HTTP

def test_ask_returns_answer_with_citations(seeded_dir):
    client, services = api(seeded_dir, FakeLLM(script=[cite_section("3.2", "Yes, after 8:00 PM.")]))
    resp = ask(client, services)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered" and body["llm_calls"] == 1
    assert body["citations"][0]["section_id"] == "3.2"
    assert set(body) == {"status", "answer", "citations", "request_id", "llm_calls", "latency_ms"}


# ------------------------------------------------------------- LLM failures

@pytest.mark.parametrize("failure", [LLMTimeout("slow"), LLMUnavailable("429 quota exceeded"),
                                     LLMUnavailable("503 provider down")])
def test_llm_failure_is_503_try_again(seeded_dir, failure):
    client, services = api(seeded_dir, FakeLLM(script=[failure]))
    resp = ask(client, services)
    assert resp.status_code == 503 and resp.headers["retry-after"] == "2"
    assert resp.json()["status"] == "try_again" and resp.json()["answer"] == messages.TRY_AGAIN
    assert resp.json()["citations"] == []


def test_slow_llm_is_cut_off_within_the_time_budget(seeded_dir):
    client, services = api(seeded_dir, FakeLLM(script=[Slow(10.0)]), answer_timeout_s=0.3)
    start = time.monotonic()
    resp = ask(client, services)
    assert resp.status_code == 503 and resp.json()["status"] == "try_again"
    assert time.monotonic() - start < 1.5


def test_invalid_json_from_provider_is_cannot_answer_not_retry(seeded_dir):
    client, services = api(seeded_dir, FakeLLM(script=[LLMBadResponse("json_validate_failed")]))
    resp = ask(client, services)
    assert resp.status_code == 200 and resp.json()["status"] == "cannot_answer"
    assert "transport desk" in resp.json()["answer"]


# ------------------------------------------------------------- HyDE failures and the call budget

@pytest.mark.parametrize("hyde_failure", [LLMTimeout("slow"), LLMUnavailable("429")])
def test_hyde_failure_falls_back_and_still_answers(seeded_dir, hyde_failure):
    client, services = api(seeded_dir, FakeLLM(script=[hyde_failure, cite_first()]), use_hyde=True)
    resp = ask(client, services)
    assert resp.status_code == 200 and resp.json()["status"] == "answered"
    assert resp.json()["llm_calls"] == 2


def test_no_time_left_skips_the_answer_call(seeded_dir):
    llm = FakeLLM(script=[Slow(0.5, then="passage")])      # HyDE eats the whole budget
    client, services = api(seeded_dir, llm, use_hyde=True, total_budget_s=0.6, hyde_timeout_s=0.6)
    resp = ask(client, services)
    assert resp.status_code == 503 and resp.json()["status"] == "try_again"
    assert len(llm.calls) == 1                             # never started a call it could not finish


# ------------------------------------------------------------- retrieval failures

def test_retrieval_error_is_try_again(seeded_dir, monkeypatch):
    client, services = api(seeded_dir, FakeLLM())
    monkeypatch.setattr(services.store, "query", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    resp = ask(client, services)
    assert resp.status_code == 503 and resp.json()["status"] == "try_again"


def test_slow_retrieval_is_try_again_without_hanging(seeded_dir, monkeypatch):
    client, services = api(seeded_dir, FakeLLM(), retrieval_timeout_s=0.2)
    monkeypatch.setattr(services.store, "query", lambda *a, **k: time.sleep(2) or [])
    start = time.monotonic()
    resp = ask(client, services)
    assert resp.status_code == 503 and time.monotonic() - start < 1.0


# ------------------------------------------------------------- bad input and unexpected errors

def test_unexpected_error_is_a_plain_500(seeded_dir, monkeypatch):
    client, services = api(seeded_dir, FakeLLM())
    monkeypatch.setattr(services.answerer, "answer", lambda *a, **k: 1 / 0)
    resp = ask(client, services)
    assert resp.status_code == 500
    assert resp.json()["answer"] == messages.INTERNAL_ERROR
    assert "ZeroDivisionError" not in resp.text and "Traceback" not in resp.text
