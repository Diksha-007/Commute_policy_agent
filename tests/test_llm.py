"""LLM layer: budgets, deadline, Groq error mapping (mocked HTTP), fake LLM."""
import json
import time

import httpx
import pytest

from app.llm import (
    CallBudget,
    CallBudgetExceeded,
    Deadline,
    FakeLLM,
    GroqLLM,
    LLMBadResponse,
    LLMTimeout,
    LLMUnavailable,
    Slow,
)

MSGS = [{"role": "user", "content": "hi"}]


def call(llm, timeout=1.0, json_mode=False):
    return llm.complete(model="m", messages=MSGS, timeout=timeout, json_mode=json_mode)


def groq_with(handler, key="test-key"):
    """GroqLLM whose HTTP layer is a local function: no network."""
    return GroqLLM(key, "https://example.invalid/v1", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def ok_body(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# ------------------------------------------------------------- budgets

def test_call_budget_allows_two_calls_and_blocks_the_third_before_calling():
    fake = FakeLLM(script=["a", "b", "c"])
    budget = CallBudget(max_calls=2)
    assert budget.complete(fake, model="m", messages=MSGS, timeout=1) == "a"
    assert budget.complete(fake, model="m", messages=MSGS, timeout=1) == "b"
    with pytest.raises(CallBudgetExceeded):
        budget.complete(fake, model="m", messages=MSGS, timeout=1)
    assert len(fake.calls) == 2          # the 3rd never reached the LLM


def test_a_failed_call_still_counts_against_the_budget():
    fake = FakeLLM(script=[LLMTimeout("slow"), "b"])
    budget = CallBudget(max_calls=1)
    with pytest.raises(LLMTimeout):
        budget.complete(fake, model="m", messages=MSGS, timeout=1)
    with pytest.raises(CallBudgetExceeded):
        budget.complete(fake, model="m", messages=MSGS, timeout=1)


def test_deadline_caps_each_step_by_time_left():
    now = [100.0]
    deadline = Deadline(2.7, clock=lambda: now[0])
    assert deadline.timeout_for(1.0) == 1.0
    now[0] += 2.0
    assert deadline.timeout_for(1.8) == pytest.approx(0.7)
    now[0] += 5.0
    assert deadline.remaining() == 0.0 and deadline.timeout_for(1.0) == 0.0


# ------------------------------------------------------------- Groq: success

def test_groq_returns_message_content_and_sends_json_mode():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=ok_body('{"status": "answered"}'))

    out = call(groq_with(handler), json_mode=True)
    assert out == '{"status": "answered"}'
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["temperature"] == 0
    assert seen["auth"] == "Bearer test-key"


# ------------------------------------------------------------- Groq: failures

@pytest.mark.parametrize("code", [429, 500, 503])
def test_groq_quota_and_server_errors_are_unavailable(code):
    with pytest.raises(LLMUnavailable):
        call(groq_with(lambda r: httpx.Response(code, json={"error": {"message": "x"}})))


def test_groq_timeout_is_enforced_on_wall_clock():
    def slow_handler(request):
        time.sleep(1.5)
        return httpx.Response(200, json=ok_body("late"))

    start = time.monotonic()
    with pytest.raises(LLMTimeout):
        call(groq_with(slow_handler), timeout=0.2)
    assert time.monotonic() - start < 0.6     # we stopped waiting at ~0.2s, not 1.5s


def test_groq_without_key_fails_fast_without_network():
    def handler(request):  # pragma: no cover - must not be reached
        raise AssertionError("network was called")

    with pytest.raises(LLMUnavailable, match="no LLM API key"):
        call(groq_with(handler, key=""))


# ------------------------------------------------------------- fake LLM

