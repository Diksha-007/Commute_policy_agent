"""LLM access, behind a tiny interface so tests and the offline eval use a fake.

    LLMClient    protocol: complete(model, messages, timeout, json_mode, max_tokens) -> str
    GroqLLM      real client (Groq's OpenAI-compatible HTTP API) with a STRICT wall-clock timeout
    FakeLLM      scripted stand-in: returns canned text or raises chosen failures; records calls
    CallBudget   per-question counter; the 3rd call raises BEFORE any network I/O
    Deadline     per-question time budget; each step gets min(step timeout, time left)

Error taxonomy (what the answer path does with each):
    LLMTimeout       -> "try again"      (slow provider)
    LLMUnavailable   -> "try again"      (429 quota, 5xx, network, missing/invalid key)
    LLMBadResponse   -> "cannot answer"  (provider says the model produced invalid JSON,
                                          or the response body is malformed)
    CallBudgetExceeded is a programming error, not a user-facing condition.
"""
from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union

import httpx

log = logging.getLogger(__name__)


# ------------------------------------------------------------ errors

class LLMError(Exception):
    """Base class for LLM failures the answer path handles."""


class LLMTimeout(LLMError):
    pass


class LLMUnavailable(LLMError):
    pass


class LLMBadResponse(LLMError):
    pass


class CallBudgetExceeded(RuntimeError):
    """More LLM calls than allowed for one question. A bug, never sent to the provider."""


# ------------------------------------------------------------ budgets

class Deadline:
    """Time budget for one question, measured with a monotonic clock."""

    def __init__(self, total_s: float, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._end = clock() + total_s

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def timeout_for(self, step_s: float) -> float:
        """A step may use its own timeout, but never more than what is left."""
        return min(step_s, self.remaining())


class CallBudget:
    """Counts LLM calls for one question. A call counts when it is ATTEMPTED,
    even if it then fails, so retries can never sneak past the limit."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0

    def complete(self, llm: "LLMClient", **kwargs: Any) -> str:
        if self.used >= self.max_calls:
            raise CallBudgetExceeded(f"LLM call budget of {self.max_calls} per question exhausted")
        self.used += 1
        return llm.complete(**kwargs)


# ------------------------------------------------------------ interface

class LLMClient(Protocol):
    def complete(
        self, *, model: str, messages: list[dict], timeout: float,
        json_mode: bool = False, max_tokens: int = 300,
        reasoning_effort: Optional[str] = None,
    ) -> str: ...


# ------------------------------------------------------------ Groq

# Requests run on worker threads so the caller can stop waiting at exactly
# `timeout` seconds. httpx's own timeouts apply per phase (connect, read...) and
# could add up to more than the budget.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")


def _provider_message(response: httpx.Response, limit: int = 300) -> str:
    """The provider's error text, shortened, with the organisation id removed."""
    try:
        text = str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        text = response.text
    text = re.sub(r"organization `[^`]*`", "organization `...`", text)
    return " ".join(text.split())[:limit]


class GroqLLM:
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1",
                 http_client: Optional[httpx.Client] = None):
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._http = http_client or httpx.Client()

    def complete(self, *, model: str, messages: list[dict], timeout: float,
                 json_mode: bool = False, max_tokens: int = 300,
                 reasoning_effort: Optional[str] = None) -> str:
        if not self._api_key:
            raise LLMUnavailable("no LLM API key configured")
        if timeout <= 0:
            raise LLMTimeout("no time left in the budget for an LLM call")

        payload: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            # Reasoning models think before answering; keeping this low/none keeps latency down.
            payload["reasoning_effort"] = reasoning_effort

        future = _EXECUTOR.submit(self._post, payload, timeout)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise LLMTimeout(f"LLM did not answer within {timeout:.2f}s")
        return self._parse(response)

    def _post(self, payload: dict, timeout: float) -> httpx.Response:
        try:
            return self._http.post(
                self._url, json=payload, timeout=timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as e:
            raise LLMTimeout(f"LLM request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"LLM request failed: {type(e).__name__}") from e

    @staticmethod
    def _parse(response: httpx.Response) -> str:
        code = response.status_code
        if code == 400 and "json_validate_failed" in response.text:
            # Groq refuses to return output that is not valid JSON in JSON mode.
            raise LLMBadResponse("model output failed JSON validation")
        if code == 429:
            # Keep the provider's explanation (which limit, how long to wait) for the server log.
            # It never reaches the user: the answer path replaces it with a fixed "try again".
            raise LLMUnavailable(f"LLM rate limit or quota exceeded (429): {_provider_message(response)}")
        if code >= 500:
            raise LLMUnavailable(f"LLM provider error ({code})")
        if code >= 400:
            # 401/403 = bad key, 404 = unknown model, 400 = bad request: all configuration
            # problems the user cannot fix; tell them to try again later.
            raise LLMUnavailable(f"LLM request rejected ({code})")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMBadResponse("malformed LLM response body") from e
        if not isinstance(content, str) or not content.strip():
            # A reasoning model can spend all max_tokens thinking and return nothing.
            raise LLMBadResponse("LLM returned no answer text")
        return content


# ------------------------------------------------------------ fake

@dataclass
class Slow:
    """Fake step: take `seconds` to answer. If that exceeds the call's timeout,
    behave like a real timeout (wait `timeout`, then raise LLMTimeout)."""
    seconds: float
    then: Union[str, Exception] = '{"status": "not_covered", "answer": "", "cited_chunk_ids": []}'


Step = Union[str, Exception, Slow, Callable[[dict], str]]


@dataclass
class FakeLLM:
    """Scripted LLM for tests and the offline eval.

    Each call consumes the next step of `script`:
        str        -> returned as the model output
        Exception  -> raised (e.g. LLMTimeout(), LLMUnavailable("429"))
        Slow       -> simulated latency, honouring the timeout
        callable   -> called with the request dict, returns the output
    When the script is empty, `responder` (if given) handles every call.
    """
    script: list[Step] = field(default_factory=list)
    responder: Optional[Callable[[dict], str]] = None
    calls: list[dict] = field(default_factory=list)

    def complete(self, *, model: str, messages: list[dict], timeout: float,
                 json_mode: bool = False, max_tokens: int = 300,
                 reasoning_effort: Optional[str] = None) -> str:
        request = {"model": model, "messages": messages, "timeout": timeout,
                   "json_mode": json_mode, "max_tokens": max_tokens,
                   "reasoning_effort": reasoning_effort}
        self.calls.append(request)
        if self.script:
            step = self.script.pop(0)
        elif self.responder is not None:
            step = self.responder
        else:
            raise AssertionError("FakeLLM called more times than scripted")
        return self._run(step, request, timeout)

    def _run(self, step: Step, request: dict, timeout: float) -> str:
        if isinstance(step, Slow):
            if step.seconds > timeout:
                time.sleep(max(0.0, timeout))
                raise LLMTimeout(f"fake LLM exceeded {timeout:.2f}s")
            time.sleep(step.seconds)
            return self._run(step.then, request, timeout)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(request)
        return step


def build_llm(settings) -> LLMClient:
    return GroqLLM(settings.groq_key, settings.groq_base_url)
