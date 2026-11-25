from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import llm
from app.config import ProviderConfig

PROVIDER = ProviderConfig("fake", "https://fake.test/v1", "fake-model", "key")
CHUNKS = [{"text": "ADDI adds an immediate.", "section": "2.4"}, {"text": "Another excerpt."}]
NOW = datetime(2026, 11, 23, 10, 0, 0)


class FakeClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        message = SimpleNamespace(content=self.outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def rate_limit_error(headers):
    request = httpx.Request("POST", "https://fake.test/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError("limit", response=response, body=None)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    llm._limited_until.clear()
    monkeypatch.setattr(llm, "_now", lambda: NOW)
    yield
    llm._limited_until.clear()


def use_client(monkeypatch, outcome):
    client = FakeClient(outcome)
    monkeypatch.setattr(llm, "make_client", lambda provider: client)
    return client


def test_generate_sends_numbered_chunks_with_zero_temperature(monkeypatch):
    client = use_client(monkeypatch, " Answer [S1] ")
    result = llm.generate("What does ADDI do?", CHUNKS, PROVIDER)

    assert result.text == "Answer [S1]"
    assert result.model == "fake-model"
    call = client.calls[0]
    assert call["temperature"] == 0
    assert "[S1] (2.4)\nADDI adds" in call["messages"][1]["content"]
    assert "[S2]\nAnother" in call["messages"][1]["content"]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "120"}, "rate limited until 10:02"),
        (
            {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "2m59.5s"},
            "rate limited until 10:03",
        ),
        ({}, "rate limited until 10:01"),
    ],
)
def test_429_marks_provider_limited(monkeypatch, headers, expected):
    use_client(monkeypatch, rate_limit_error(headers))
    with pytest.raises(llm.ProviderRateLimitedError):
        llm.generate("q", CHUNKS, PROVIDER)
    assert llm.provider_status("fake") == expected


def test_limited_provider_is_not_called_until_expiry(monkeypatch):
    client = use_client(monkeypatch, "ok")
    llm._limited_until["fake"] = NOW + timedelta(seconds=30)

    with pytest.raises(llm.ProviderRateLimitedError):
        llm.generate("q", CHUNKS, PROVIDER)
    assert client.calls == []

    monkeypatch.setattr(llm, "_now", lambda: NOW + timedelta(seconds=31))
    assert llm.provider_status("fake") == llm.STATUS_AVAILABLE
    assert llm.generate("q", CHUNKS, PROVIDER).text == "ok"


def test_provider_error_becomes_llm_error(monkeypatch):
    request = httpx.Request("POST", "https://fake.test/v1/chat/completions")
    use_client(monkeypatch, openai.APITimeoutError(request=request))
    with pytest.raises(llm.LLMError):
        llm.generate("q", CHUNKS, PROVIDER)


def test_connection_error_reports_cause_without_url(monkeypatch):
    request = httpx.Request("POST", "https://private.test/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    error.__cause__ = httpx.ConnectError("failed for https://private.test")
    use_client(monkeypatch, error)

    with pytest.raises(llm.LLMError) as caught:
        llm.generate("q", CHUNKS, PROVIDER)

    assert "cause: ConnectError" in str(caught.value)
    assert "private.test" not in str(caught.value)


def test_empty_answer_is_an_error(monkeypatch):
    use_client(monkeypatch, "   ")
    with pytest.raises(llm.LLMError):
        llm.generate("q", CHUNKS, PROVIDER)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, llm.TIMEOUT_SECONDS),
        ("300", 300.0),
        ("abc", llm.TIMEOUT_SECONDS),
        ("0", llm.TIMEOUT_SECONDS),
    ],
)
def test_timeout_comes_from_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", value)

    assert llm.timeout_seconds() == expected
    assert llm.make_client(ProviderConfig("p", "https://fake.test/v1", "m")).timeout == expected
