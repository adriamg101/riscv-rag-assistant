from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app import llm, rag
from app.config import ProviderConfig
from app.embeddings import get_embedder
from app.llm import GenerationResult, LLMError
from app.main import app, get_chroma, get_mode, get_providers
from app.prompts import REFUSAL
from app.retrieve import FLAT

SECRET_URL = "https://private-endpoint.test/v1"
SECRET_KEY = "super-secret-key"
PROVIDER = ProviderConfig("fake", SECRET_URL, "fake-model", SECRET_KEY)


@pytest.fixture
def overrides(indexed, embedder):
    values = {"providers": {"fake": PROVIDER}}
    app.dependency_overrides[get_providers] = lambda: values["providers"]
    app.dependency_overrides[get_mode] = lambda: FLAT
    app.dependency_overrides[get_chroma] = lambda: indexed
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield values
    app.dependency_overrides.clear()


@pytest.fixture
def api(overrides):
    return TestClient(app)


def answer_with(monkeypatch, text):
    monkeypatch.setattr(
        rag, "generate", lambda question, chunks, provider: GenerationResult(text, "fake-model", 5)
    )


def test_ask_returns_answer_and_sources(api, monkeypatch):
    answer_with(monkeypatch, "Memory load [S1].")

    response = api.post("/ask", json={"question": "load"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Memory load [S1]."
    assert body["abstained"] is False
    assert body["model"] == "fake-model"
    assert body["sources"][0]["label"] == "S1"
    assert body["sources"][0]["source_url"] == "https://example.test/mini.html#gamma"


def test_ask_with_invented_citation_abstains(api, monkeypatch):
    answer_with(monkeypatch, "Invented [S99].")

    body = api.post("/ask", json={"question": "load"}).json()

    assert body["abstained"] is True
    assert body["answer"] == REFUSAL
    assert body["sources"] == []


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "   "}, {}, {"question": "x" * 2001}],
)
def test_invalid_question_is_422(api, payload):
    assert api.post("/ask", json=payload).status_code == 422


def test_unknown_model_is_422(api):
    response = api.post("/ask", json={"question": "load", "model": "nope"})
    assert response.status_code == 422


def test_llm_failure_is_502_with_clear_message(api, monkeypatch):
    def fail(question, chunks, provider):
        raise LLMError("fake: APITimeoutError")

    monkeypatch.setattr(rag, "generate", fail)

    response = api.post("/ask", json={"question": "load"})

    assert response.status_code == 502
    assert "LLM provider" in response.json()["detail"]
    assert SECRET_KEY not in response.text


def test_retrieval_only_mode(api, overrides):
    overrides["providers"] = {}

    body = api.post("/ask", json={"question": "load"}).json()

    assert body["answer"] is None
    assert body["abstained"] is False
    assert body["sources"]


def test_models_shows_status_without_secrets(api):
    llm._limited_until["fake"] = datetime.now() + timedelta(minutes=5)

    response = api.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_only"] is False
    assert body["providers"][0]["name"] == "fake"
    assert body["providers"][0]["status"].startswith("rate limited until")
    assert SECRET_URL not in response.text
    assert SECRET_KEY not in response.text


def test_health(api):
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["retrieval_mode"] == FLAT
    assert body["indexed_chunks"] == 4
