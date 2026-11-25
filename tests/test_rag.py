from datetime import datetime

import pytest

from app import rag
from app.config import ProviderConfig
from app.llm import GenerationResult, LLMError, ProviderRateLimitedError
from app.prompts import REFUSAL
from app.rag import UnknownProviderError, answer_from_chunks, answer_question
from app.retrieve import FLAT, RetrievedChunk

FIRST = ProviderConfig("first", "https://first.test/v1", "first-model", "key")
SECOND = ProviderConfig("second", "https://second.test/v1", "second-model", "key")
PROVIDERS = {"first": FIRST, "second": SECOND}


def chunk(number, score=0.9):
    return RetrievedChunk(
        id=f"id{number}",
        text=f"Text {number}",
        section=f"Section {number}",
        section_id=f"p#s{number}",
        path=f"Chapter > Section {number}",
        source_url=f"https://example.test/p.html#s{number}",
        chunk_index=0,
        score=score,
    )


CHUNKS = [chunk(1), chunk(2), chunk(3)]


def fake_generate(monkeypatch, text):
    calls = []

    def generate(question, chunks, provider):
        calls.append(provider.name)
        return GenerationResult(text, provider.model, 5)

    monkeypatch.setattr(rag, "generate", generate)
    return calls


def test_valid_answer_returns_only_cited_sources(monkeypatch):
    fake_generate(monkeypatch, "Answer [S2].")

    result = answer_from_chunks("q", CHUNKS, PROVIDERS)

    assert result["answer"] == "Answer [S2]."
    assert not result["abstained"]
    assert result["model"] == "first-model"
    assert [source["label"] for source in result["sources"]] == ["S2"]
    assert result["sources"][0]["section"] == "Section 2"
    assert result["latency_ms"] >= 0


def test_invented_citation_turns_into_abstention(monkeypatch):
    fake_generate(monkeypatch, "Answer [S1] with a false source [S9].")

    result = answer_from_chunks("q", CHUNKS, PROVIDERS)

    assert result["abstained"]
    assert result["rejection"] == "invalid_citation"
    assert result["raw_answer"] == "Answer [S1] with a false source [S9]."
    assert result["answer"] == REFUSAL
    assert result["sources"] == []


def test_answer_without_citations_turns_into_abstention(monkeypatch):
    fake_generate(monkeypatch, "Answer without a citation.")

    result = answer_from_chunks("q", CHUNKS, PROVIDERS)

    assert result["abstained"]
    assert result["rejection"] == "no_citation"


def test_model_refusal_is_an_abstention(monkeypatch):
    fake_generate(monkeypatch, f"{REFUSAL}.")

    result = answer_from_chunks("q", CHUNKS, PROVIDERS)

    assert result["abstained"]
    assert result["answer"] == REFUSAL
    assert result["model"] == "first-model"
    assert result["rejection"] == "model_refusal"


def test_low_similarity_abstains_without_calling_llm(monkeypatch):
    calls = fake_generate(monkeypatch, "Should not be called [S1].")

    result = answer_from_chunks("q", [chunk(1, score=0.1)], PROVIDERS)

    assert result["abstained"]
    assert result["model"] is None
    assert calls == []


def test_retrieval_only_mode_returns_chunks_without_answer(monkeypatch):
    calls = fake_generate(monkeypatch, "Should not be called [S1].")

    result = answer_from_chunks("q", CHUNKS, {})

    assert result["answer"] is None
    assert not result["abstained"]
    assert result["model"] is None
    assert [source["label"] for source in result["sources"]] == ["S1", "S2", "S3"]
    assert calls == []


def test_falls_back_to_next_provider_when_no_model_is_requested(monkeypatch):
    def generate(question, chunks, provider):
        if provider.name == "first":
            raise ProviderRateLimitedError("first", datetime(2026, 11, 23, 10, 0))
        return GenerationResult("Answer [S1].", provider.model, 5)

    monkeypatch.setattr(rag, "generate", generate)
    assert answer_from_chunks("q", CHUNKS, PROVIDERS)["model"] == "second-model"


def test_requested_model_has_no_fallback(monkeypatch):
    calls = []

    def generate(question, chunks, provider):
        calls.append(provider.name)
        raise LLMError("failed")

    monkeypatch.setattr(rag, "generate", generate)
    with pytest.raises(LLMError):
        answer_from_chunks("q", CHUNKS, PROVIDERS, model="second")
    assert calls == ["second"]


def test_all_providers_failing_raises(monkeypatch):
    def generate(question, chunks, provider):
        raise LLMError(provider.name)

    monkeypatch.setattr(rag, "generate", generate)
    with pytest.raises(LLMError):
        answer_from_chunks("q", CHUNKS, PROVIDERS)


def test_unknown_provider_is_rejected():
    with pytest.raises(UnknownProviderError):
        answer_from_chunks("q", CHUNKS, PROVIDERS, model="nope")


def test_full_flow_with_retrieval(monkeypatch, indexed, embedder):
    fake_generate(monkeypatch, "Memory load [S1].")

    result = answer_question(
        "load", providers=PROVIDERS, mode=FLAT, client=indexed, embedder=embedder
    )

    assert not result["abstained"]
    assert result["sources"][0]["section"] == "Gamma"
