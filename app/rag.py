"""Orchestrates retrieval, abstention, generation and citation validation."""

import time
from collections.abc import Mapping, Sequence
from typing import TypedDict

from chromadb.api import ClientAPI

from app.config import ProviderConfig, is_retrieval_only, load_providers
from app.guardrails import is_refusal, should_abstain, validate_citations
from app.llm import GenerationResult, LLMError, generate
from app.prompts import REFUSAL
from app.retrieve import QueryEmbedder, RetrievedChunk, retrieve


class UnknownProviderError(ValueError):
    """The requested provider is not configured."""


class Source(TypedDict):
    label: str
    section: str
    path: str
    source_url: str
    score: float
    text: str


class RagAnswer(TypedDict):
    answer: str | None
    sources: list[Source]
    abstained: bool
    model: str | None
    latency_ms: int
    raw_answer: str | None  # the model's text before validation
    rejection: str | None  # why the system abstained


def _source(chunk: RetrievedChunk, number: int) -> Source:
    return Source(
        label=f"S{number}",
        section=chunk.section,
        path=chunk.path,
        source_url=chunk.source_url,
        score=round(chunk.score, 4),
        text=chunk.text,
    )


def _check_provider(providers: Mapping[str, ProviderConfig], model: str | None) -> None:
    if model is not None and model not in providers:
        raise UnknownProviderError(f"Provider not configured: {model}")


def _generate_with_fallback(
    question: str,
    chunks: Sequence[RetrievedChunk],
    providers: Mapping[str, ProviderConfig],
    model: str | None,
) -> GenerationResult:
    """Use only the requested model; without one, try each provider in order."""
    candidates = [providers[model]] if model else list(providers.values())
    payload = [{"text": chunk.text} for chunk in chunks]
    last_error: LLMError | None = None
    for provider in candidates:
        try:
            return generate(question, payload, provider)
        except LLMError as error:
            last_error = error
    raise last_error


def answer_from_chunks(
    question: str,
    chunks: Sequence[RetrievedChunk],
    providers: Mapping[str, ProviderConfig],
    model: str | None = None,
) -> RagAnswer:
    """Generate the answer from chunks that were already retrieved."""
    _check_provider(providers, model)
    start = time.perf_counter()

    def result(
        answer: str | None,
        sources: list[Source],
        abstained: bool,
        used: str | None,
        raw_answer: str | None = None,
        rejection: str | None = None,
    ) -> RagAnswer:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return RagAnswer(
            answer=answer,
            sources=sources,
            abstained=abstained,
            model=used,
            latency_ms=latency_ms,
            raw_answer=raw_answer,
            rejection=rejection,
        )

    if should_abstain(chunks):
        return result(REFUSAL, [], True, None, rejection="low_similarity")
    if is_retrieval_only(providers):
        sources = [_source(chunk, number) for number, chunk in enumerate(chunks, start=1)]
        return result(None, sources, False, None)

    generation = _generate_with_fallback(question, chunks, providers, model)
    raw = generation.text
    if is_refusal(raw):
        return result(REFUSAL, [], True, generation.model, raw, "model_refusal")

    check = validate_citations(raw, chunks)
    if not check.valid:  # missing or unknown citations turn the answer into the refusal
        reason = "invalid_citation" if check.invalid else "no_citation"
        return result(REFUSAL, [], True, generation.model, raw, reason)
    sources = [_source(chunks[number - 1], number) for number in check.cited]
    return result(raw, sources, False, generation.model, raw)


def answer_question(
    question: str,
    model: str | None = None,
    providers: Mapping[str, ProviderConfig] | None = None,
    mode: str | None = None,
    client: ClientAPI | None = None,
    embedder: QueryEmbedder | None = None,
) -> RagAnswer:
    """Full pipeline. Latency includes retrieval."""
    providers = load_providers() if providers is None else providers
    _check_provider(providers, model)
    start = time.perf_counter()
    chunks = retrieve(question, mode=mode, client=client, embedder=embedder)
    response = answer_from_chunks(question, chunks, providers, model)
    response["latency_ms"] = round((time.perf_counter() - start) * 1000)
    return response
