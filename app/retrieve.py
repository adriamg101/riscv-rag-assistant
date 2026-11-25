"""Chunk retrieval: direct search (flat) or two stages (two_stage)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

from app.embeddings import get_embedder
from app.store import CHUNKS_COLLECTION, SECTIONS_COLLECTION, get_client, get_collection

FLAT = "flat"
TWO_STAGE = "two_stage"
MODES = (FLAT, TWO_STAGE)
TOP_K = 5
TOP_SECTIONS = 3


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    section: str
    section_id: str
    path: str
    source_url: str
    chunk_index: int
    score: float


def retrieval_mode(env: Mapping[str, str] | None = None) -> str:
    """Read RETRIEVAL_MODE, defaulting to flat."""
    if env is None:
        load_dotenv()
        env = os.environ
    mode = env.get("RETRIEVAL_MODE", "").strip().lower() or FLAT
    if mode not in MODES:
        raise ValueError(f"Invalid RETRIEVAL_MODE: {mode}. Use one of: {', '.join(MODES)}")
    return mode


def _query(
    collection: Collection, vector: list[float], k: int, where: dict | None = None
) -> list[RetrievedChunk]:
    available = collection.count()
    if available == 0:
        return []
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(k, available),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    rows = zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        strict=True,
    )
    return [
        RetrievedChunk(
            id=record_id,
            text=document,
            section=metadata["section"],
            section_id=metadata["section_id"],
            path=metadata["path"],
            source_url=metadata["source_url"],
            chunk_index=int(metadata.get("chunk_index", 0)),
            score=1.0 - distance,
        )
        for record_id, document, metadata, distance in rows
    ]


def retrieve(
    question: str,
    mode: str | None = None,
    k: int = TOP_K,
    client: ClientAPI | None = None,
    embedder: QueryEmbedder | None = None,
) -> list[RetrievedChunk]:
    """Return up to k chunks, sorted from highest to lowest similarity."""
    mode = mode or retrieval_mode()
    if mode not in MODES:
        raise ValueError(f"Invalid retrieval mode: {mode}")
    client = client if client is not None else get_client()
    embedder = embedder if embedder is not None else get_embedder()

    vector = embedder.embed_query(question)
    chunks = get_collection(client, CHUNKS_COLLECTION)
    if mode == FLAT:
        return _query(chunks, vector, k)

    sections = _query(get_collection(client, SECTIONS_COLLECTION), vector, TOP_SECTIONS)
    if not sections:
        return []
    section_ids = [section.section_id for section in sections]
    return _query(chunks, vector, k, where={"section_id": {"$in": section_ids}})
