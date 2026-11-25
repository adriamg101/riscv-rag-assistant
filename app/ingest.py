"""Incremental, idempotent ingestion into Chroma. Run: python -m app.ingest"""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.chunking import Chunk, chunk_tree
from app.corpus import BLOCK_SEPARATOR, Node, load_nodes
from app.embeddings import Embedder
from app.store import CHUNKS_COLLECTION, SECTIONS_COLLECTION, get_client, get_collection


class PassageEmbedder(Protocol):
    model_name: str

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Record:
    id: str
    document: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class SyncStats:
    added: int
    kept: int
    deleted: int


def content_id(source_url: str, section_id: str, text: str) -> str:
    """SHA-256 of source, section and text. Unchanged content keeps its ID across runs."""
    payload = "\0".join((source_url, section_id, text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_records(chunks: Iterable[Chunk], model_name: str) -> list[Record]:
    return [
        Record(
            id=content_id(chunk.source_url, chunk.section_id, chunk.text),
            document=chunk.text,
            metadata={
                "section": chunk.section,
                "section_id": chunk.section_id,
                "path": chunk.path,
                "source_url": chunk.source_url,
                "chunk_index": chunk.chunk_index,
                "embedding_model": model_name,
            },
        )
        for chunk in chunks
    ]


def section_records(nodes: Iterable[Node], model_name: str) -> list[Record]:
    """Each section is represented by its title and first paragraph."""
    records = []
    for node in nodes:
        if not node.text.strip():
            continue
        first_block = node.text.split(BLOCK_SEPARATOR, 1)[0]
        document = f"{node.title}{BLOCK_SEPARATOR}{first_block}"
        records.append(
            Record(
                id=content_id(node.source_url, node.id, document),
                document=document,
                metadata={
                    "section": node.title,
                    "section_id": node.id,
                    "path": node.path,
                    "source_url": node.source_url,
                    "embedding_model": model_name,
                },
            )
        )
    return records


def sync_collection(
    collection: Collection, records: Iterable[Record], embedder: PassageEmbedder
) -> SyncStats:
    """Make the collection match `records`, computing only the embeddings it needs."""
    wanted = {record.id: record for record in records}
    current = collection.get(include=["metadatas"])
    existing = dict(zip(current["ids"], current["metadatas"], strict=True))
    reusable = {
        record_id
        for record_id, metadata in existing.items()
        if metadata.get("embedding_model") == embedder.model_name
    }

    to_embed = [record for record_id, record in wanted.items() if record_id not in reusable]
    to_update = [
        record
        for record_id, record in wanted.items()
        if record_id in reusable and dict(existing[record_id]) != record.metadata
    ]
    to_delete = [record_id for record_id in existing if record_id not in wanted]

    # Add first and delete last, so a failure loses no content.
    if to_embed:
        collection.upsert(
            ids=[record.id for record in to_embed],
            embeddings=embedder.embed_passages([record.document for record in to_embed]),
            documents=[record.document for record in to_embed],
            metadatas=[record.metadata for record in to_embed],
        )
    if to_update:
        collection.update(
            ids=[record.id for record in to_update],
            metadatas=[record.metadata for record in to_update],
        )
    if to_delete:
        collection.delete(ids=to_delete)
    return SyncStats(len(to_embed), len(wanted) - len(to_embed), len(to_delete))


def run_ingest(
    nodes: Sequence[Node], client: ClientAPI, embedder: PassageEmbedder
) -> dict[str, SyncStats]:
    model = embedder.model_name
    return {
        CHUNKS_COLLECTION: sync_collection(
            get_collection(client, CHUNKS_COLLECTION),
            chunk_records(chunk_tree(nodes), model),
            embedder,
        ),
        SECTIONS_COLLECTION: sync_collection(
            get_collection(client, SECTIONS_COLLECTION),
            section_records(nodes, model),
            embedder,
        ),
    }


def main() -> None:
    results = run_ingest(load_nodes(), get_client(), Embedder())
    for name, stats in results.items():
        print(f"{name}: {stats.added} added, {stats.kept} kept, {stats.deleted} deleted")


if __name__ == "__main__":
    main()
