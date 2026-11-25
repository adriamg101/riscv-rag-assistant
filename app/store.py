"""Access to the Chroma vector store."""

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

CHROMA_DIR = Path("chroma_db")
CHUNKS_COLLECTION = "chunks"
SECTIONS_COLLECTION = "sections"


@lru_cache
def get_client(path: Path = CHROMA_DIR) -> ClientAPI:
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def get_collection(client: ClientAPI, name: str) -> Collection:
    # embedding_function=None: vectors always come from app.embeddings.
    return client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
