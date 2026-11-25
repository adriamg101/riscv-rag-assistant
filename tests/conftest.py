import ipaddress
import re
import socket

import chromadb
import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings

from app import llm
from app.corpus import Node
from app.ingest import run_ingest
from app.store import CHUNKS_COLLECTION, SECTIONS_COLLECTION, get_collection

VOCABULARY = ("add", "branch", "load", "fence")
_real_connect = socket.socket.connect


def _guarded_connect(sock, address):
    """Tests never use the network. Only loopback is allowed (asyncio needs it on Windows)."""
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        if not ipaddress.ip_address(address[0]).is_loopback:
            raise RuntimeError(f"Network connection blocked in tests: {address[0]}")
    return _real_connect(sock, address)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)


@pytest.fixture(autouse=True)
def clean_llm_state():
    llm._limited_until.clear()
    yield
    llm._limited_until.clear()


@pytest.fixture
def client():
    # In memory, because persistent Chroma locks its temp files on Windows.
    # Chroma shares one in-memory system per process, so each test needs a fresh one.
    SharedSystemClient.clear_system_cache()
    yield chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    SharedSystemClient.clear_system_cache()


class KeywordEmbedder:
    """Deterministic embedding where each vocabulary word is an axis."""

    model_name = "keyword-model"

    def _vector(self, text):
        tokens = re.findall(r"[a-z]+", text.lower())
        # The small baseline avoids orthogonal vectors, which HNSW can fail to reach.
        counts = [(word in tokens) + 0.1 for word in VOCABULARY]
        norm = sum(value * value for value in counts) ** 0.5
        return [value / norm for value in counts]

    def embed_passages(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def embedder():
    return KeywordEmbedder()


def words(label, count=200):
    return " ".join([label] * count)


def make_node(key, title, text):
    return Node(
        id=f"mini#{key}",
        title=title,
        path=f"Chapter > {title}",
        level=2,
        parent=None,
        source_url=f"https://example.test/mini.html#{key}",
        text=text,
    )


def _ingest_until_searchable(nodes, client, embedder, attempts=20):
    """A tiny new Chroma index occasionally stays empty for queries, so rebuild it."""
    names = (CHUNKS_COLLECTION, SECTIONS_COLLECTION)
    vector = embedder.embed_query("load")
    for _ in range(attempts):
        run_ingest(nodes, client, embedder)
        collections = [get_collection(client, name) for name in names]
        if all(c.query(query_embeddings=[vector], n_results=1)["ids"][0] for c in collections):
            return
        for name in names:
            client.delete_collection(name)
    raise RuntimeError("Could not build a searchable test index")


@pytest.fixture
def indexed(client, embedder):
    """Alpha covers branch and then add. Beta only covers add."""
    nodes = [
        make_node("alpha", "Alpha", f"{words('branch')}\n\n{words('add')}"),
        make_node("beta", "Beta", words("add")),
        make_node("gamma", "Gamma", words("load")),
    ]
    _ingest_until_searchable(nodes, client, embedder)
    return client
