import hashlib

from app.corpus import Node
from app.ingest import SyncStats, content_id, run_ingest
from app.store import CHUNKS_COLLECTION, SECTIONS_COLLECTION, get_collection


class FakeEmbedder:
    def __init__(self, model_name="fake-model"):
        self.model_name = model_name
        self.embedded: list[str] = []

    def embed_passages(self, texts):
        self.embedded.extend(texts)
        return [
            [byte / 255 + 0.01 for byte in hashlib.sha256(text.encode()).digest()[:8]]
            for text in texts
        ]


def make_node(key, text):
    return Node(
        id=f"mini#{key}",
        title=f"Section {key}",
        path=f"Chapter > Section {key}",
        level=2,
        parent=None,
        source_url=f"https://example.test/mini.html#{key}",
        text=text,
    )


def words(label, count=200):
    return " ".join([label] * count)


def test_first_ingest_adds_chunks_and_sections(client):
    stats = run_ingest([make_node("a", "one"), make_node("b", "two")], client, FakeEmbedder())
    assert stats[CHUNKS_COLLECTION] == SyncStats(added=2, kept=0, deleted=0)
    assert stats[SECTIONS_COLLECTION] == SyncStats(added=2, kept=0, deleted=0)


def test_reingest_is_idempotent_and_computes_no_embeddings(client):
    nodes = [make_node("a", "one"), make_node("b", "two")]
    run_ingest(nodes, client, FakeEmbedder())
    embedder = FakeEmbedder()

    stats = run_ingest(nodes, client, embedder)

    assert stats[CHUNKS_COLLECTION] == SyncStats(added=0, kept=2, deleted=0)
    assert stats[SECTIONS_COLLECTION] == SyncStats(added=0, kept=2, deleted=0)
    assert embedder.embedded == []
    assert get_collection(client, CHUNKS_COLLECTION).count() == 2


def test_changed_text_only_embeds_the_new_version(client):
    run_ingest([make_node("a", "one"), make_node("b", "two")], client, FakeEmbedder())
    embedder = FakeEmbedder()

    stats = run_ingest([make_node("a", "one"), make_node("b", "three")], client, embedder)

    assert stats[CHUNKS_COLLECTION] == SyncStats(added=1, kept=1, deleted=1)
    assert all("three" in text for text in embedder.embedded)
    documents = get_collection(client, CHUNKS_COLLECTION).get()["documents"]
    assert not any("two" in document for document in documents)


def test_removed_section_is_deleted_from_both_collections(client):
    run_ingest([make_node("a", "one"), make_node("b", "two")], client, FakeEmbedder())

    stats = run_ingest([make_node("a", "one")], client, FakeEmbedder())

    assert stats[CHUNKS_COLLECTION].deleted == 1
    assert stats[SECTIONS_COLLECTION].deleted == 1
    assert get_collection(client, SECTIONS_COLLECTION).count() == 1


def test_shifted_chunk_updates_metadata_without_embedding(client):
    original = make_node("a", f"{words('first')}\n\n{words('second')}")
    run_ingest([original], client, FakeEmbedder())
    embedder = FakeEmbedder()

    stats = run_ingest([make_node("a", words("second"))], client, embedder)

    assert stats[CHUNKS_COLLECTION] == SyncStats(added=0, kept=1, deleted=1)
    assert len(embedder.embedded) == 1  # only the section, whose first paragraph changed
    stored = get_collection(client, CHUNKS_COLLECTION).get(include=["metadatas"])
    assert [metadata["chunk_index"] for metadata in stored["metadatas"]] == [0]


def test_changing_embedding_model_recomputes_vectors(client):
    nodes = [make_node("a", "one")]
    run_ingest(nodes, client, FakeEmbedder("model-1"))
    embedder = FakeEmbedder("model-2")

    stats = run_ingest(nodes, client, embedder)

    assert stats[CHUNKS_COLLECTION] == SyncStats(added=1, kept=0, deleted=0)
    assert get_collection(client, CHUNKS_COLLECTION).count() == 1
    stored = get_collection(client, CHUNKS_COLLECTION).get(include=["metadatas"])
    assert stored["metadatas"][0]["embedding_model"] == "model-2"


def test_content_id_depends_on_source_section_and_text():
    base = content_id("url", "sec", "text")
    assert base == content_id("url", "sec", "text")
    assert base != content_id("other", "sec", "text")
    assert base != content_id("url", "other", "text")
    assert base != content_id("url", "sec", "changed")
