import pytest

from app import retrieve as retrieve_module
from app.retrieve import FLAT, TWO_STAGE, retrieval_mode, retrieve


def test_flat_returns_top_k_sorted_by_score(indexed, embedder):
    chunks = retrieve("add", mode=FLAT, k=2, client=indexed, embedder=embedder)

    assert len(chunks) == 2
    assert {chunk.section_id for chunk in chunks} == {"mini#alpha", "mini#beta"}
    assert chunks[0].score >= chunks[1].score
    assert chunks[0].score == pytest.approx(1.0, abs=0.01)


def test_retrieved_chunk_keeps_metadata(indexed, embedder):
    chunk = retrieve("load", mode=FLAT, k=1, client=indexed, embedder=embedder)[0]

    assert chunk.section == "Gamma"
    assert chunk.path == "Chapter > Gamma"
    assert chunk.source_url == "https://example.test/mini.html#gamma"
    assert chunk.chunk_index == 0
    assert chunk.text.startswith("Gamma\n\nload")


def test_two_stage_only_searches_inside_best_sections(indexed, embedder, monkeypatch):
    monkeypatch.setattr(retrieve_module, "TOP_SECTIONS", 1)

    flat = retrieve("add", mode=FLAT, k=3, client=indexed, embedder=embedder)
    two_stage = retrieve("add", mode=TWO_STAGE, k=3, client=indexed, embedder=embedder)

    assert "mini#alpha" in {chunk.section_id for chunk in flat}
    assert {chunk.section_id for chunk in two_stage} == {"mini#beta"}


def test_empty_index_returns_nothing(client, embedder):
    assert retrieve("add", mode=FLAT, client=client, embedder=embedder) == []
    assert retrieve("add", mode=TWO_STAGE, client=client, embedder=embedder) == []


def test_invalid_mode_is_rejected(indexed, embedder):
    with pytest.raises(ValueError):
        retrieve("add", mode="other", client=indexed, embedder=embedder)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, FLAT),
        ({"RETRIEVAL_MODE": ""}, FLAT),
        ({"RETRIEVAL_MODE": " TWO_STAGE "}, TWO_STAGE),
    ],
)
def test_retrieval_mode_from_env(env, expected):
    assert retrieval_mode(env) == expected


def test_invalid_retrieval_mode_in_env_fails():
    with pytest.raises(ValueError):
        retrieval_mode({"RETRIEVAL_MODE": "hybrid"})
