"""Local embeddings with sentence-transformers."""

from collections.abc import Sequence
from functools import lru_cache

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# BGE recommends this prefix for queries only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:  # lazy load: importing torch is slow
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{QUERY_PREFIX}{text}"])[0]


@lru_cache
def get_embedder() -> Embedder:
    """Single instance, because loading the model is the slowest part of startup."""
    return Embedder()
