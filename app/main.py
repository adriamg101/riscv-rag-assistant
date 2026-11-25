"""HTTP API. Run: uvicorn app.main:app"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from chromadb.api import ClientAPI
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from app.config import ProviderConfig, is_retrieval_only, load_providers
from app.embeddings import Embedder, get_embedder
from app.llm import LLMError, provider_status
from app.rag import UnknownProviderError, answer_question
from app.retrieve import retrieval_mode
from app.store import CHUNKS_COLLECTION, get_client, get_collection

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class AskRequest(BaseModel):
    question: Question
    model: str | None = None


class SourceOut(BaseModel):
    label: str
    section: str
    path: str
    source_url: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str | None
    sources: list[SourceOut]
    abstained: bool
    model: str | None
    latency_ms: int


class ProviderOut(BaseModel):
    name: str
    model: str
    status: str


class ModelsResponse(BaseModel):
    retrieval_only: bool
    providers: list[ProviderOut]


class HealthResponse(BaseModel):
    status: str
    retrieval_mode: str
    retrieval_only: bool
    indexed_chunks: int


@lru_cache
def get_providers() -> Mapping[str, ProviderConfig]:
    return load_providers()


@lru_cache
def get_mode() -> str:
    return retrieval_mode()


def get_chroma() -> ClientAPI:
    return get_client()


Providers = Annotated[Mapping[str, ProviderConfig], Depends(get_providers)]
Mode = Annotated[str, Depends(get_mode)]
Chroma = Annotated[ClientAPI, Depends(get_chroma)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_mode()  # an invalid RETRIEVAL_MODE stops the API at startup
    yield


app = FastAPI(title="RISC-V RAG Assistant", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest, providers: Providers, mode: Mode, client: Chroma, embedder: EmbedderDep
) -> dict:
    try:
        return answer_question(
            request.question,
            model=request.model,
            providers=providers,
            mode=mode,
            client=client,
            embedder=embedder,
        )
    except UnknownProviderError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except LLMError as error:
        raise HTTPException(status_code=502, detail=f"The LLM provider failed: {error}") from error


@app.get("/models", response_model=ModelsResponse)
def models(providers: Providers) -> ModelsResponse:
    return ModelsResponse(
        retrieval_only=is_retrieval_only(providers),
        providers=[
            ProviderOut(name=name, model=provider.model, status=provider_status(name))
            for name, provider in providers.items()
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health(providers: Providers, mode: Mode, client: Chroma) -> HealthResponse:
    return HealthResponse(
        status="ok",
        retrieval_mode=mode,
        retrieval_only=is_retrieval_only(providers),
        indexed_chunks=get_collection(client, CHUNKS_COLLECTION).count(),
    )
