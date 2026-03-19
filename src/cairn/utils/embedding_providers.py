"""Pluggable embedding providers for Cairn's vector index.

Cairn supports multiple embedding backends. The default is determined by
available API keys:

    - If VOYAGE_API_KEY is set: VoyageProvider (best quality, recommended by Anthropic)
    - Otherwise: FastEmbedProvider (local ONNX model, no API key needed, ~50MB download)

Usage:
    provider = get_default_provider()
    vectors = await provider.embed_documents(["hello world"])
    query_vec = await provider.embed_query("search text")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger("cairn")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implementations must provide async methods for embedding documents and queries,
    plus metadata properties for dimension validation and provider switching.
    """

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Returns one vector per input text."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query. May use different model parameters than documents."""
        ...

    @property
    def dimensions(self) -> int:
        """Expected dimensionality of output vectors."""
        ...

    @property
    def provider_id(self) -> str:
        """Stable identifier for this provider+model combo.

        Used to detect provider switches and invalidate stale embeddings.
        """
        ...


class VoyageProvider:
    """Voyage AI embedding provider (recommended by Anthropic for use with Claude).

    Requires VOYAGE_API_KEY environment variable. Uses voyage-3-lite (512 dimensions).
    Distinguishes document vs query embeddings via input_type parameter.
    """

    MODEL = "voyage-3-lite"
    DIMENSIONS = 512

    def __init__(self, api_key: str | None = None) -> None:
        try:
            import voyageai  # noqa: F401
        except ImportError:
            raise ImportError(
                "voyageai is required for VoyageProvider. "
                "Install it with: pip install cairn-memory[voyage]"
            )
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            import voyageai
            self._client = voyageai.AsyncClient(api_key=self._api_key)
        return self._client

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        result = await client.embed(texts, model=self.MODEL, input_type="document")
        return result.embeddings

    async def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        result = await client.embed([text], model=self.MODEL, input_type="query")
        return result.embeddings[0]

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    @property
    def provider_id(self) -> str:
        return f"voyage-{self.MODEL}"


class FastEmbedProvider:
    """Local ONNX embedding provider via fastembed (no API key needed).

    Uses BAAI/bge-small-en-v1.5 (384 dimensions, ~50MB model download on first use).
    All computation runs locally. Async methods use asyncio.to_thread since
    fastembed is synchronous.
    """

    MODEL = "BAAI/bge-small-en-v1.5"
    DIMENSIONS = 384

    def __init__(self) -> None:
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError(
                    "fastembed is required for local embeddings. "
                    "Install it with: pip install fastembed"
                )
            logger.info(
                "cairn: initializing local embedding model %s "
                "(~50MB download on first use)",
                self.MODEL,
            )
            self._model = TextEmbedding(model_name=self.MODEL)
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        # fastembed returns a generator of numpy arrays
        return [vec.tolist() for vec in model.embed(texts)]

    def _query_sync(self, text: str) -> list[float]:
        model = self._get_model()
        # query_embed optimizes for query-side embedding if supported
        results = list(model.query_embed(text))
        return results[0].tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._query_sync, text)

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    @property
    def provider_id(self) -> str:
        return f"fastembed-{self.MODEL}"


def get_default_provider() -> EmbeddingProvider:
    """Auto-detect the best available embedding provider.

    Selection logic:
        1. VOYAGE_API_KEY set -> VoyageProvider (best quality)
        2. Otherwise -> FastEmbedProvider (local, no key needed)
    """
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if voyage_key:
        logger.info("cairn: using Voyage AI embeddings (voyage-3-lite)")
        return VoyageProvider(api_key=voyage_key)

    logger.info("cairn: using local embeddings (fastembed, BAAI/bge-small-en-v1.5)")
    return FastEmbedProvider()
