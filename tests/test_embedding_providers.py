"""Tests for embedding provider abstraction and factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cairn.utils.embedding_providers import (
    EmbeddingProvider,
    FastEmbedProvider,
    VoyageProvider,
    get_default_provider,
)


class TestGetDefaultProvider:
    def test_returns_voyage_when_key_set(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        provider = get_default_provider()
        assert isinstance(provider, VoyageProvider)

    def test_returns_fastembed_when_no_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        provider = get_default_provider()
        assert isinstance(provider, FastEmbedProvider)


class TestVoyageProvider:
    def test_provider_id(self):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}):
            p = VoyageProvider(api_key="test")
        assert p.provider_id == "voyage-voyage-3-lite"

    def test_dimensions(self):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}):
            p = VoyageProvider(api_key="test")
        assert p.dimensions == 512

    def test_import_error_without_voyageai(self, monkeypatch):
        """VoyageProvider raises ImportError if voyageai not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "voyageai":
                raise ImportError("no voyageai")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="voyageai is required"):
            VoyageProvider(api_key="test")


class TestFastEmbedProvider:
    def test_provider_id(self):
        p = FastEmbedProvider()
        assert p.provider_id == "fastembed-BAAI/bge-small-en-v1.5"

    def test_dimensions(self):
        p = FastEmbedProvider()
        assert p.dimensions == 384

    def test_model_lazy_init(self):
        """Model is not loaded on construction."""
        p = FastEmbedProvider()
        assert p._model is None


class TestProtocolCompliance:
    def test_voyage_satisfies_protocol(self):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}):
            p = VoyageProvider(api_key="test")
        assert isinstance(p, EmbeddingProvider)

    def test_fastembed_satisfies_protocol(self):
        p = FastEmbedProvider()
        assert isinstance(p, EmbeddingProvider)
