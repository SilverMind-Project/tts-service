"""Tests for the engine registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.engine_registry import EngineRegistry
from tests.conftest import MockEngine


class TestEngineRegistry:
    def test_get_default(self, engine_registry: EngineRegistry):
        engine = engine_registry.get()
        assert engine is not None
        assert engine.info().name == "mock"

    def test_get_by_name(self, engine_registry: EngineRegistry):
        engine = engine_registry.get("mock")
        assert engine is not None
        assert engine.info().name == "mock"

    def test_get_unknown_falls_back_to_default(self, engine_registry: EngineRegistry):
        engine = engine_registry.get("nonexistent")
        assert engine is not None
        assert engine.info().name == "mock"

    def test_get_none_returns_default(self, engine_registry: EngineRegistry):
        engine = engine_registry.get(None)
        assert engine is not None

    def test_available_engines(self, engine_registry: EngineRegistry):
        available = engine_registry.available_engines()
        assert "mock" in available
        assert len(available) == 1

    def test_default_engine_name(self, engine_registry: EngineRegistry):
        assert engine_registry.default_engine_name == "mock"

    def test_empty_registry(self):
        reg = EngineRegistry()
        assert reg.get() is None
        assert reg.available_engines() == {}
        assert reg.default_engine_name == ""

    @pytest.mark.asyncio
    async def test_unload_engines(self, engine_registry: EngineRegistry):
        await engine_registry.unload_engines()
        assert engine_registry.available_engines() == {}

    @pytest.mark.asyncio
    async def test_load_unknown_engine_skipped(self):
        reg = EngineRegistry()
        with patch("app.config.get") as mock_config:
            mock_config.side_effect = lambda key, default=None: {
                "engines.enabled": ["nonexistent_engine"],
                "engines.default": "nonexistent_engine",
            }.get(key, default)
            await reg.load_engines()
        assert reg.available_engines() == {}
