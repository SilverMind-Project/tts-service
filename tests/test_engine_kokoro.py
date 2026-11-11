"""Tests for the Kokoro engine, using a fake ``kokoro`` module (no model download)."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import torch

from app.services import engine_kokoro
from app.services.engine_kokoro import KokoroEngine
from app.services.engine_registry import _ENGINE_CLASSES, EngineRegistry

_CONFIG = {
    "engines.kokoro.repo_id": "hexgrad/Kokoro-82M",
    "engines.kokoro.device": "cpu",
    "engines.kokoro.default_voice": "af_heart",
}


class _FakeResult:
    def __init__(self, audio: torch.Tensor | None) -> None:
        self.audio = audio


class _FakeModel:
    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        self.device = None

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> _FakeModel:
        return self


class _FakePipeline:
    """Records how it was built and called. Yields two chunks of 0.1 s each."""

    instances: list[_FakePipeline] = []
    fallback: object | None = object()
    chunks: list[torch.Tensor | None] = [torch.full((2400,), 0.25), torch.full((2400,), -0.25)]

    def __init__(self, lang_code: str, repo_id: str, model: _FakeModel) -> None:
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.model = model
        self.g2p = types.SimpleNamespace(fallback=_FakePipeline.fallback)
        self.loaded_voices: list[str] = []
        self.calls: list[tuple[str, str, float]] = []
        _FakePipeline.instances.append(self)

    def load_voice(self, voice: str) -> None:
        self.loaded_voices.append(voice)

    def __call__(self, text: str, voice: str, speed: float):
        self.calls.append((text, voice, speed))
        for chunk in _FakePipeline.chunks:
            yield _FakeResult(chunk)


@pytest.fixture
def fake_kokoro(monkeypatch):
    """Install a fake ``kokoro`` module and patch the engine's config."""
    _FakePipeline.instances = []
    _FakePipeline.fallback = object()
    _FakePipeline.chunks = [torch.full((2400,), 0.25), torch.full((2400,), -0.25)]
    module = types.ModuleType("kokoro")
    module.KModel = _FakeModel
    module.KPipeline = _FakePipeline
    monkeypatch.setitem(sys.modules, "kokoro", module)
    with patch("app.config.get") as mock_get:
        mock_get.side_effect = lambda key, default=None: _CONFIG.get(key, default)
        yield mock_get


@pytest.fixture
async def loaded_engine(fake_kokoro) -> KokoroEngine:
    engine = KokoroEngine()
    await engine.load()
    return engine


def _pipeline(code: str) -> _FakePipeline:
    return next(p for p in _FakePipeline.instances if p.lang_code == code)


class TestImportAndRegistry:
    def test_module_import_does_not_import_kokoro(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "kokoro", None)
        importlib.reload(engine_kokoro)

    def test_registered(self):
        assert _ENGINE_CLASSES["kokoro"] == "app.services.engine_kokoro:KokoroEngine"

    async def test_registry_loads_kokoro_by_name(self, fake_kokoro):
        fake_kokoro.side_effect = lambda key, default=None: {
            **_CONFIG,
            "engines.enabled": ["kokoro"],
            "engines.default": "kokoro",
        }.get(key, default)
        reg = EngineRegistry()
        await reg.load_engines()
        assert reg.default_engine_name == "kokoro"
        assert reg.get("kokoro").info().name == "kokoro"


class TestLoad:
    async def test_missing_dependency_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "kokoro", None)
        with pytest.raises(ImportError):
            await KokoroEngine().load()

    async def test_two_pipelines_share_one_model(self, loaded_engine):
        american, british = _pipeline("a"), _pipeline("b")
        assert american.model is british.model
        assert american.model.device == "cpu"
        assert american.repo_id == british.repo_id == "hexgrad/Kokoro-82M"

    async def test_default_voice_fetched_at_load(self, loaded_engine):
        assert _pipeline("a").loaded_voices == ["af_heart"]

    async def test_missing_espeak_fallback_fails_load(self, fake_kokoro):
        _FakePipeline.fallback = None
        with pytest.raises(RuntimeError, match="espeak-ng"):
            await KokoroEngine().load()

    async def test_unknown_default_voice_fails_load(self, fake_kokoro):
        fake_kokoro.side_effect = lambda key, default=None: {
            **_CONFIG,
            "engines.kokoro.default_voice": "af_nobody",
        }.get(key, default)
        with pytest.raises(ValueError, match="af_nobody"):
            await KokoroEngine().load()


class TestSynthesize:
    async def test_returns_float32_at_24k(self, loaded_engine):
        result = await loaded_engine.synthesize("Traffic.", voice="af_heart", language="en-us")
        assert result.sample_rate == 24000
        assert result.audio.dtype == np.float32
        assert result.audio.shape == (4800,)
        assert result.audio[0] == pytest.approx(0.25)
        assert result.audio[-1] == pytest.approx(-0.25)

    async def test_default_voice_maps_to_configured_voice(self, loaded_engine):
        await loaded_engine.synthesize("Traffic.", voice="default")
        assert _pipeline("a").calls == [("Traffic.", "af_heart", 1.0)]

    async def test_unknown_voice_raises(self, loaded_engine):
        with pytest.raises(ValueError, match="af_nobody"):
            await loaded_engine.synthesize("Traffic.", voice="af_nobody", language="en-us")

    @pytest.mark.parametrize(
        ("voice", "language", "pipeline"),
        [
            ("af_heart", None, "a"),
            ("af_heart", "en", "a"),
            ("bf_emma", None, "b"),
            ("bf_emma", "en", "b"),
            ("af_heart", "en-gb", "b"),
            ("bf_emma", "en-US", "a"),
            ("af_heart", "en_GB", "b"),
        ],
    )
    async def test_language_picks_pronunciation(self, loaded_engine, voice, language, pipeline):
        await loaded_engine.synthesize("Traffic.", voice=voice, language=language)
        assert _pipeline(pipeline).calls == [("Traffic.", voice, 1.0)]

    @pytest.mark.parametrize("language", ["fr", "ta", "en-in"])
    async def test_unsupported_language_raises(self, loaded_engine, language):
        with pytest.raises(ValueError, match="en-us"):
            await loaded_engine.synthesize("Traffic.", voice="af_heart", language=language)

    async def test_speed_passed_to_kokoro(self, loaded_engine):
        await loaded_engine.synthesize("Traffic.", voice="af_heart", language="en-us", speed=1.2)
        assert _pipeline("a").calls == [("Traffic.", "af_heart", 1.2)]

    async def test_no_audio_raises(self, loaded_engine):
        _FakePipeline.chunks = [None]
        with pytest.raises(RuntimeError, match="no audio"):
            await loaded_engine.synthesize("...", voice="af_heart", language="en-us")

    async def test_not_loaded_raises(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            await KokoroEngine().synthesize("Traffic.", voice="af_heart", language="en-us")


class TestMetadata:
    def test_info(self, fake_kokoro):
        info = KokoroEngine().info()
        assert info.name == "kokoro"
        assert info.license == "Apache-2.0"
        assert info.supports_cloning is False
        assert info.languages == ["en", "en-gb", "en-us"]

    def test_voices(self):
        voices = {v.id: v for v in KokoroEngine().voices()}
        assert voices["af_heart"].language == "en-us"
        assert voices["bf_emma"].language == "en-gb"
        assert all(v.sample_rate == 24000 for v in voices.values())
        assert all(v.id[0] in "ab" for v in voices.values())


class TestApi:
    async def test_speech_endpoint_returns_wav(self, loaded_engine):
        from app.main import app
        from app.services.audio_converter import AudioConverter
        from app.services.voice_store import VoiceStore

        reg = EngineRegistry()
        reg._engines["kokoro"] = loaded_engine
        reg._default = "kokoro"
        app.state.engine_registry = reg
        app.state.audio_converter = AudioConverter()
        app.state.voice_store = MagicMock(spec=VoiceStore)
        app.state.voice_store.get.return_value = None
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Traffic, two o'clock, high, two miles.",
                    "voice": "af_heart",
                    "language": "en-us",
                    "speed": 1.0,
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 200
        assert resp.content[:4] == b"RIFF"
        assert _pipeline("a").calls == [("Traffic, two o'clock, high, two miles.", "af_heart", 1.0)]
