"""Shared test fixtures for tts-service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from app.services.engine_base import EngineInfo, EngineVoice, SynthesisResult, TTSEngine
from app.services.engine_registry import EngineRegistry


class MockEngine(TTSEngine):
    """Deterministic TTS engine for testing.

    Returns a 440 Hz sine wave of configurable duration.
    """

    SAMPLE_RATE = 24000
    DURATION = 0.5  # seconds

    async def load(self) -> None:
        pass

    async def unload(self) -> None:
        pass

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="mock",
            display_name="Mock Engine",
            supports_streaming=True,
            supports_cloning=False,
            languages=["en"],
            model_id="test/mock",
            license="MIT",
        )

    def voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(id="default", name="Default", language="en"),
            EngineVoice(id="alt", name="Alternate", language="en"),
        ]

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        language: str | None = None,
        speed: float = 1.0,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
    ) -> SynthesisResult:
        audio = self._generate_sine(self.DURATION, self.SAMPLE_RATE)
        return SynthesisResult(audio=audio, sample_rate=self.SAMPLE_RATE)

    @staticmethod
    def _generate_sine(duration: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)


@pytest.fixture
def mock_engine() -> MockEngine:
    return MockEngine()


@pytest.fixture
def audio_samples() -> dict[str, np.ndarray]:
    """Known audio arrays for deterministic testing."""
    sr = 24000
    return {
        "sine_440": MockEngine._generate_sine(0.5, sr, 440.0),
        "silence": np.zeros(sr, dtype=np.float32),
        "short": MockEngine._generate_sine(0.01, sr, 440.0),
    }


@pytest.fixture
def engine_registry(mock_engine: MockEngine) -> EngineRegistry:
    """Pre-loaded registry with a mock engine."""
    reg = EngineRegistry()
    reg._engines["mock"] = mock_engine
    reg._default = "mock"
    return reg


@pytest_asyncio.fixture
async def app_client(engine_registry: EngineRegistry):
    """httpx AsyncClient wired to the FastAPI app with mocked services."""
    import httpx
    from app.main import app
    from app.services.audio_converter import AudioConverter
    from app.services.voice_store import VoiceStore

    # Wire mock services onto app.state
    app.state.engine_registry = engine_registry
    app.state.audio_converter = AudioConverter()
    app.state.voice_store = MagicMock(spec=VoiceStore)
    app.state.voice_store.get.return_value = None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
