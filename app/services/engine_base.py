"""Abstract base class for TTS engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SynthesisResult:
    """Raw audio output from a TTS engine."""

    audio: np.ndarray  # float32, mono
    sample_rate: int = 24000


@dataclass
class EngineVoice:
    """A voice provided by an engine."""

    id: str
    name: str
    language: str
    description: str = ""
    supports_cloning: bool = False
    sample_rate: int = 24000


@dataclass
class EngineInfo:
    """Metadata about a TTS engine."""

    name: str
    display_name: str
    supports_streaming: bool = False
    supports_cloning: bool = False
    languages: list[str] = field(default_factory=list)
    model_id: str = ""
    license: str = ""


class TTSEngine(ABC):
    """Base class all TTS engines must implement."""

    @abstractmethod
    async def load(self) -> None:
        """Load model weights onto the device. Called once at startup."""

    @abstractmethod
    async def unload(self) -> None:
        """Release GPU / memory resources."""

    @abstractmethod
    def info(self) -> EngineInfo:
        """Return engine metadata."""

    @abstractmethod
    def voices(self) -> list[EngineVoice]:
        """List the voices this engine provides."""

    @abstractmethod
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
        """Generate audio for the given text.

        Parameters
        ----------
        text:
            Input text to synthesise.
        voice:
            Voice identifier (engine-specific).
        language:
            BCP-47 language code. ``None`` = auto-detect / default.
        speed:
            Playback speed multiplier (1.0 = normal).
        reference_audio:
            Optional reference waveform for voice cloning (float32 numpy).
        reference_sr:
            Sample rate of the reference audio.
        """

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str = "default",
        language: str | None = None,
        speed: float = 1.0,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        chunk_size: int = 4096,
    ) -> AsyncIterator[bytes]:
        """Stream raw PCM int16 chunks as they are generated.

        Default implementation falls back to full synthesis then chunking.
        Engines that support true streaming should override this method.
        """
        result = await self.synthesize(
            text,
            voice=voice,
            language=language,
            speed=speed,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
        )
        pcm = (result.audio * 32767).astype(np.int16).tobytes()
        for offset in range(0, len(pcm), chunk_size):
            yield pcm[offset : offset + chunk_size]
