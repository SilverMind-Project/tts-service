"""OpenAI Edge TTS engine — pass-through to an openai-edge-tts service.

A lightweight pass-through engine that proxies TTS requests to a remote
openai-edge-tts instance (https://github.com/travisvn/openai-edge-tts).
No local model loading — all synthesis happens on the remote service using
Microsoft Edge TTS voices.

Voice cloning: Not supported.
Streaming:     Supported (proxied from remote service).
GPU:           Not required (remote service handles synthesis).
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterator

import numpy as np

from app import config
from app.services.engine_base import EngineInfo, EngineVoice, SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)

# Popular Edge TTS voices for Indian languages and English
_DEFAULT_VOICES = {
    "en-IN-NeerjaExpressiveNeural": ("Neerja Expressive (Indian English Female)", "en"),
    "en-IN-NeerjaNeural": ("Neerja (Indian English Female)", "en"),
    "en-IN-PrabhatNeural": ("Prabhat (Indian English Male)", "en"),
    "en-US-JennyNeural": ("Jenny (US English Female)", "en"),
    "en-US-GuyNeural": ("Guy (US English Male)", "en"),
    "en-GB-SoniaNeural": ("Sonia (British English Female)", "en"),
    "ta-IN-PallaviNeural": ("Pallavi (Tamil Female)", "ta"),
    "ta-IN-ValluvarNeural": ("Valluvar (Tamil Male)", "ta"),
    "hi-IN-SwaraNeural": ("Swara (Hindi Female)", "hi"),
    "hi-IN-MadhurNeural": ("Madhur (Hindi Male)", "hi"),
    "te-IN-ShrutiNeural": ("Shruti (Telugu Female)", "te"),
    "te-IN-MohanNeural": ("Mohan (Telugu Male)", "te"),
    "kn-IN-SapnaNeural": ("Sapna (Kannada Female)", "kn"),
    "ml-IN-SobhanaNeural": ("Sobhana (Malayalam Female)", "ml"),
}


class EdgeTTSEngine(TTSEngine):
    """Pass-through engine proxying to an openai-edge-tts service."""

    def __init__(self) -> None:
        self._base_url: str = ""
        self._default_voice: str = ""
        self._default_speed: float = 1.0
        self._session = None

    async def load(self) -> None:
        import httpx

        self._base_url = config.get(
            "engines.edge_tts.base_url", "http://localhost:5050/v1"
        ).rstrip("/")
        self._default_voice = config.get(
            "engines.edge_tts.default_voice", "en-IN-NeerjaExpressiveNeural"
        )
        self._default_speed = float(config.get("engines.edge_tts.default_speed", 0.85))
        self._session = httpx.AsyncClient(timeout=60.0)

        logger.info(
            "EdgeTTS engine configured — endpoint=%s, default_voice=%s, default_speed=%.2f",
            self._base_url,
            self._default_voice,
            self._default_speed,
        )

    async def unload(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="edge_tts",
            display_name="OpenAI Edge TTS (Microsoft Edge voices)",
            supports_streaming=True,
            supports_cloning=False,
            languages=["en", "ta", "hi", "te", "kn", "ml", "bn", "mr", "gu"],
            model_id="travisvn/openai-edge-tts",
            license="MIT",
        )

    def voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id=voice_id,
                name=display_name,
                language=lang,
                description=f"Microsoft Edge TTS — {display_name}",
                sample_rate=24000,
            )
            for voice_id, (display_name, lang) in _DEFAULT_VOICES.items()
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
        import soundfile as sf

        if voice == "default":
            voice = self._default_voice
        if speed == 1.0:
            speed = self._default_speed

        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "wav",
        }

        response = await self._session.post(
            f"{self._base_url}/audio/speech",
            json=payload,
        )
        response.raise_for_status()

        audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Normalise
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        return SynthesisResult(audio=audio, sample_rate=sr)

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
        if voice == "default":
            voice = self._default_voice
        if speed == 1.0:
            speed = self._default_speed

        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "pcm",
        }

        async with self._session.stream(
            "POST",
            f"{self._base_url}/audio/speech",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size):
                yield chunk
