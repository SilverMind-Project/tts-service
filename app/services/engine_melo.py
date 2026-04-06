"""MeloTTS engine — myshell-ai/MeloTTS.

A lightweight (~100M param) TTS model that runs efficiently on CPU.
Supports English with Indian accent (EN-IN speaker), plus US, UK, AU accents.

This engine is ideal as a fast fallback when GPU is unavailable or for
low-latency English-only synthesis.

Voice cloning: Not supported.
Streaming:     Not natively supported (batch inference only). Audio is generated
               in full then chunked for streaming responses.
Tamil:         Not supported — use svara or parler for Tamil.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import numpy as np

from app import config
from app.services.engine_base import EngineInfo, EngineVoice, SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)

_MELO_SPEAKERS = {
    "EN-IN": ("Indian English Female", "en"),
    "EN-US": ("American English Female", "en"),
    "EN-BR": ("British English Female", "en"),
    "EN-AU": ("Australian English Female", "en"),
    "EN-Default": ("English Default", "en"),
}


class MeloEngine(TTSEngine):
    """MeloTTS engine — CPU-friendly, low-latency English TTS."""

    def __init__(self) -> None:
        self._tts = None
        self._speaker_ids: dict[str, int] = {}

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from melo.api import TTS

        device = config.get("engines.melo.device", "auto")
        logger.info("Loading MeloTTS (device=%s)", device)

        self._tts = TTS(language="EN", device=device)
        self._speaker_ids = self._tts.hps.data.spk2id
        logger.info("MeloTTS loaded — speakers: %s", list(self._speaker_ids.keys()))

    async def unload(self) -> None:
        del self._tts
        self._tts = None

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="melo",
            display_name="MeloTTS (myshell-ai/MeloTTS)",
            supports_streaming=False,
            supports_cloning=False,
            languages=["en"],
            model_id="myshell-ai/MeloTTS",
            license="MIT",
        )

    def voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id=spk_key,
                name=_MELO_SPEAKERS.get(spk_key, (spk_key, "en"))[0],
                language=_MELO_SPEAKERS.get(spk_key, (spk_key, "en"))[1],
                description=f"MeloTTS {spk_key} voice (CPU-optimised)",
            )
            for spk_key in self._speaker_ids
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
        loop = asyncio.get_event_loop()
        audio, sr = await loop.run_in_executor(
            None, self._synthesize_sync, text, voice, speed,
        )
        return SynthesisResult(audio=audio, sample_rate=sr)

    def _synthesize_sync(
        self, text: str, voice: str, speed: float,
    ) -> tuple[np.ndarray, int]:
        import soundfile as sf

        # Resolve speaker
        if voice == "default":
            voice = config.get("engines.melo.default_voice", "EN-IN")

        speaker_id = self._speaker_ids.get(voice)
        if speaker_id is None:
            logger.warning("Unknown MeloTTS voice '%s', using EN-IN", voice)
            speaker_id = self._speaker_ids.get("EN-IN", 0)

        # MeloTTS writes to a file — use a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp_path = tmp.name

        self._tts.tts_to_file(text, speaker_id, tmp_path, speed=speed)

        audio, sr = sf.read(tmp_path, dtype="float32")
        Path(tmp_path).unlink(missing_ok=True)

        # Normalise
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        return audio, sr
