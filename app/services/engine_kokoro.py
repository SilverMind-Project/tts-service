"""Kokoro engine: hexgrad/Kokoro-82M.

An 82M-parameter English model with American and British voices, Apache-2.0.
It runs comfortably on CPU (about 1 s for a short phrase on the GB10), which
keeps it off a GPU that other services are using.

Voices:    Kokoro's own voice names, such as ``af_heart`` or ``bm_george``.
           ``default`` maps to ``engines.kokoro.default_voice``. Any other
           unknown name raises ``ValueError``, so a caller never gets a
           different voice from the one it asked for.
Language:  ``en-us`` for American pronunciation, ``en-gb`` for British. With
           ``en`` or no language, the voice's own accent is used: ``a*``
           voices are American, ``b*`` voices British. Any other language
           raises ``ValueError``.
Speed:     Passed to Kokoro, which changes the pace without changing pitch.
Words:     Kokoro needs its espeak-ng fallback for words outside its
           dictionary. Without it those words are silently left out, so
           ``load()`` fails instead.
Streaming: Not native. The base class's chunked fallback is used.
Cloning:   Not supported.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np

from app import config
from app.services.engine_base import EngineInfo, EngineVoice, SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24000

# Kokoro pipeline language codes.
_AMERICAN = "a"
_BRITISH = "b"

# English voices from the model's VOICES.md, with the authors' overall grade.
_VOICES: dict[str, str] = {
    "af_heart": "American English female, grade A",
    "af_bella": "American English female, grade A-",
    "af_nicole": "American English female, grade B-",
    "af_aoede": "American English female, grade C+",
    "af_kore": "American English female, grade C+",
    "af_sarah": "American English female, grade C+",
    "af_alloy": "American English female, grade C",
    "af_nova": "American English female, grade C",
    "af_sky": "American English female, grade C-",
    "af_jessica": "American English female, grade D",
    "af_river": "American English female, grade D",
    "am_fenrir": "American English male, grade C+",
    "am_michael": "American English male, grade C+",
    "am_puck": "American English male, grade C+",
    "am_echo": "American English male, grade D",
    "am_eric": "American English male, grade D",
    "am_liam": "American English male, grade D",
    "am_onyx": "American English male, grade D",
    "am_santa": "American English male, grade D-",
    "am_adam": "American English male, grade F+",
    "bf_emma": "British English female, grade B-",
    "bf_isabella": "British English female, grade C",
    "bf_alice": "British English female, grade D",
    "bf_lily": "British English female, grade D",
    "bm_fable": "British English male, grade C",
    "bm_george": "British English male, grade C",
    "bm_lewis": "British English male, grade D+",
    "bm_daniel": "British English male, grade D",
}

# Explicit language -> pipeline. "en" is absent on purpose: it means "the voice's own accent".
_LANGUAGE_TO_PIPELINE: dict[str, str] = {
    "en-us": _AMERICAN,
    "en-gb": _BRITISH,
}


def _voice_accent(voice_id: str) -> str:
    """Pipeline code for a voice's own accent: ``a*`` American, ``b*`` British."""
    return voice_id[0]


def _voice_language(voice_id: str) -> str:
    return "en-us" if _voice_accent(voice_id) == _AMERICAN else "en-gb"


class KokoroEngine(TTSEngine):
    """Kokoro-82M, English only, American and British voices."""

    def __init__(self) -> None:
        self._model = None
        self._pipelines: dict[str, object] = {}
        self._default_voice = ""
        self._repo_id = ""
        self._device = ""
        # One model and one spaCy pipeline are shared, so synthesis runs one at a time.
        self._lock = threading.Lock()

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from kokoro import KModel, KPipeline

        repo_id = config.get("engines.kokoro.repo_id", "hexgrad/Kokoro-82M")
        device = config.get("engines.kokoro.device", "cpu")
        default_voice = config.get("engines.kokoro.default_voice", "af_heart")
        if default_voice not in _VOICES:
            raise ValueError(
                f"engines.kokoro.default_voice {default_voice!r} is not a Kokoro voice"
            )

        logger.info("Loading Kokoro model %s on %s", repo_id, device)
        model = KModel(repo_id=repo_id).to(device).eval()

        pipelines: dict[str, object] = {}
        for code in (_AMERICAN, _BRITISH):
            pipeline = KPipeline(lang_code=code, repo_id=repo_id, model=model)
            if pipeline.g2p.fallback is None:
                raise RuntimeError(
                    "Kokoro's espeak-ng fallback did not start, so words outside its "
                    "dictionary would be silently left out"
                )
            pipelines[code] = pipeline

        # Fetch the default voice now, so the first request doesn't wait on a download.
        pipelines[_voice_accent(default_voice)].load_voice(default_voice)

        self._model = model
        self._pipelines = pipelines
        self._default_voice = default_voice
        self._repo_id = repo_id
        self._device = device
        logger.info("Kokoro loaded, default voice %s", default_voice)

    async def unload(self) -> None:
        self._pipelines = {}
        self._model = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="kokoro",
            display_name="Kokoro (hexgrad/Kokoro-82M)",
            supports_streaming=False,
            supports_cloning=False,
            languages=["en", "en-gb", "en-us"],
            model_id=self._repo_id or config.get("engines.kokoro.repo_id", "hexgrad/Kokoro-82M"),
            license="Apache-2.0",
        )

    def voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id=voice_id,
                name=voice_id,
                language=_voice_language(voice_id),
                description=description,
                supports_cloning=False,
                sample_rate=_SAMPLE_RATE,
            )
            for voice_id, description in _VOICES.items()
        ]

    def resolve_voice(self, voice: str) -> str:
        """Map ``default`` to the configured voice. Unknown names raise ``ValueError``."""
        if voice == "default":
            return self._default_voice
        if voice not in _VOICES:
            raise ValueError(f"Unknown Kokoro voice {voice!r}")
        return voice

    @staticmethod
    def resolve_pipeline(language: str | None, voice_id: str) -> str:
        """Pipeline code for a request. ``en`` or no language uses the voice's own accent."""
        if language is None:
            return _voice_accent(voice_id)
        lang = language.strip().lower().replace("_", "-")
        if lang == "en":
            return _voice_accent(voice_id)
        if lang not in _LANGUAGE_TO_PIPELINE:
            raise ValueError(f"Kokoro speaks only en, en-us and en-gb, not {language!r}")
        return _LANGUAGE_TO_PIPELINE[lang]

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
        if not self._pipelines:
            raise RuntimeError("Kokoro engine is not loaded")
        voice_id = self.resolve_voice(voice)
        code = self.resolve_pipeline(language, voice_id)
        if reference_audio is not None:
            logger.warning("Kokoro can't clone voices; ignoring reference audio for %r", voice)

        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None, self._synthesize_sync, text, voice_id, code, speed,
        )
        return SynthesisResult(audio=audio, sample_rate=_SAMPLE_RATE)

    def _synthesize_sync(self, text: str, voice_id: str, code: str, speed: float) -> np.ndarray:
        pipeline = self._pipelines[code]
        parts: list[np.ndarray] = []
        with self._lock:
            for result in pipeline(text, voice=voice_id, speed=speed):
                if result.audio is not None:
                    parts.append(result.audio.detach().cpu().numpy())
        if not parts:
            raise RuntimeError("Kokoro produced no audio for the text")
        return np.concatenate(parts).astype(np.float32)
