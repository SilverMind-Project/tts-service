"""Indic Parler TTS engine — ai4bharat/indic-parler-tts.

A 938M-parameter model fine-tuned for Indian languages with emotion support.
Supports: English (Indian accent), Tamil, Hindi, Telugu, Kannada, Malayalam,
          Bengali, Marathi, Gujarati, Odia, Punjabi.

Voice control is via text description prompts rather than voice cloning.
Example: "A calm female voice with a clear Tamil accent speaks slowly."

Streaming: Supported via chunked generation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np
import torch

from app import config
from app.services.engine_base import EngineInfo, EngineVoice, SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)

_LANG_MAP = {
    "en": "english", "ta": "tamil", "hi": "hindi", "te": "telugu",
    "kn": "kannada", "ml": "malayalam", "bn": "bengali", "mr": "marathi",
    "gu": "gujarati", "or": "odia", "pa": "punjabi",
}

# Pre-built voice descriptions (used as "voice" presets)
_VOICE_PRESETS: dict[str, str] = {
    "female_calm": (
        "A calm female speaker with a clear Indian English accent "
        "delivers the text at a moderate pace in a warm, reassuring tone."
    ),
    "male_clear": (
        "A male speaker with a clear Indian English accent "
        "speaks at a steady pace with a confident, friendly tone."
    ),
    "female_tamil": (
        "A gentle female speaker with a native Tamil accent "
        "reads the text slowly and clearly."
    ),
    "male_tamil": (
        "A male speaker with a native Tamil accent "
        "speaks clearly at a moderate pace."
    ),
    "female_elderly_friendly": (
        "A warm, gentle female voice speaks slowly and clearly, "
        "as if speaking to an elderly person. The tone is patient and kind."
    ),
}


class ParlerEngine(TTSEngine):
    """Indic Parler TTS engine."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device: str = "cpu"

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        model_id = config.get("engines.parler.model_id", "ai4bharat/indic-parler-tts")
        device = config.get("engines.parler.device", "cuda")
        dtype_str = config.get("engines.parler.dtype", "float16")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = "cpu"
            dtype_str = "float32"

        self._device = device
        dtype = getattr(torch, dtype_str, torch.float16)

        logger.info("Loading Parler TTS model %s on %s (%s)", model_id, device, dtype_str)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()

        logger.info("Parler TTS engine loaded")

    async def unload(self) -> None:
        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="parler",
            display_name="Indic Parler TTS (ai4bharat/indic-parler-tts)",
            supports_streaming=True,
            supports_cloning=False,
            languages=list(_LANG_MAP.keys()),
            model_id=config.get("engines.parler.model_id", "ai4bharat/indic-parler-tts"),
            license="Apache-2.0",
        )

    def voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id=preset_id,
                name=preset_id.replace("_", " ").title(),
                language="ta" if "tamil" in preset_id else "en",
                description=desc,
            )
            for preset_id, desc in _VOICE_PRESETS.items()
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
        audio = await loop.run_in_executor(
            None, self._synthesize_sync, text, voice, language, speed,
        )
        sample_rate = self._model.config.sampling_rate if self._model else 24000
        return SynthesisResult(audio=audio, sample_rate=sample_rate)

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        language: str | None,
        speed: float,
    ) -> np.ndarray:
        # Resolve voice description
        if voice in _VOICE_PRESETS:
            description = _VOICE_PRESETS[voice]
        elif voice == "default":
            description = _VOICE_PRESETS["female_calm"]
        else:
            # Treat as a raw description string
            description = voice

        # Inject language hint if provided
        if language and language in _LANG_MAP:
            lang_name = _LANG_MAP[language]
            if lang_name not in description.lower():
                description += f" The text is in {lang_name}."

        desc_ids = self._tokenizer(description, return_tensors="pt").input_ids.to(self._device)
        prompt_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        max_tokens = config.get("engines.parler.max_tokens", 2048)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=desc_ids,
                prompt_input_ids=prompt_ids,
                max_new_tokens=int(max_tokens),
            )

        audio = generation.cpu().numpy().squeeze().astype(np.float32)

        # Normalise
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        if speed != 1.0:
            new_len = int(len(audio) / speed)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        return audio

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
        """Stream audio chunks.

        Parler TTS supports chunked generation via its streamer API.
        This implementation uses full generation then chunked output.
        """
        result = await self.synthesize(
            text, voice=voice, language=language, speed=speed,
        )
        pcm = (result.audio * 32767).astype(np.int16).tobytes()
        for offset in range(0, len(pcm), chunk_size):
            yield pcm[offset : offset + chunk_size]
