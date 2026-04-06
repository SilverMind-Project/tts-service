"""SeamlessM4T v2 TTS engine — facebook/seamless-m4t-v2-large.

Meta's SeamlessM4T v2 model provides text-to-speech synthesis in 36 languages
via the Transformers library. Part of the same model family as
facebook/seamless-streaming.

Supports: English, Tamil, Hindi, Telugu, Bengali, + 31 other languages.

Voice cloning: Not supported.
Streaming:     Chunked (full generation then chunked output).

License: CC-BY-NC-4.0 (non-commercial use; commercial use requires
         permission from Meta).

VRAM: ~6-8 GB (float16). Runs on RTX 4080 / DGX Spark.
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

# BCP-47 → SeamlessM4T language code mapping (speech output supported)
_LANG_MAP: dict[str, str] = {
    "en": "eng",
    "ta": "tam",
    "hi": "hin",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
    "bn": "ben",
    "mr": "mar",
    "gu": "guj",
    "pa": "pan",
    "ur": "urd",
    "ar": "arb",
    "zh": "cmn",
    "ja": "jpn",
    "ko": "kor",
    "es": "spa",
    "pt": "por",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "ru": "rus",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "th": "tha",
    "vi": "vie",
    "id": "ind",
    "ms": "zsm",
    "tr": "tur",
    "ro": "ron",
    "uk": "ukr",
    "cs": "ces",
    "fi": "fin",
    "da": "dan",
    "el": "ell",
    "hu": "hun",
}

# Speaker IDs available in SeamlessM4T vocoder (integer IDs)
_DEFAULT_SPEAKERS = [
    ("default", "Default Speaker"),
    ("speaker_0", "Speaker 0"),
    ("speaker_1", "Speaker 1"),
    ("speaker_2", "Speaker 2"),
]


class SeamlessEngine(TTSEngine):
    """SeamlessM4T v2 engine using the Transformers library."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str = "cpu"
        self._sample_rate: int = 16000

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        model_id = config.get(
            "engines.seamless.model_id", "facebook/seamless-m4t-v2-large"
        )
        device = config.get("engines.seamless.device", "cuda")
        dtype_str = config.get("engines.seamless.dtype", "float16")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = "cpu"
            dtype_str = "float32"

        self._device = device
        dtype = getattr(torch, dtype_str, torch.float16)

        logger.info(
            "Loading SeamlessM4T v2 model %s on %s (%s)", model_id, device, dtype_str
        )
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = SeamlessM4Tv2Model.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()

        # SeamlessM4T v2 outputs at 16kHz
        self._sample_rate = self._model.config.sampling_rate or 16000

        logger.info(
            "SeamlessM4T v2 engine loaded — model: %s, sample_rate: %d",
            model_id,
            self._sample_rate,
        )

    async def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="seamless",
            display_name="SeamlessM4T v2 (facebook/seamless-m4t-v2-large)",
            supports_streaming=False,
            supports_cloning=False,
            languages=list(_LANG_MAP.keys()),
            model_id=config.get(
                "engines.seamless.model_id", "facebook/seamless-m4t-v2-large"
            ),
            license="CC-BY-NC-4.0",
        )

    def voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id=vid,
                name=name,
                language="en",
                description=f"SeamlessM4T {name.lower()}",
                sample_rate=self._sample_rate,
            )
            for vid, name in _DEFAULT_SPEAKERS
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
        return SynthesisResult(audio=audio, sample_rate=self._sample_rate)

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        language: str | None,
        speed: float,
    ) -> np.ndarray:
        # Resolve language code
        lang = language or "en"
        tgt_lang = _LANG_MAP.get(lang, "eng")

        # Resolve speaker ID (SeamlessM4T uses integer speaker IDs)
        speaker_id = self._resolve_speaker_id(voice)

        # Tokenize text input
        text_inputs = self._processor(
            text=text, src_lang=tgt_lang, return_tensors="pt"
        ).to(self._device)

        max_tokens = config.get("engines.seamless.max_tokens", 512)

        with torch.no_grad():
            output = self._model.generate(
                **text_inputs,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id,
                generate_speech=True,
                max_new_tokens=int(max_tokens),
            )

        # output is a tuple; the audio waveform is the first element
        # Shape: (batch, channels, samples) or (batch, samples)
        if isinstance(output, tuple):
            audio_tensor = output[0]
        else:
            audio_tensor = output

        audio = audio_tensor.squeeze().cpu().float().numpy()

        # Normalise to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        if speed != 1.0:
            audio = self._apply_speed(audio, speed)

        return audio

    @staticmethod
    def _resolve_speaker_id(voice: str) -> int:
        """Convert voice string to a SeamlessM4T speaker ID."""
        if voice == "default":
            return 0
        # Accept "speaker_N" format
        if voice.startswith("speaker_"):
            try:
                return int(voice.split("_", 1)[1])
            except (ValueError, IndexError):
                pass
        # Try direct int
        try:
            return int(voice)
        except ValueError:
            logger.warning("Unknown voice '%s', using speaker 0", voice)
            return 0

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """Simple time-stretch via linear interpolation."""
        if abs(speed - 1.0) < 0.01:
            return audio
        new_len = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
