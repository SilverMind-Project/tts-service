"""Fish Speech S2-Pro TTS engine — fishaudio/s2-pro.

A dual autoregressive architecture TTS model supporting 80+ languages
with high-quality voice cloning from 10-30 second reference audio.

Supports: English, Tamil, Hindi, Telugu, + 80 other languages.

Voice cloning: Yes — provide a reference audio sample (10-30s WAV).
Streaming:     Chunked (full generation then chunked output).

License: Fish Audio Research License (free for research & non-commercial use;
         commercial use requires a separate licence from Fish Audio).

Installation: Requires the fish-speech package from GitHub:
    pip install git+https://github.com/fishaudio/fish-speech.git

VRAM: ~8-12 GB (bf16). Tested on RTX 4080 / DGX Spark.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import torch

from app import config
from app.services.engine_base import EngineInfo, EngineVoice, SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)

# Language codes supported by Fish Speech (subset of 80+ languages)
_SUPPORTED_LANGUAGES = [
    "en", "ta", "hi", "te", "kn", "ml", "bn", "mr", "gu", "pa", "ur",
    "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de", "it",
]

_DEFAULT_VOICES = [
    ("default", "Default Voice"),
]


class FishSpeechEngine(TTSEngine):
    """Fish Speech S2-Pro engine using the fish-speech package."""

    def __init__(self) -> None:
        self._llm_model = None
        self._decode_model = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._dtype = torch.bfloat16
        self._sample_rate: int = 44100

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        device = config.get("engines.fish_speech.device", "cuda")
        dtype_str = config.get("engines.fish_speech.dtype", "bfloat16")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = "cpu"
            dtype_str = "float32"

        self._device = device
        self._dtype = getattr(torch, dtype_str, torch.bfloat16)

        model_id = config.get(
            "engines.fish_speech.model_id", "fishaudio/s2-pro"
        )

        logger.info(
            "Loading Fish Speech model %s on %s (%s)", model_id, device, dtype_str
        )

        try:
            self._load_via_fish_speech_package(model_id, device, dtype_str)
        except ImportError:
            logger.error(
                "fish-speech package not found. Install with: "
                "pip install git+https://github.com/fishaudio/fish-speech.git"
            )
            raise

    def _load_via_fish_speech_package(
        self, model_id: str, device: str, dtype_str: str
    ) -> None:
        """Load models using the fish-speech package."""
        from fish_speech.models.text2semantic.llama import (
            BaseModelArgs,
            DualARTransformer,
        )
        from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
        from fish_speech.tokenizer import FishTokenizer

        # Load the tokenizer
        self._tokenizer = FishTokenizer.from_pretrained(model_id)

        # Load the dual-AR text-to-semantic model
        self._llm_model = DualARTransformer.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._llm_model.eval()

        # Load the VQGAN / Firefly decoder
        decoder_id = config.get(
            "engines.fish_speech.decoder_id", model_id
        )
        self._decode_model = FireflyArchitecture.from_pretrained(
            decoder_id,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._decode_model.eval()

        # Fish Speech outputs at 44.1kHz
        self._sample_rate = config.get("engines.fish_speech.sample_rate", 44100)

        logger.info("Fish Speech engine loaded — model: %s", model_id)

    async def unload(self) -> None:
        del self._llm_model, self._decode_model, self._tokenizer
        self._llm_model = self._decode_model = self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="fish_speech",
            display_name="Fish Speech S2-Pro (fishaudio/s2-pro)",
            supports_streaming=True,
            supports_cloning=True,
            languages=_SUPPORTED_LANGUAGES,
            model_id=config.get("engines.fish_speech.model_id", "fishaudio/s2-pro"),
            license="Fish Audio Research License",
        )

    def voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id=vid,
                name=name,
                language="en",
                description=f"Fish Speech {name.lower()}",
                supports_cloning=True,
                sample_rate=self._sample_rate,
            )
            for vid, name in _DEFAULT_VOICES
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
            None,
            self._synthesize_sync,
            text,
            voice,
            language,
            speed,
            reference_audio,
            reference_sr,
        )
        return SynthesisResult(audio=audio, sample_rate=self._sample_rate)

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        language: str | None,
        speed: float,
        reference_audio: np.ndarray | None,
        reference_sr: int,
    ) -> np.ndarray:
        from fish_speech.conversation import TextPart, VQPart
        from fish_speech.inference_engine import generate_semantic_tokens

        # Encode reference audio for voice cloning if provided
        prompt_tokens = None
        if reference_audio is not None:
            prompt_tokens = self._encode_reference(reference_audio, reference_sr)

        max_tokens = config.get("engines.fish_speech.max_tokens", 4096)
        temperature = config.get("engines.fish_speech.temperature", 0.7)
        top_p = config.get("engines.fish_speech.top_p", 0.8)
        repetition_penalty = config.get(
            "engines.fish_speech.repetition_penalty", 1.2
        )

        # Generate semantic tokens from text
        with torch.no_grad():
            semantic_tokens = generate_semantic_tokens(
                model=self._llm_model,
                tokenizer=self._tokenizer,
                text=text,
                prompt_tokens=prompt_tokens,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            )

            # Decode semantic tokens to audio waveform
            audio_tensor = self._decode_model.decode(
                indices=semantic_tokens.unsqueeze(0),
            )

        audio = audio_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Normalise to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        if speed != 1.0:
            audio = self._apply_speed(audio, speed)

        return audio

    def _encode_reference(
        self, reference_audio: np.ndarray, reference_sr: int
    ) -> torch.Tensor:
        """Encode reference audio into prompt tokens for voice cloning."""
        import torchaudio

        audio_tensor = torch.from_numpy(reference_audio).float().unsqueeze(0)

        # Resample to model's expected sample rate if needed
        if reference_sr != self._sample_rate:
            resampler = torchaudio.transforms.Resample(
                reference_sr, self._sample_rate
            )
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.to(self._device)

        with torch.no_grad():
            # Encode reference audio to VQ codes
            codes = self._decode_model.encode(audio_tensor)

        return codes.squeeze(0)

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """Simple time-stretch via linear interpolation."""
        if abs(speed - 1.0) < 0.01:
            return audio
        new_len = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

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

        Fish Speech supports streaming via its dual-AR architecture.
        This implementation uses full generation then chunked output.
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
