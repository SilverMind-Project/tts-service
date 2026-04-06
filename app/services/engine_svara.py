"""Svara TTS engine — kenpath/svara-tts-v1.

Svara is an Orpheus-style discrete-audio-token model. The upstream demo uses a
prompt shaped like ``"{Language} ({Gender}): {text} <style>"`` with sentinel
IDs wrapped around the tokenized prompt, and decodes 7-way interleaved SNAC
codes from the generated token stream.

This adapter follows that reference flow and keeps a few backwards-compatible
voice aliases (for example ``speaker_0`` and ``ta_female``) so existing callers
continue to work.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator

import numpy as np
import torch

from app import config
from app.services.engine_base import EngineInfo, EngineVoice, SynthesisResult, TTSEngine

logger = logging.getLogger(__name__)

_LANGUAGE_NAMES: dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "bho": "Bhojpuri",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hne": "Chhattisgarhi",
    "kn": "Kannada",
    "mai": "Maithili",
    "mag": "Magahi",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "od": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

_DISPLAY_SPEAKERS = [
    ("en_female", "Default Female", "Female"),
    ("en_male", "Default Male", "Male"),
]

_FEMALE_ALIASES = {
    "",
    "default",
    "speaker_0",
    "speaker0",
    "female",
    "f",
    "woman",
    "girl",
    "en_female",
    "ta_female",
    "hi_female",
    "te_female",
    "kn_female",
    "ml_female",
    "mr_female",
    "bn_female",
    "gu_female",
    "pa_female",
}

_MALE_ALIASES = {
    "speaker_1",
    "speaker1",
    "male",
    "m",
    "man",
    "boy",
    "en_male",
    "ta_male",
    "hi_male",
    "te_male",
    "kn_male",
    "ml_male",
    "mr_male",
    "bn_male",
    "gu_male",
    "pa_male",
}

_VOICE_ALIAS_TO_LANGUAGE: dict[str, str] = {
    "en_female": "en",
    "en_male": "en",
    "ta_female": "ta",
    "ta_male": "ta",
    "hi_female": "hi",
    "hi_male": "hi",
    "te_female": "te",
    "te_male": "te",
    "kn_female": "kn",
    "kn_male": "kn",
    "ml_female": "ml",
    "ml_male": "ml",
    "mr_female": "mr",
    "mr_male": "mr",
    "bn_female": "bn",
    "bn_male": "bn",
    "gu_female": "gu",
    "gu_male": "gu",
    "pa_female": "pa",
    "pa_male": "pa",
}

_DIRECT_SPEAKER_RE = re.compile(
    r"^(?P<language>[A-Za-z][A-Za-z\s-]+?)\s*\((?P<gender>Male|Female)\)$",
    re.IGNORECASE,
)
_STYLE_TAG_RE = re.compile(
    r"<(neutral|formal|chat|clear|happy|surprise|sad|fear|anger|disgust|"
    r"narrative|enthusiastic|laugh|yawn|angry)>",
    re.IGNORECASE,
)
_CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

_START_TOKEN_ID = 128259
_HEAD_TOKEN_ID = 128257
_AUDIO_EOS_TOKEN_ID = 128258
_PROMPT_END_TOKEN_ID = 128260
_PROMPT_SEPARATOR_TOKEN_ID = 128009
_AUDIO_TOKEN_OFFSET = 128266
_CODEBOOK_SIZE = 4096
_SNAC_SAMPLE_RATE = 24000


class SvaraEngine(TTSEngine):
    """Svara TTS engine using transformers + SNAC decoder."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._snac = None
        self._device: str = "cpu"
        self._dtype = torch.float32

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config.get("engines.svara.model_id", "kenpath/svara-tts-v1")
        snac_id = config.get("engines.svara.snac_model", "hubertsiuzdak/snac_24khz")
        device = config.get("engines.svara.device", "cuda")
        dtype_str = config.get("engines.svara.dtype", "bfloat16")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = "cpu"
            dtype_str = "float32"

        self._device = device
        self._dtype = getattr(torch, dtype_str, torch.bfloat16)

        logger.info("Loading svara model %s on %s (%s)", model_id, device, dtype_str)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self._model = self._model.to(device)
        self._model.eval()

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        logger.info("Loading SNAC decoder %s", snac_id)
        self._snac = SNAC.from_pretrained(snac_id).to(self._device)
        self._snac.eval()

        logger.info("Svara engine loaded — model: %s", model_id)

    async def unload(self) -> None:
        del self._model, self._tokenizer, self._snac
        self._model = self._tokenizer = self._snac = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="svara",
            display_name="Svara TTS (kenpath/svara-tts-v1)",
            supports_streaming=True,
            supports_cloning=False,
            languages=sorted(set(_LANGUAGE_NAMES.keys())),
            model_id=config.get("engines.svara.model_id", "kenpath/svara-tts-v1"),
            license="Apache-2.0",
        )

    def voices(self) -> list[EngineVoice]:
        result = [
            EngineVoice(
                id=voice_id,
                name=name,
                language="multi",
                description=(
                    f"Compatibility alias that maps to the requested language's "
                    f"{gender.lower()} Svara speaker"
                ),
            )
            for voice_id, name, gender in _DISPLAY_SPEAKERS
        ]

        for lang_code, language_name in sorted(_LANGUAGE_NAMES.items()):
            result.append(
                EngineVoice(
                    id=f"{language_name} (Female)",
                    name=f"{language_name} Female",
                    language=lang_code,
                    description="Direct Svara speaker id",
                )
            )
            result.append(
                EngineVoice(
                    id=f"{language_name} (Male)",
                    name=f"{language_name} Male",
                    language=lang_code,
                    description="Direct Svara speaker id",
                )
            )
        return result

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
        return SynthesisResult(audio=audio, sample_rate=_SNAC_SAMPLE_RATE)

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        language: str | None,
        speed: float,
        reference_audio: np.ndarray | None,
        reference_sr: int,
    ) -> np.ndarray:
        if reference_audio is not None:
            logger.warning(
                "Svara base model does not support reference-audio cloning; ignoring sample (sr=%s)",
                reference_sr,
            )

        input_ids, attention_mask, prompt = self._prepare_inputs(text, voice, language)

        max_tokens = config.get("engines.svara.max_tokens", 4096)
        temperature = config.get("engines.svara.temperature", 0.6)
        top_p = config.get("engines.svara.top_p", 0.95)
        repetition_penalty = config.get("engines.svara.repetition_penalty", 1.1)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                num_return_sequences=1,
                eos_token_id=_AUDIO_EOS_TOKEN_ID,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        code_list = self._parse_generated_codes(generated_ids)
        if not code_list:
            preview = self._preview_generated_tokens(generated_ids)
            logger.warning(
                "Svara generated no audio tokens for prompt=%r preview=%s",
                prompt[:160],
                preview,
            )
            raise RuntimeError("Svara generated no audio tokens")

        audio = self._decode_code_list(code_list)
        if speed != 1.0:
            audio = self._apply_speed(audio, speed)

        return audio

    def _prepare_inputs(
        self,
        text: str,
        voice: str,
        language: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Svara requires non-empty input text")

        speaker_id = self._resolve_speaker_id(voice, language, clean_text)

        style_match = _STYLE_TAG_RE.search(clean_text)
        style_tag = f"<{style_match.group(1).lower()}>" if style_match else ""
        text_without_tag = _STYLE_TAG_RE.sub("", clean_text).strip()
        tail = f" {style_tag}" if style_tag and style_tag != "<neutral>" else ""
        prompt = f"{speaker_id}: {text_without_tag}{tail}"

        tokenized = self._tokenizer(prompt, return_tensors="pt")
        start_token = torch.tensor([[_START_TOKEN_ID]], dtype=torch.int64)
        end_tokens = torch.tensor(
            [[_PROMPT_SEPARATOR_TOKEN_ID, _PROMPT_END_TOKEN_ID]],
            dtype=torch.int64,
        )
        input_ids = torch.cat([start_token, tokenized["input_ids"], end_tokens], dim=1)
        attention_mask = torch.ones_like(input_ids)

        return input_ids.to(self._device), attention_mask.to(self._device), prompt

    def _resolve_speaker_id(
        self,
        voice: str,
        language: str | None,
        text: str,
    ) -> str:
        normalized_voice = (voice or "default").strip()
        direct_match = _DIRECT_SPEAKER_RE.match(normalized_voice)
        if direct_match:
            resolved_language = direct_match.group("language").strip().title()
            resolved_gender = direct_match.group("gender").title()
            return f"{resolved_language} ({resolved_gender})"

        voice_key = normalized_voice.casefold()
        inferred_language = self._infer_language_code(language, voice_key, text)
        language_name = _LANGUAGE_NAMES.get(inferred_language, "English")

        if voice_key in _MALE_ALIASES:
            return f"{language_name} (Male)"
        if voice_key in _FEMALE_ALIASES:
            return f"{language_name} (Female)"

        logger.warning(
            "Unknown Svara voice '%s'; falling back to %s (Female)",
            normalized_voice,
            language_name,
        )
        return f"{language_name} (Female)"

    def _infer_language_code(
        self,
        language: str | None,
        voice_key: str,
        text: str,
    ) -> str:
        if language:
            lang = language.strip().lower()
            if lang in _LANGUAGE_NAMES:
                return lang
            if lang.split("-")[0] in _LANGUAGE_NAMES:
                return lang.split("-")[0]

        alias_language = _VOICE_ALIAS_TO_LANGUAGE.get(voice_key)
        if alias_language:
            return alias_language

        # Lightweight script detection so callers that omit ``language`` still
        # land on a sensible default when the text is clearly script-specific.
        for char in text:
            codepoint = ord(char)
            if 0x0B80 <= codepoint <= 0x0BFF:
                return "ta"
            if 0x0C00 <= codepoint <= 0x0C7F:
                return "te"
            if 0x0C80 <= codepoint <= 0x0CFF:
                return "kn"
            if 0x0D00 <= codepoint <= 0x0D7F:
                return "ml"
            if 0x0A80 <= codepoint <= 0x0AFF:
                return "gu"
            if 0x0A00 <= codepoint <= 0x0A7F:
                return "pa"
            if 0x0980 <= codepoint <= 0x09FF:
                return "bn"
            if 0x0900 <= codepoint <= 0x097F:
                return "hi"

        return "en"

    def _parse_generated_codes(self, generated_ids: torch.Tensor) -> list[int]:
        raw_id_codes = self._parse_raw_id_codes(generated_ids)
        if raw_id_codes:
            return raw_id_codes

        custom_token_codes = self._parse_custom_token_codes(generated_ids)
        if custom_token_codes:
            logger.info("Fell back to custom-token decoding path for Svara output")
            return custom_token_codes

        return []

    def _parse_raw_id_codes(self, generated_ids: torch.Tensor) -> list[int]:
        token_indices = (generated_ids == _HEAD_TOKEN_ID).nonzero(as_tuple=True)
        if len(token_indices[1]) == 0:
            return []

        cropped = generated_ids[:, token_indices[1][-1] + 1 :]
        row = cropped[0]
        row = row[row != _AUDIO_EOS_TOKEN_ID]
        if row.numel() == 0:
            return []

        trimmed = row[: (row.numel() // 7) * 7]
        if trimmed.numel() == 0:
            return []

        codes = [int(token.item()) - _AUDIO_TOKEN_OFFSET for token in trimmed]
        if any(code < 0 for code in codes[: min(14, len(codes))]):
            return []
        return codes

    def _parse_custom_token_codes(self, generated_ids: torch.Tensor) -> list[int]:
        token_ids = generated_ids[0].tolist()
        token_strings = self._tokenizer.convert_ids_to_tokens(token_ids)

        codes: list[int] = []
        audio_index = 0
        for token in token_strings:
            match = _CUSTOM_TOKEN_RE.fullmatch(token)
            if not match:
                continue

            code = int(match.group(1)) - 10 - ((audio_index % 7) * _CODEBOOK_SIZE)
            audio_index += 1
            if code >= 0:
                codes.append(code)

        return codes[: (len(codes) // 7) * 7]

    def _decode_code_list(self, code_list: list[int]) -> np.ndarray:
        layer_0: list[int] = []
        layer_1: list[int] = []
        layer_2: list[int] = []

        frame_count = len(code_list) // 7
        for frame_index in range(frame_count):
            base = frame_index * 7
            layer_0.append(code_list[base + 0])
            layer_1.append(code_list[base + 1] - (1 * _CODEBOOK_SIZE))
            layer_2.append(code_list[base + 2] - (2 * _CODEBOOK_SIZE))
            layer_2.append(code_list[base + 3] - (3 * _CODEBOOK_SIZE))
            layer_1.append(code_list[base + 4] - (4 * _CODEBOOK_SIZE))
            layer_2.append(code_list[base + 5] - (5 * _CODEBOOK_SIZE))
            layer_2.append(code_list[base + 6] - (6 * _CODEBOOK_SIZE))

        if not layer_0 or not self._codes_in_range(layer_0, layer_1, layer_2):
            raise RuntimeError("Svara produced invalid SNAC code ranges")

        codes = [
            torch.tensor(layer_0, device=self._device, dtype=torch.int32).unsqueeze(0),
            torch.tensor(layer_1, device=self._device, dtype=torch.int32).unsqueeze(0),
            torch.tensor(layer_2, device=self._device, dtype=torch.int32).unsqueeze(0),
        ]

        with torch.inference_mode():
            waveform = self._snac.decode(codes)

        audio = waveform.detach().squeeze().cpu().numpy().astype(np.float32)
        if audio.size == 0:
            raise RuntimeError("SNAC decode returned empty audio")

        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak
        return audio

    @staticmethod
    def _codes_in_range(*layers: list[int]) -> bool:
        for layer in layers:
            if any(code < 0 or code > _CODEBOOK_SIZE for code in layer):
                return False
        return True

    def _preview_generated_tokens(self, generated_ids: torch.Tensor) -> str:
        tail_ids = generated_ids[0].tolist()[-24:]
        try:
            tail_tokens = self._tokenizer.convert_ids_to_tokens(tail_ids)
        except Exception:
            tail_tokens = ["<unavailable>"]
        return f"tail_ids={tail_ids} tail_tokens={tail_tokens}"

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

        This uses full generation followed by PCM chunking for reliability.
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
