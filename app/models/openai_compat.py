"""OpenAI-compatible TTS API models (POST /v1/audio/speech)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """Mirrors the OpenAI ``POST /v1/audio/speech`` request body."""

    model: str = Field(
        default="svara",
        description="TTS engine to use: svara, parler, fish_speech, seamless, edge_tts",
    )
    input: str = Field(
        ...,
        description="Text to synthesise.",
        max_length=4096,
    )
    voice: str = Field(
        default="default",
        description=(
            "Voice identifier. For svara: speaker tag (e.g. 'speaker_0'). "
            "For parler: voice description or preset name. "
            "For fish_speech: 'default' or custom voice sample ID. "
            "For edge_tts: Edge TTS voice name (e.g. 'en-IN-NeerjaExpressiveNeural')."
        ),
    )
    response_format: str = Field(
        default="mp3",
        description="Audio format: mp3, wav, opus, flac, pcm",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier.",
    )
    language: str | None = Field(
        default=None,
        description="Language code override (e.g. 'en', 'ta'). Auto-detected if omitted.",
    )
    stream: bool = Field(
        default=False,
        description="If true, stream audio chunks as they are generated.",
    )


class SpeechModelInfo(BaseModel):
    """Returned by GET /v1/models for TTS engines."""

    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[SpeechModelInfo]
