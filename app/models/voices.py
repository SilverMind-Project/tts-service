"""Voice management models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceInfo(BaseModel):
    """Information about an available voice."""

    id: str
    name: str
    engine: str = Field(description="TTS engine this voice belongs to")
    language: str
    description: str = ""
    supports_cloning: bool = False
    sample_rate: int = 24000
    is_default: bool = False


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]
    total: int


class VoiceSampleUpload(BaseModel):
    """Metadata for uploading a voice reference sample."""

    voice_id: str = Field(description="Identifier for this custom voice")
    name: str = Field(description="Display name")
    language: str = Field(default="en", description="Primary language of the sample")
    description: str = ""
