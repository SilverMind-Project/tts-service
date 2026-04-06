"""Voice management endpoints — list, upload, and delete voice samples."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.models.voices import VoiceInfo, VoiceListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["voices"])


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(request: Request, engine: str | None = None):
    """List all available voices across all engines, plus custom voice samples."""
    registry = request.app.state.engine_registry
    voice_store = request.app.state.voice_store

    voices: list[VoiceInfo] = []

    # Engine built-in voices
    for eng_name, eng in registry.available_engines().items():
        if engine and eng_name != engine:
            continue
        info = eng.info()
        for v in eng.voices():
            voices.append(VoiceInfo(
                id=v.id,
                name=v.name,
                engine=eng_name,
                language=v.language,
                description=v.description,
                supports_cloning=info.supports_cloning,
                sample_rate=v.sample_rate,
                is_default=(eng_name == registry.default_engine_name and v.id in ("default", "speaker_0")),
            ))

    # Custom voice samples (for engines that support cloning)
    for sample in voice_store.list_samples():
        voices.append(VoiceInfo(
            id=sample.voice_id,
            name=sample.name,
            engine="custom",
            language=sample.language,
            description=sample.description,
            supports_cloning=True,
            sample_rate=sample.sample_rate,
        ))

    return VoiceListResponse(voices=voices, total=len(voices))


@router.post("/voices/upload")
async def upload_voice_sample(
    request: Request,
    voice_id: str = Form(...),
    name: str = Form(...),
    language: str = Form("en"),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    """Upload a voice reference sample for voice cloning.

    The audio file should be a 10-30 second WAV recording of the target
    voice speaking clearly. Supported formats: WAV, MP3, FLAC.
    """
    voice_store = request.app.state.voice_store

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    audio_bytes = await file.read()
    if len(audio_bytes) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

    ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "wav"
    filename = f"reference.{ext}"

    sample = voice_store.save_sample(
        voice_id=voice_id,
        name=name,
        language=language,
        description=description,
        audio_bytes=audio_bytes,
        filename=filename,
    )

    return {
        "voice_id": sample.voice_id,
        "name": sample.name,
        "language": sample.language,
        "status": "uploaded",
    }


@router.delete("/voices/{voice_id}")
async def delete_voice_sample(voice_id: str, request: Request):
    """Delete a custom voice sample."""
    voice_store = request.app.state.voice_store
    deleted = voice_store.delete_sample(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice sample '{voice_id}' not found")
    return {"deleted": True, "voice_id": voice_id}
