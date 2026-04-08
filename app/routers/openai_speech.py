"""OpenAI-compatible TTS API — POST /v1/audio/speech and GET /v1/models."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.models.openai_compat import ModelListResponse, SpeechModelInfo, SpeechRequest
from app.services.audio_converter import AudioConverter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@router.post("/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    """Generate audio from text (OpenAI-compatible).

    Supports streaming when ``stream=true`` is set in the request body.
    Streaming returns raw PCM int16 chunks in chunked transfer encoding.

    When streaming is not requested, returns the full audio in the requested
    format (mp3, wav, opus, flac, pcm).
    """
    registry = request.app.state.engine_registry
    converter: AudioConverter = request.app.state.audio_converter
    voice_store = request.app.state.voice_store

    engine = registry.get(body.model)
    if engine is None:
        raise HTTPException(status_code=400, detail=f"No TTS engine available (requested: {body.model})")

    # Check for voice cloning reference
    reference_audio = None
    reference_sr = 24000
    sample = voice_store.get(body.voice)
    if sample:
        reference_audio, reference_sr = sample.load_audio()

    if body.stream:
        # Streaming response — raw PCM int16 chunks
        info = engine.info()
        if not info.supports_streaming:
            logger.info("Engine '%s' does not support streaming, using chunked fallback", info.name)

        async def _stream():
            async for chunk in engine.synthesize_stream(
                body.input,
                voice=body.voice,
                language=body.language,
                speed=body.speed,
                reference_audio=reference_audio,
                reference_sr=reference_sr,
            ):
                yield chunk

        return StreamingResponse(
            _stream(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(engine.voices()[0].sample_rate if engine.voices() else 24000),
                "X-Sample-Width": "16",
                "X-Channels": "1",
            },
        )

    # Non-streaming — full synthesis then format conversion
    result = await engine.synthesize(
        body.input,
        voice=body.voice,
        language=body.language,
        speed=body.speed,
        reference_audio=reference_audio,
        reference_sr=reference_sr,
    )

    audio_bytes = converter.convert(result.audio, result.sample_rate, body.response_format)
    content_type = converter.content_type(body.response_format)

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="speech.{body.response_format}"'},
    )


@router.get("/models")
async def list_models(request: Request):
    """List available TTS models (OpenAI-compatible)."""
    registry = request.app.state.engine_registry
    models = []
    for name, engine in registry.available_engines().items():
        info = engine.info()
        models.append(SpeechModelInfo(id=name, owned_by=info.license))
    return ModelListResponse(data=models)
