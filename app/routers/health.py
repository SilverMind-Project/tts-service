"""Health check endpoint."""

from __future__ import annotations

import torch
from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    registry = request.app.state.engine_registry
    engines = {}
    for name, engine in registry.available_engines().items():
        info = engine.info()
        engines[name] = {
            "display_name": info.display_name,
            "languages": info.languages,
            "supports_streaming": info.supports_streaming,
            "supports_cloning": info.supports_cloning,
            "license": info.license,
        }

    voice_store = request.app.state.voice_store
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "default_engine": registry.default_engine_name,
        "engines": engines,
        "voice_samples": len(voice_store.list_samples()),
    }
