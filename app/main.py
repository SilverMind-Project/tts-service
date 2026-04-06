"""TTS Service — Self-hosted text-to-speech with an OpenAI-compatible API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routers import health, openai_speech, voices
from app.services.audio_converter import AudioConverter
from app.services.engine_registry import EngineRegistry
from app.services.voice_store import VoiceStore

logger = logging.getLogger("tts-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_level = config.get("logging.level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    logger.info("Initializing TTS service …")

    # Voice store (manages reference audio samples)
    voice_store = VoiceStore(
        samples_dir=config.get("storage.voice_samples_dir", "data/voice_samples"),
    )
    voice_store.load()
    app.state.voice_store = voice_store

    # Audio format converter
    app.state.audio_converter = AudioConverter()

    # Engine registry — loads the configured TTS engine(s)
    registry = EngineRegistry()
    await registry.load_engines()
    app.state.engine_registry = registry

    logger.info(
        "TTS service ready — engines: %s, default: %s",
        list(registry.available_engines()),
        registry.default_engine_name,
    )
    yield

    # Cleanup
    await registry.unload_engines()
    logger.info("TTS service shut down.")


app = FastAPI(
    title=config.get("app.name", "TTS Service"),
    version=config.get("app.version", "1.0.0"),
    description="Self-hosted TTS with an OpenAI-compatible API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(openai_speech.router)
app.include_router(voices.router)
