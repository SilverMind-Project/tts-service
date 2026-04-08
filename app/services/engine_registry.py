"""Engine registry — manages loading, switching, and listing TTS engines."""

from __future__ import annotations

import logging

from app import config
from app.services.engine_base import TTSEngine

logger = logging.getLogger(__name__)

# Lazy imports to avoid pulling in heavy deps for unused engines
_ENGINE_CLASSES: dict[str, str] = {
    "svara": "app.services.engine_svara:SvaraEngine",
    "parler": "app.services.engine_parler:ParlerEngine",
    "fish_speech": "app.services.engine_fish_speech:FishSpeechEngine",
    "seamless": "app.services.engine_seamless:SeamlessEngine",
    "edge_tts": "app.services.engine_edge_tts:EdgeTTSEngine",
}


def _import_engine(dotted_path: str) -> type[TTSEngine]:
    module_path, class_name = dotted_path.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class EngineRegistry:
    """Loads and manages TTS engine instances."""

    def __init__(self) -> None:
        self._engines: dict[str, TTSEngine] = {}
        self._default: str = ""

    @property
    def default_engine_name(self) -> str:
        return self._default

    async def load_engines(self) -> None:
        """Load engines listed in config ``engines.enabled``."""
        enabled: list[str] = config.get("engines.enabled", ["svara"])
        default = config.get("engines.default", enabled[0] if enabled else "svara")

        for name in enabled:
            if name not in _ENGINE_CLASSES:
                logger.error("Unknown engine '%s' — skipping", name)
                continue
            try:
                cls = _import_engine(_ENGINE_CLASSES[name])
                engine = cls()
                await engine.load()
                self._engines[name] = engine
                logger.info("Engine '%s' loaded", name)
            except Exception:
                logger.exception("Failed to load engine '%s'", name)

        if default in self._engines:
            self._default = default
        elif self._engines:
            self._default = next(iter(self._engines))
            logger.warning(
                "Default engine '%s' not available, using '%s'", default, self._default
            )
        else:
            logger.error("No TTS engines loaded!")

    async def unload_engines(self) -> None:
        for name, engine in self._engines.items():
            try:
                await engine.unload()
                logger.info("Engine '%s' unloaded", name)
            except Exception:
                logger.exception("Error unloading engine '%s'", name)
        self._engines.clear()

    def get(self, name: str | None = None) -> TTSEngine | None:
        """Get an engine by name, or the default engine."""
        if name and name in self._engines:
            return self._engines[name]
        if name and name not in self._engines:
            logger.warning("Engine '%s' not loaded, falling back to default", name)
        return self._engines.get(self._default)

    def available_engines(self) -> dict[str, TTSEngine]:
        return dict(self._engines)
