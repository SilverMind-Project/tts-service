# CLAUDE.md

Quick reference for Claude Code agents in `tts-service/`. The full reference is [AGENTS.md](AGENTS.md); this file is the orientation pointer plus the few invariants you must hold from the first edit.

---

## What this is

Self-hosted text-to-speech microservice for the Cognitive Companion platform. FastAPI backend, multi-engine plugin architecture (5 engines), OpenAI-compatible API (`/v1/audio/speech`). GPU-accelerated with token-level streaming on the Svara engine. Integrates with Home Assistant via a Wyoming protocol sidecar.

---

## Read before editing

1. [AGENTS.md](AGENTS.md): canonical reference (architecture, engine contract, streaming architecture, testing conventions).
2. `app/main.py` lifespan: source of truth for service wiring and `app.state` keys.
3. `app/services/engine_base.py`: `TTSEngine` ABC, `SynthesisResult`, `EngineInfo`, `EngineVoice`.
4. `config/settings.yaml`: every tunable, engine config, server port.

---

## Commands

```bash
# Development
uvicorn app.main:app --host 0.0.0.0 --port 8600 --reload

# Tests
pytest tests/ -v

# Lint
ruff check app/ tests/
```

---

## Non-negotiable invariants

- **Engines implement `TTSEngine`.** Every engine subclasses the ABC in `engine_base.py`. Five methods required: `load`, `unload`, `info`, `voices`, `synthesize`. `synthesize_stream` has a default chunked fallback; override for true streaming.
- **Services live in the lifespan, not in routers.** Use `request.app.state.<name>`. Never instantiate inside a router.
- **Heavy model imports are lazy.** Engine modules are imported only when `EngineRegistry.load_engines()` runs at startup, not at module level. The registry maps engine names to dotted import paths.
- **Lazy imports for optional deps.** Engines that need non-standard packages (fish-speech from GitHub, parler-tts, snac) import them inside their `load()` method, not at module level. This keeps the service importable even when optional engines aren't installed.
- **All audio is float32 numpy.** Engines output `SynthesisResult(audio=np.ndarray[float32], sample_rate=int)`. `AudioConverter` handles format conversion at the API boundary.
- **Streaming returns raw PCM int16.** `synthesize_stream()` yields `bytes` chunks of 16-bit signed little-endian PCM. The HTTP layer wraps this in `StreamingResponse` with `X-Sample-Rate`, `X-Sample-Width`, `X-Channels` headers.
- **Config via YAML with `${ENV_VAR}` interpolation.** `app/config.py` is the singleton. Access with `config.get("engines.svara.device")`.
- **Port 8600.** This service binds to 8600. The person-id service binds to 8200. Cognitive Companion binds to 8000.

---

## Engine contract (the 30-second version)

```python
from app.services.engine_base import TTSEngine, EngineInfo, EngineVoice, SynthesisResult

class YourEngine(TTSEngine):
    async def load(self) -> None: ...
    async def unload(self) -> None: ...
    def info(self) -> EngineInfo: ...
    def voices(self) -> list[EngineVoice]: ...
    async def synthesize(self, text, *, voice, language, speed,
                         reference_audio, reference_sr) -> SynthesisResult: ...
    # synthesize_stream has a default chunked fallback; override for true streaming
```

---

## Engines (5)

| Name | File | Model | Key traits |
| --- | --- | --- | --- |
| `svara` | `engine_svara.py` | kenpath/svara-tts-v1 | 3B params, SNAC codec, Indian languages, token-level streaming |
| `parler` | `engine_parler.py` | ai4bharat/indic-parler-tts | 938M params, text-prompt voice control, emotion support |
| `fish_speech` | `engine_fish_speech.py` | fishaudio/s2-pro | Dual-AR, 80+ languages, voice cloning via VQGAN encoder |
| `seamless` | `engine_seamless.py` | facebook/seamless-m4t-v2-large | 36 languages, CC-BY-NC-4.0 license |
| `edge_tts` | `engine_edge_tts.py` | travisvn/openai-edge-tts | Pass-through to remote, Microsoft Edge voices, no GPU needed |

---

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | GPU status, loaded engines, voice sample count |
| `POST` | `/v1/audio/speech` | Generate speech (OpenAI-compatible), supports streaming |
| `GET` | `/v1/models` | List loaded engines |
| `GET` | `/api/v1/voices` | List all voices (built-in + custom samples) |
| `POST` | `/api/v1/voices/upload` | Upload voice cloning reference sample |
| `DELETE` | `/api/v1/voices/{voice_id}` | Delete a custom voice sample |

---

## Correctness expectations for every change

1. Tests pass: `pytest tests/ -v`. Tests use `MockEngine` (440 Hz sine wave) via `app_client` fixture with `ASGITransport`.
2. New engines: success path, missing-dependency path, and at least one edge case test.
3. Strongly-typed public surfaces. `@dataclass` for results, Pydantic for HTTP wire models.
4. Lazy imports for optional engine dependencies (fish-speech, parler-tts, snac).
5. No `print()`. Use stdlib `logging.getLogger(__name__)`.

---

## What NOT to do

- Hardcode model IDs or paths. Use `config.get()`.
- Import heavy ML packages at module level in engine files. Import inside `load()`.
- Block the event loop. Run model inference in `loop.run_in_executor(None, ...)`.
- Return raw tensors across the engine boundary. Convert to `np.ndarray[float32]`.
- Add a dependency without updating `pyproject.toml` optional-dependencies.
