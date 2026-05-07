# AGENTS.md - TTS Service

## Project Overview

Self-hosted text-to-speech microservice for the Cognitive Companion platform. Provides an OpenAI-compatible API for speech synthesis using GPU-accelerated Indian language TTS models. Integrates with Home Assistant via the Wyoming protocol through a wyoming_openai sidecar.

## Architecture

```
tts-service/
+-- app/
|   +-- main.py                        # FastAPI app with lifespan (engine init)
|   +-- config.py                      # YAML config with ${ENV_VAR} interpolation
|   +-- models/                        # Pydantic request/response models
|   |   +-- openai_compat.py           # POST /v1/audio/speech models
|   |   +-- voices.py                  # Voice management models
|   +-- routers/                       # FastAPI route handlers
|   |   +-- health.py                  # GET /health
|   |   +-- openai_speech.py           # POST /v1/audio/speech, GET /v1/models
|   |   +-- voices.py                  # GET/POST/DELETE /api/v1/voices
|   +-- services/                      # Business logic
|       +-- engine_base.py             # TTSEngine ABC + SynthesisResult
|       +-- engine_svara.py            # Svara TTS (3B, Indian langs, token-level streaming)
|       +-- engine_parler.py           # Indic Parler TTS (938M, emotion)
|       +-- engine_fish_speech.py      # Fish Speech S2-Pro (80+ langs, cloning)
|       +-- engine_seamless.py         # SeamlessM4T v2 (36 langs, Meta)
|       +-- engine_edge_tts.py         # OpenAI Edge TTS (remote, pass-through)
|       +-- engine_registry.py         # Engine loading + selection
|       +-- audio_converter.py         # WAV/MP3/Opus/FLAC/PCM conversion
|       +-- voice_store.py             # Voice cloning sample management
+-- tests/
|   +-- conftest.py                    # MockEngine fixture, app_client, audio_samples
|   +-- test_audio_converter.py        # Format conversion tests
|   +-- test_engine_base.py            # ABC default streaming tests
|   +-- test_engine_registry.py        # Registry load/get/unload tests
|   +-- test_openai_speech.py          # API integration tests
|   +-- test_voice_store.py            # Voice sample management tests
+-- config/
|   +-- settings.yaml                  # All runtime configuration
+-- data/
|   +-- voice_samples/                 # Reference audio for voice cloning
+-- kubernetes/                        # K8s manifests (migrated to ../kubernetes/tts-service/)
+-- Dockerfile                         # NVIDIA CUDA + Python 3.12
+-- docker-compose.yml                 # Single service with GPU
+-- pyproject.toml                     # Dependencies + pytest config
```

## Key Patterns

### Engine Abstraction

All TTS engines implement `TTSEngine` (in `engine_base.py`):

```python
class TTSEngine(ABC):
    async def load() -> None          # Load model on startup
    async def unload() -> None        # Release resources
    def info() -> EngineInfo          # Engine metadata
    def voices() -> list[EngineVoice] # Available voices
    async def synthesize(text, voice, language, speed, ...) -> SynthesisResult
    async def synthesize_stream(text, ...) -> AsyncIterator[bytes]  # Streaming
```

### Streaming Architecture

Two streaming strategies are implemented:

**Token-level streaming (Svara)**: A custom `_TokenQueueStreamer` feeds generated tokens into a `queue.Queue`. The async `synthesize_stream()` generator consumes tokens, accumulates 7-token SNAC frames into batches of `stream_frame_buffer` frames (default 21 = ~210ms), and decodes each batch together through the SNAC decoder. Batching is essential because SNAC's convolutional decoder needs temporal context from neighboring frames for clean audio. The decoded PCM is clipped to [-1, 1] before int16 conversion to prevent overflow.

**Proxied streaming (Edge TTS)**: Uses `httpx.stream()` to proxy PCM chunks from the remote openai-edge-tts service. No local decoding needed.

**Chunked fallback (base class)**: Engines without native streaming (parler, fish_speech, seamless) use the default `synthesize_stream()` implementation that runs full synthesis then chunks the result into PCM segments.

### Adding a New Engine

1. Create `app/services/engine_newmodel.py` implementing `TTSEngine`
2. Register in `engine_registry.py`'s `_ENGINE_CLASSES` dict
3. Add config section in `config/settings.yaml` under `engines.newmodel`
4. Add optional dependency group in `pyproject.toml`

### Home Assistant Integration (Wyoming)

Home Assistant uses the **Wyoming protocol** for voice pipelines. A [wyoming_openai](https://github.com/roryeckel/wyoming_openai) sidecar bridges the OpenAI-compatible API to Wyoming:

```text
Home Assistant (Wyoming) <--> wyoming-openai (:10300) <--> tts-service (:8600) /v1/audio/speech
```

Start the sidecar with `docker compose --profile wyoming up -d`.

### Audio Pipeline

```
Text input
  -> Engine selection (by model param or voice key)
  -> Voice sample lookup (for cloning)
  -> TTSEngine.synthesize() -> float32 numpy array
  -> AudioConverter.convert() -> mp3/wav/opus/flac/pcm bytes
  -> HTTP response
```

## Configuration

All config lives in `config/settings.yaml` with `${ENV_VAR:default}` interpolation. Access via `config.get("engines.svara.device")`.

Key settings:

| Path | Description |
|------|-------------|
| `engines.enabled` | List of engines to load at startup |
| `engines.default` | Default engine for requests without model param |
| `engines.svara.device` | GPU device (cuda, cuda:0, cpu) |
| `engines.svara.dtype` | Model precision (bfloat16, float16, float32) |
| `engines.svara.stream_frame_buffer` | SNAC frames per decode batch during streaming (default 21 = ~210ms) |
| `storage.voice_samples_dir` | Path to voice cloning samples |

## Port

This service runs on port **8600** (vs person-id on 8600, cognitive-companion on 8000).

## Engines

| Engine | Module | Model | Key Traits |
|--------|--------|-------|------------|
| `svara` | `engine_svara.py` | kenpath/svara-tts-v1 | 3B params, SNAC codec, Indian langs, token-level streaming |
| `parler` | `engine_parler.py` | ai4bharat/indic-parler-tts | 938M params, text-prompt voice control |
| `fish_speech` | `engine_fish_speech.py` | fishaudio/s2-pro | Dual-AR architecture, 80+ langs, voice cloning |
| `seamless` | `engine_seamless.py` | facebook/seamless-m4t-v2-large | SeamlessM4T v2, 36 speech output langs, CC-BY-NC-4.0 |
| `edge_tts` | `engine_edge_tts.py` | travisvn/openai-edge-tts | Pass-through to remote, Microsoft Edge voices, true streaming |

### Fish Speech Notes

- Requires `fish-speech` installed from GitHub (`pip install git+https://github.com/fishaudio/fish-speech.git`)
- Uses a dual autoregressive architecture with VQGAN/Firefly decoder
- Voice cloning via reference audio encoding through the VQGAN encoder
- Outputs at 44.1kHz (other engines output at 16-24kHz)

### SeamlessM4T Notes

- Uses standard `transformers` library (`SeamlessM4Tv2Model`, `AutoProcessor`)
- Text-to-speech via `model.generate()` with `tgt_lang` parameter
- Language codes use 3-letter ISO format internally (e.g., `eng`, `tam`, `hin`)
- Outputs at 16kHz

### OpenAI Edge TTS Notes

- Pass-through engine: no local model loading, proxies to a remote openai-edge-tts service
- Requires `httpx` for async HTTP requests
- Configurable endpoint via `engines.edge_tts.base_url` or `EDGE_TTS_URL` env var
- Supports true streaming (proxied PCM chunks from remote service)
- Uses Microsoft Edge TTS voices: 40+ languages including Indian languages

## Testing

Tests are in `tests/` using pytest + pytest-asyncio.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

**Test structure:**

| File | Coverage |
|------|----------|
| `conftest.py` | MockEngine fixture (440 Hz sine wave), app_client with ASGITransport, audio_samples |
| `test_audio_converter.py` | WAV/PCM/FLAC conversion, round-trip accuracy, content-type mapping |
| `test_engine_base.py` | Default `synthesize_stream()` chunking behavior |
| `test_engine_registry.py` | Engine loading, get/fallback, unload |
| `test_openai_speech.py` | Full API integration: non-streaming, streaming, model listing |
| `test_voice_store.py` | Sample discovery, save, delete, error cases |

**MockEngine**: A deterministic engine that returns a 440 Hz sine wave. Used in API integration tests via `app_client` fixture that wires it into the FastAPI app with mocked services.

## Dependencies

- **GPU**: NVIDIA CUDA 12.4+ for svara/parler/fish_speech/seamless engines
- **ffmpeg**: Required for MP3/Opus encoding (installed in Docker image)
- **HuggingFace models**: Downloaded on first run to `$HF_HOME` (default: `data/hf_cache`)
- **fish-speech**: Installed from GitHub (not on PyPI). Only needed if `fish_speech` engine is enabled

## Integration

Cognitive Companion's `backend/integrations/tts.py` (`TTSClient`) uses the OpenAI-compatible endpoint. Point `tts.url` in cognitive-companion's settings to this service. The `TTSClient` supports both batch (`generate_audio()`) and streaming (`stream_audio()`) modes.

Home Assistant integrates via the Wyoming protocol through a wyoming_openai sidecar (see docker-compose.yml, `wyoming` profile).
