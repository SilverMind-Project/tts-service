# AGENTS.md - TTS Service

Reference for AI coding agents working in `tts-service/`. This document is the canonical, deep guide. `CLAUDE.md` is a tight pointer aimed at the same audience; `README.md` is human-facing.

If a fact appears here, it traces to a file in this tree at the time of writing. Verify before relying on it: `git log` is authoritative for "what changed", and `grep` against `app/` is authoritative for "what exists".

---

## 1. Mission and scope

The TTS Service is a self-hosted, GPU-accelerated text-to-speech microservice for the Cognitive Companion platform. It exposes an OpenAI-compatible API (`POST /v1/audio/speech`) so any OpenAI TTS client can use it as a drop-in replacement. It also serves as the TTS backend for Home Assistant via a Wyoming protocol sidecar.

Three things make this codebase non-trivial:

1. **Multi-engine plugin architecture.** Five engines (Svara, Parler, Fish Speech, SeamlessM4T, Edge TTS) each implement the same `TTSEngine` ABC. Adding a new engine is a single file with zero changes to the API layer.
2. **Token-level streaming.** The Svara engine intercepts tokens during `model.generate()` via a custom streamer, accumulates SNAC frames into configurable batches, and decodes them incrementally. This gives time-to-first-audio of ~210ms instead of waiting for full generation.
3. **Voice cloning infrastructure.** The `VoiceStore` manages reference audio samples on disk with a structured directory layout. Fish Speech encodes these samples through its VQGAN encoder to produce voice-cloned speech.

---

## 2. Tech stack

| Layer | Choice |
| --- | --- |
| Backend | Python 3.11+, FastAPI, Pydantic 2 |
| ML | PyTorch 2.2+, Transformers 4.44+, soundfile, pydub |
| Audio | numpy (float32 arrays), SNAC neural codec (Svara), VQGAN (Fish Speech) |
| HTTP client | httpx (Edge TTS engine) |
| Container | Docker with nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 |
| Config | YAML with `${ENV_VAR}` interpolation |
| Tests | pytest + pytest-asyncio, httpx ASGITransport for API tests |
| Lint | ruff (E, F, I, W) |

---

## 3. Repository layout

```text
tts-service/
├── app/
│   ├── main.py                       FastAPI app factory, lifespan (wiring source-of-truth)
│   ├── config.py                     YAML config with ${ENV_VAR} interpolation
│   ├── models/
│   │   ├── openai_compat.py          SpeechRequest, SpeechModelInfo, ModelListResponse
│   │   └── voices.py                 VoiceInfo, VoiceListResponse, VoiceSampleUpload
│   ├── routers/
│   │   ├── health.py                 GET /health
│   │   ├── openai_speech.py          POST /v1/audio/speech, GET /v1/models
│   │   └── voices.py                 GET/POST/DELETE /api/v1/voices
│   └── services/
│       ├── engine_base.py            TTSEngine ABC, SynthesisResult, EngineInfo, EngineVoice
│       ├── engine_registry.py        Engine loading, selection, fallback
│       ├── engine_svara.py           Svara TTS (3B, SNAC codec, token-level streaming)
│       ├── engine_parler.py          Indic Parler TTS (938M, text-prompt voice control)
│       ├── engine_fish_speech.py     Fish Speech S2-Pro (80+ langs, voice cloning)
│       ├── engine_seamless.py        SeamlessM4T v2 (36 langs, Meta)
│       ├── engine_edge_tts.py        OpenAI Edge TTS pass-through (remote, no GPU)
│       ├── audio_converter.py        WAV, MP3, Opus, FLAC, PCM conversion
│       └── voice_store.py            Voice cloning sample management on disk
├── config/
│   └── settings.yaml                 All runtime configuration
├── data/
│   └── voice_samples/                Reference audio for voice cloning
├── tests/
│   ├── conftest.py                   MockEngine, app_client fixture, audio_samples
│   ├── test_engine_base.py           ABC default streaming behavior
│   ├── test_engine_registry.py       Registry load, get, unload
│   ├── test_audio_converter.py       Format conversion, round-trip accuracy
│   ├── test_openai_speech.py         Full API integration tests
│   └── test_voice_store.py           Voice sample CRUD
├── Dockerfile                        Multi-stage: CUDA 13.0 + Python 3.12
├── docker-compose.yml                GPU service + optional wyoming-openai sidecar
├── pyproject.toml                    Dependencies with optional engine extras
├── CLAUDE.md                         Agent quick-reference
└── README.md                         Human-facing
```

---

## 4. Architecture

### 4.1 Backend layering

```text
app/models/          Pydantic wire models
app/routers/         FastAPI route handlers (thin: validate, dispatch, convert)
app/services/        Business logic (engines, registry, audio conversion, voice store)
app/main.py          App factory + lifespan (wiring source-of-truth)
app/config.py        Configuration singleton
```

**Key principle.** Routers are thin. They extract the engine from the registry, call `synthesize()` or `synthesize_stream()`, convert the output format, and return an HTTP response. No ML logic lives in routers.

### 4.2 Service injection

Services are constructed in the FastAPI lifespan (`app/main.py`) and stashed on `app.state`:

```python
app.state.voice_store = voice_store
app.state.audio_converter = audio_converter
app.state.engine_registry = registry
```

Routers read them through `request.app.state.<name>`. Never instantiate a service or engine inside a router.

### 4.3 Engine lifecycle

1. **Startup.** `EngineRegistry.load_engines()` reads `engines.enabled` from config. For each engine name, it lazy-imports the engine class, instantiates it, and calls `engine.load()`.
2. **Runtime.** Routers call `registry.get(model)` to resolve an engine by name (or get the default). The `SpeechRequest.model` field maps directly to the engine name.
3. **Shutdown.** The lifespan calls `registry.unload_engines()`, which calls `engine.unload()` on each engine and clears GPU memory.

### 4.4 Audio pipeline

```text
Text input
  -> Router extracts: model, voice, language, speed, stream flag, reference_audio
  -> VoiceStore.get(voice) -> reference audio sample (for cloning engines)
  -> EngineRegistry.get(model) -> TTSEngine
  -> engine.synthesize(text, voice=voice, language=lang, speed=speed,
                       reference_audio=ref_audio, reference_sr=ref_sr)
     -> float32 numpy array (mono)
  -> AudioConverter.convert(audio, sample_rate, output_format)
     -> bytes (mp3, wav, opus, flac, pcm)
  -> HTTP Response
```

For streaming: `synthesize_stream()` yields raw PCM int16 bytes. The router wraps them in `StreamingResponse(media_type="audio/pcm")`.

### 4.5 Voice resolution

When a request specifies a `voice` parameter:

1. `VoiceStore.get(voice)` checks for a custom voice sample with that ID.
2. If found, the reference audio is loaded and passed as `reference_audio` to the engine.
3. If not found, the voice string is passed directly to the engine (engine-specific interpretation: Svara speaker tags, Parler preset names, Edge TTS voice names, etc.).

This means voice cloning is transparent: the same `voice` parameter selects a custom cloned voice when a sample exists, and falls back to a built-in engine voice otherwise.

### 4.6 Streaming architecture

Three distinct streaming strategies exist in the codebase:

**Token-level streaming (Svara).** A custom `_TokenQueueStreamer` feeds generated token IDs into a `queue.Queue` during `model.generate()`. The async `synthesize_stream()` generator consumes tokens from the queue, accumulates 7-token SNAC frames into batches of `stream_frame_buffer` frames (default 21, ~210ms), and decodes each batch through the SNAC decoder. Batching is essential because SNAC's convolutional decoder needs temporal context from neighboring frames to produce clean audio. The decoded PCM is clipped to [-1, 1] before int16 conversion to prevent overflow.

Generation runs in a background thread (`loop.run_in_executor`). The async generator awaits each token from the queue without blocking the event loop. A `finally` block ensures the generation thread completes even if the client disconnects.

**Proxied streaming (Edge TTS).** Uses `httpx.AsyncClient.stream()` to open a streaming POST to the remote openai-edge-tts service. PCM chunks are yielded directly as they arrive. No local decoding.

**Chunked fallback (base class).** Engines without native streaming (Parler, Fish Speech, SeamlessM4T) use the default `synthesize_stream()` in `engine_base.py`. It runs full `synthesize()`, converts to PCM int16, and yields fixed-size chunks.

---

## 5. Engine contract

Every engine must implement `TTSEngine` (in `engine_base.py`):

```python
class TTSEngine(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    def info(self) -> EngineInfo: ...

    @abstractmethod
    def voices(self) -> list[EngineVoice]: ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        language: str | None = None,
        speed: float = 1.0,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
    ) -> SynthesisResult: ...

    async def synthesize_stream(
        self, text, *, voice, language, speed,
        reference_audio, reference_sr, chunk_size=4096,
    ) -> AsyncIterator[bytes]: ...
```

### Data types

| Type | Fields | Purpose |
| --- | --- | --- |
| `SynthesisResult` | `audio: np.ndarray[float32]`, `sample_rate: int` | Engine output |
| `EngineInfo` | `name`, `display_name`, `supports_streaming`, `supports_cloning`, `languages: list[str]`, `model_id`, `license` | Engine metadata |
| `EngineVoice` | `id`, `name`, `language`, `description`, `supports_cloning`, `sample_rate` | Voice descriptor |

### Adding a new engine

1. Create `app/services/engine_newmodel.py` with a class that subclasses `TTSEngine`.
2. Register it in `engine_registry.py` in the `_ENGINE_CLASSES` dict: `"newmodel": "app.services.engine_newmodel:NewModelEngine"`.
3. Add a config section in `config/settings.yaml` under `engines.newmodel`.
4. Add an optional dependency group in `pyproject.toml` (`[project.optional-dependencies]`).
5. Add tests under `tests/` with a mock that generates a deterministic waveform.

No changes needed in routers, models, or `main.py`. The registry auto-discovers the engine at startup.

---

## 6. Engines in detail

### 6.1 Svara (`engine_svara.py`)

| Attribute | Value |
| --- | --- |
| Model | kenpath/svara-tts-v1 (3B params) |
| Codec | SNAC (7-layer, 24kHz) via hubertsiuzdak/snac_24khz |
| Languages | 22 Indian languages + English |
| Streaming | Token-level (true streaming via custom streamer) |
| Voice control | Speaker tags: `speaker_0` (female), `speaker_1` (male), or direct `"Language (Gender)"` format |
| License | Apache-2.0 |

**Architecture.** An Orpheus-style discrete-audio-token model. The prompt is formatted as `"Language (Gender): text <style>"` with sentinel token IDs wrapped around the tokenized prompt. The model generates interleaved 7-way SNAC codes, which are unrolled into 7 parallel codec layers and decoded through the SNAC decoder.

**Language detection.** When no `language` parameter is provided, the engine performs lightweight script detection on the input text using Unicode codepoint ranges (Tamil: 0x0B80-0x0BFF, Hindi: 0x0900-0x097F, etc.). Falls back to English.

**Style tags.** The engine recognizes `<style>` tags embedded in the input text (e.g., `<clear>`, `<narrative>`, `<happy>`). Tags are extracted via regex and appended to the prompt.

### 6.2 Parler (`engine_parler.py`)

| Attribute | Value |
| --- | --- |
| Model | ai4bharat/indic-parler-tts (938M params) |
| Languages | 11 Indian languages |
| Streaming | Chunked fallback only |
| Voice control | Text descriptions (e.g., `"A calm female voice with a clear Tamil accent"`) |
| License | Apache-2.0 |

**Architecture.** Uses the `parler-tts` library (`ParlerTTSForConditionalGeneration`). Takes two inputs: a voice description prompt and the text to synthesise. The description prompt controls speaker characteristics, emotion, and pace. Five built-in presets are provided: `female_calm`, `male_clear`, `female_tamil`, `male_tamil`, `female_elderly_friendly`.

When the `voice` parameter matches a preset, the stored description is used. Otherwise, the voice string is treated as a raw description prompt, allowing fully custom voice control.

### 6.3 Fish Speech (`engine_fish_speech.py`)

| Attribute | Value |
| --- | --- |
| Model | fishaudio/s2-pro (dual autoregressive) |
| Languages | 80+ languages |
| Streaming | Chunked fallback only |
| Voice cloning | Yes (via VQGAN/Firefly encoder) |
| Sample rate | 44100 Hz |
| License | Fish Audio Research License (non-commercial) |

**Architecture.** A dual-AR model: `DualARTransformer` generates semantic tokens from text, and `FireflyArchitecture` decodes semantic tokens to audio. Voice cloning works by encoding the reference audio through the VQGAN encoder to produce prompt tokens that condition the semantic token generation.

**Installation.** Requires `fish-speech` installed from GitHub (not on PyPI): `pip install git+https://github.com/fishaudio/fish-speech.git`. Not included in `[all]` because its dependency chain conflicts with `parler-tts`.

**Voice cloning flow:**
1. Reference audio is loaded from disk (via `VoiceStore`) or uploaded via API.
2. The audio is resampled to 44100 Hz if needed.
3. The `FireflyArchitecture.encode()` method produces VQ codes from the reference audio.
4. These codes become `prompt_tokens` for the semantic token generation step.
5. The `DualARTransformer` generates semantic tokens conditioned on both the text and the prompt tokens.

### 6.4 SeamlessM4T (`engine_seamless.py`)

| Attribute | Value |
| --- | --- |
| Model | facebook/seamless-m4t-v2-large |
| Languages | 36 speech output languages |
| Streaming | Not supported (info reports false) |
| Voice control | Integer speaker IDs (0-2) |
| Sample rate | 16000 Hz |
| License | CC-BY-NC-4.0 (non-commercial only) |

**Architecture.** Uses the standard `transformers` library (`SeamlessM4Tv2Model`, `AutoProcessor`). Text-to-speech via `model.generate()` with `tgt_lang` and `speaker_id` parameters. Internal language codes use 3-letter ISO format (e.g., `eng`, `tam`, `hin`); the engine maps from BCP-47 codes.

### 6.5 Edge TTS (`engine_edge_tts.py`)

| Attribute | Value |
| --- | --- |
| Model | travisvn/openai-edge-tts (remote service) |
| Languages | 40+ via Microsoft Edge voices |
| Streaming | Proxied (true streaming from remote) |
| GPU | Not required |
| Voice control | Microsoft Edge voice names |
| License | MIT |

**Architecture.** A pass-through engine: no local model loading. Proxies HTTP requests to a remote `openai-edge-tts` service. The remote service uses Microsoft Edge TTS voices. Uses `httpx.AsyncClient` for both batch and streaming requests.

**Voices.** 14 Indian-language Edge voices are pre-listed: English (IN), Tamil, Hindi, Telugu, Kannada, Malayalam, plus US and UK English fallbacks. The `default_voice` and `default_speed` are configurable.

---

## 7. API reference

### OpenAI-compatible endpoints

**`POST /v1/audio/speech`** — Generate speech from text.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | string | `"svara"` | Engine name |
| `input` | string | required | Text to synthesise (max 4096 chars) |
| `voice` | string | `"default"` | Voice ID or description |
| `response_format` | string | `"mp3"` | `mp3`, `wav`, `opus`, `flac`, `pcm` |
| `speed` | float | `1.0` | 0.25 to 4.0 |
| `language` | string | auto | BCP-47 code (e.g., `en`, `ta`) |
| `stream` | bool | `false` | Stream raw PCM int16 chunks |

When `stream=true`, returns `audio/pcm` with headers `X-Sample-Rate`, `X-Sample-Width`, `X-Channels`. Otherwise returns the requested format with `Content-Disposition: inline; filename="speech.{format}"`.

**`GET /v1/models`** — List loaded engines. Returns `ModelListResponse` with `data: [SpeechModelInfo]`.

### Voice management endpoints

**`GET /api/v1/voices`** — List all voices. Query param `?engine=svara` to filter. Returns `VoiceListResponse`.

**`POST /api/v1/voices/upload`** — Upload a voice reference sample. Multipart form: `voice_id`, `name`, `language`, `description`, `file`. Max 50 MB.

**`DELETE /api/v1/voices/{voice_id}`** — Delete a voice sample.

### Health

**`GET /health`** — Returns GPU status, loaded engines with capabilities, default engine, and voice sample count.

---

## 8. Configuration

All config lives in `config/settings.yaml` with `${ENV_VAR:default}` interpolation. Access via `config.get("dotted.path")`.

```yaml
app:
  name: "TTS Service"
  version: "1.0.0"

server:
  host: "0.0.0.0"
  port: 8600

engines:
  enabled: [svara]          # List of engines to load at startup
  default: svara            # Default engine when model is omitted
  svara:
    model_id: "kenpath/svara-tts-v1"
    snac_model: "hubertsiuzdak/snac_24khz"
    device: "${CUDA_DEVICE:cuda}"
    dtype: "bfloat16"
    max_tokens: 4096
    temperature: 0.6
    top_p: 0.95
    repetition_penalty: 1.1
    stream_frame_buffer: 21  # SNAC frames per decode batch (~210ms)

storage:
  voice_samples_dir: "data/voice_samples"

logging:
  level: "${LOG_LEVEL:INFO}"
```

---

## 9. Testing conventions

Framework: `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`). Tests mirror `app/` layout: `tests/test_audio_converter.py` tests `app/services/audio_converter.py`.

### Key fixtures (`tests/conftest.py`)

| Fixture | Returns | Purpose |
| --- | --- | --- |
| `mock_engine` | `MockEngine` | Deterministic engine producing a 440 Hz sine wave (0.5s, 24kHz) |
| `audio_samples` | `dict[str, np.ndarray]` | `sine_440`, `silence`, `short` known waveforms |
| `engine_registry` | `EngineRegistry` | Pre-loaded registry with the mock engine |
| `app_client` | `httpx.AsyncClient` | ASGITransport wired to the FastAPI app with mocked services |

### Test patterns

| What you are testing | Pattern |
| --- | --- |
| Engine ABC default behavior | `MockEngine` instance directly |
| Registry operations | `engine_registry` fixture (pre-wired) |
| Audio conversion | `AudioConverter()` + `audio_samples` fixture |
| API integration | `app_client` fixture with `ASGITransport` |
| Engine-specific logic | Unit tests on the engine class methods |
| Voice store CRUD | `VoiceStore(samples_dir=str(tmp_path))` with a file-system-backed directory |

### Running tests

```bash
pytest tests/ -v                         # Full suite
pytest tests/test_audio_converter.py -v  # Single file
```

Tests are fast (no GPU, no models, deterministic audio). The `MockEngine` generates a mathematically predictable sine wave, so tests can assert on exact byte lengths and format headers.

---

## 10. Containerization

### Dockerfile

Multi-stage build:
1. **Build stage.** Ubuntu 24.04 + CUDA 13.0, Python 3.12, `uv pip install ".[all]"`.
2. **Runtime stage.** Slimmer: Python 3.12 + ffmpeg (for pydub MP3/Opus encoding). Model cache at `/app/data/hf_cache`.

Entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8600`. Healthcheck: HTTP GET on `/health`.

### docker-compose.yml

Two services:
- **`tts`**: GPU-accelerated TTS on port 8600, persistent volume for model cache.
- **`wyoming-openai`** (profile `wyoming`): Bridges the OpenAI-compatible API to Home Assistant's Wyoming protocol on port 10300.

Both join the `nanai` external network for inter-service communication with Cognitive Companion and other services.

---

## 11. Integration with Cognitive Companion

Cognitive Companion's `backend/integrations/tts.py` (`TTSClient`) connects to this service. Configuration in Cognitive Companion's `config/settings.yaml`:

```yaml
tts:
  url: "${TTS_API_URL}"         # http://tts-service:8600
  default_voice: "speaker_0"
  default_speed: 0.85
```

The `TTSClient` provides two methods:
- `generate_audio(text, voice, speed, language)` — returns MP3 bytes.
- `stream_audio(text, voice, speed, language)` — async generator yielding PCM chunks.

Two notification channels in Cognitive Companion use this service:
- **`ha_speaker_tts`**: Generates MP3, uploads to MinIO, plays on HA media player via `media_player.play_media`.
- **`pwa_tts_announcement`**: Streams PCM chunks directly to PWA clients via WebSocket for real-time browser playback.

---

## 12. Port allocation in the nanai ecosystem

| Service | Port |
| --- | --- |
| Cognitive Companion backend | 8000 |
| Person Identification Service | 8200 |
| Semantic Memory Service | 8400 |
| **TTS Service** | **8600** |
| Tracking Orchestrator | 8000 (internal) |
| Triton Inference Server | 8700-8702 |
| RTSP Ingress | 8090 |
| Wyoming OpenAI proxy | 10300 |

---

## 13. What NOT to do

**Architecture and layering.**
- Do not instantiate engines or services in routers. Read from `request.app.state`.
- Do not import heavy ML packages at module level in engine files. Import inside `load()`.
- Do not add a new engine without registering it in `_ENGINE_CLASSES` in `engine_registry.py`.

**Audio handling.**
- Do not return raw tensors across the engine boundary. Convert to `np.ndarray[float32]`.
- Do not block the event loop with model inference. Use `loop.run_in_executor(None, ...)`.
- Do not leak GPU memory. Every engine must implement `unload()` and call `torch.cuda.empty_cache()`.

**Config and dependencies.**
- Do not hardcode model IDs or paths. Use `config.get()`.
- Do not add a runtime dependency without updating `pyproject.toml` optional-dependencies.
- Do not store secrets in YAML config. Use `${ENV_VAR}` interpolation.

**Logging and error handling.**
- Do not use `print()`. Use `logging.getLogger(__name__)`.
- Do not use bare `except:`. Log and return a zero value or re-raise.
- Do not let exceptions bubble out of `load()`. The registry catches and logs them, then continues with the remaining engines.

**Streaming.**
- Do not assume all engines support true streaming. Check `engine.info().supports_streaming`. The base class provides a chunked fallback.

---

## 14. Where to look when stuck

| Goal | File |
| --- | --- |
| Startup wiring | `app/main.py` (lifespan) |
| Engine contract | `app/services/engine_base.py` |
| How engines are loaded | `app/services/engine_registry.py` |
| How an engine does token-level streaming | `app/services/engine_svara.py` (`_generate_with_streamer`, `synthesize_stream`) |
| Voice cloning on disk | `app/services/voice_store.py` |
| Audio format conversion | `app/services/audio_converter.py` |
| API request model | `app/models/openai_compat.py` |
| Speech endpoint logic | `app/routers/openai_speech.py` |
| Config schema | `config/settings.yaml` |
| Docker setup | `Dockerfile`, `docker-compose.yml` |
| Test patterns | `tests/conftest.py` |
