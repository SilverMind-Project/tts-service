# TTS Service

Self-hosted text-to-speech microservice for the Cognitive Companion platform. Exposes an **OpenAI-compatible API** (`/v1/audio/speech`) and integrates with Home Assistant via the **Wyoming protocol** through a [wyoming_openai](https://github.com/roryeckel/wyoming_openai) sidecar.

## Features

- **Multi-engine architecture**  -  swap between TTS models via config
- **OpenAI-compatible API** (`POST /v1/audio/speech`)  -  drop-in replacement for OpenAI TTS
- **Home Assistant integration**  -  Wyoming protocol bridge via wyoming_openai sidecar
- **Streaming support**  -  chunked audio streaming for low-latency playback
- **Voice cloning**  -  upload reference audio samples (10-30s) for personalised voices
- **Multi-language**  -  English (Indian accent) and Tamil as primary languages
- **GPU accelerated**  -  runs on NVIDIA GPUs (DGX Spark, RTX 4080, etc.)

## Supported TTS Engines

| Engine | Model | Languages | Voice Cloning | Streaming | VRAM | License |
|--------|-------|-----------|---------------|-----------|------|---------|
| **svara** (default) | [kenpath/svara-tts-v1](https://huggingface.co/kenpath/svara-tts-v1) | English (IN), Tamil, Hindi, Telugu, + 10 more Indian languages | Yes | Yes (chunked) | ~8 GB (bf16) | Apache-2.0 |
| **parler** | [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) | English (IN), Tamil, Hindi, Telugu, + 7 more | No (text prompts) | Yes (chunked) | ~4 GB (fp16) | Apache-2.0 |
| **fish_speech** | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | 80+ languages incl. English, Tamil, Hindi | Yes | Yes (chunked) | ~8-12 GB (bf16) | Fish Audio Research* |
| **seamless** | [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) | 36 languages incl. English, Tamil, Hindi, Telugu | No | **No** | ~6-8 GB (fp16) | CC-BY-NC-4.0* |
| **melo** | [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) | English (IN, US, UK, AU) | No | **No** | <1 GB (CPU) | MIT |
| **edge_tts** | [openai-edge-tts](https://github.com/travisvn/openai-edge-tts) | 40+ languages (Microsoft Edge voices) | No | Yes | None (remote) | MIT |

\* **fish_speech**: Free for research & non-commercial use. Commercial use requires a licence from [Fish Audio](https://fish.audio).
\* **seamless**: CC-BY-NC-4.0  -  non-commercial use only. Commercial use requires permission from Meta.

### Streaming Support

| Engine | True Streaming | Notes |
|--------|---------------|-------|
| svara | Chunked | Full generation then chunked output. True token-level streaming possible with streamer API (future enhancement). |
| parler | Chunked | Full generation then chunked output. |
| fish_speech | Chunked | Full generation then chunked output. Fish Speech's dual-AR architecture supports true streaming (future enhancement). |
| seamless | **Not supported** | Batch inference only. Returns complete audio. When `stream=true` is requested, falls back to chunked output. |
| melo | **Not supported** | Batch inference only. Returns complete audio. When `stream=true` is requested, falls back to chunked output of the complete audio. |
| edge_tts | Yes (proxied) | Streams PCM chunks directly from the remote openai-edge-tts service. |

### Engine Selection Guide

- **Best quality for Indian English + Tamil**: `svara`  -  3B-parameter model, excellent prosody
- **Most languages + voice cloning**: `fish_speech`  -  80+ languages, high-quality cloning from short samples
- **Lower VRAM / emotion control**: `parler`  -  uses text descriptions to control voice characteristics
- **Multilingual TTS (non-commercial)**: `seamless`  -  36 speech output languages via Meta's SeamlessM4T v2
- **CPU-only / fastest latency (English only)**: `melo`  -  runs on CPU, real-time synthesis
- **No GPU / remote synthesis**: `edge_tts`  -  pass-through to a remote openai-edge-tts service using Microsoft Edge TTS voices, no local compute needed

## Quick Start

### Docker Compose (recommended)

```bash
cp .env.example .env
docker compose up -d
```

First start downloads models (~6 GB). Subsequent starts use cached models.
The default image installs `.[all]`, which currently excludes `melo` and
`fish_speech`: the latest PyPI release of MeloTTS fails in clean builds, and
Fish Speech currently conflicts with `parler-tts` on `protobuf`.

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[svara,dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload
```

If you want the CPU-only `melo` engine or Fish Speech, treat them as opt-in
dependencies for now rather than relying on `.[all]`.

## API Reference

### OpenAI-Compatible API

#### `POST /v1/audio/speech`

Generate speech from text. Drop-in compatible with the OpenAI TTS API.

```bash
curl -X POST http://localhost:8200/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "svara",
    "input": "Hello, how are you today?",
    "voice": "speaker_0",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `svara` | Engine: `svara`, `parler`, `fish_speech`, `seamless`, `melo`, `edge_tts` |
| `input` | string | required | Text to synthesise (max 4096 chars) |
| `voice` | string | `default` | Voice ID or description (engine-specific) |
| `response_format` | string | `mp3` | Output format: `mp3`, `wav`, `opus`, `flac`, `pcm` |
| `speed` | float | `1.0` | Speed multiplier (0.25 – 4.0) |
| `language` | string | auto | Language code: `en`, `ta`, `hi`, etc. |
| `stream` | bool | `false` | Stream raw PCM chunks |

**Streaming:**

```bash
curl -X POST http://localhost:8200/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "svara", "input": "Hello", "stream": true}' \
  --output stream.pcm
```

Streaming returns raw 16-bit signed PCM at 24kHz mono. Response headers include `X-Sample-Rate`, `X-Sample-Width`, and `X-Channels`.

#### `GET /v1/models`

List available TTS engines.

```bash
curl http://localhost:8200/v1/models
```

### Voice Management

#### `GET /api/v1/voices`

List all voices (built-in + custom samples).

```bash
curl http://localhost:8200/api/v1/voices
```

#### `POST /api/v1/voices/upload`

Upload a voice reference sample for voice cloning.

```bash
curl -X POST http://localhost:8200/api/v1/voices/upload \
  -F "voice_id=grandma" \
  -F "name=Grandma's Voice" \
  -F "language=ta" \
  -F "description=Warm Tamil voice" \
  -F "file=@reference.wav"
```

#### `DELETE /api/v1/voices/{voice_id}`

Delete a custom voice sample.

### Health Check

#### `GET /health`

```bash
curl http://localhost:8200/health
```

Returns GPU status, loaded engines, and voice sample count.

## Voice Cloning

Engines that support voice cloning (`svara` and `fish_speech`) can use reference audio samples to generate speech in a specific voice.

### Preparing Voice Samples

1. Record a **10-30 second** clear audio sample of the target voice
2. Use **WAV format**, 16-bit, mono, 24kHz (other formats accepted but will be resampled)
3. Ensure minimal background noise
4. The speaker should talk naturally  -  reading a paragraph works well

### Uploading Samples

**Via API:**

```bash
curl -X POST http://localhost:8200/api/v1/voices/upload \
  -F "voice_id=grandma" \
  -F "name=Grandma" \
  -F "language=ta" \
  -F "file=@grandma_sample.wav"
```

**Via filesystem (for pre-provisioning):**

Place samples in `data/voice_samples/` with this structure:

```
data/voice_samples/
├── grandma/
│   ├── meta.json
│   └── reference.wav
└── caregiver/
    ├── meta.json
    └── reference.wav
```

`meta.json` format:

```json
{
  "name": "Grandma",
  "language": "ta",
  "description": "Warm Tamil voice for familiar interactions",
  "sample_rate": 24000
}
```

### Using Cloned Voices

Once uploaded, use the `voice_id` in synthesis requests:

```bash
curl -X POST http://localhost:8200/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "svara", "input": "Hello paatti", "voice": "grandma"}' \
  --output speech.mp3
```

## Home Assistant Integration

Home Assistant's voice pipeline uses the **Wyoming protocol** for STT and TTS. The [wyoming_openai](https://github.com/roryeckel/wyoming_openai) proxy bridges the TTS service's OpenAI-compatible API to Wyoming, allowing Home Assistant to use any of the configured TTS engines.

### Start the Wyoming proxy

```bash
docker compose --profile wyoming up -d
```

This starts the `wyoming-openai` sidecar on port **10300**. It connects to the TTS service at `http://tts:8200/v1` and exposes a Wyoming server.

### Add to Home Assistant

1. Go to **Settings > Devices & Services > Add Integration**
2. Search for **Wyoming Protocol**
3. Enter the host IP and port `10300`
4. Home Assistant will discover the available TTS voices

The TTS engines then appear as voice options in any Home Assistant voice assistant pipeline.

### Wyoming Environment Variables

The Wyoming sidecar is configured via environment variables in `.env`:

```bash
WYOMING_TTS_OPENAI_URL=http://tts:8200/v1
WYOMING_TTS_MODELS=svara,parler,fish_speech,seamless,melo,edge_tts
WYOMING_TTS_VOICES=speaker_0,speaker_1,female_calm,male_clear,default
```

## Configuration

All configuration is in `config/settings.yaml` with `${ENV_VAR}` interpolation.

### Switching Engines

Edit `config/settings.yaml`:

```yaml
engines:
  enabled:
    - svara    # primary (GPU)
    - melo     # fallback (CPU)
  default: svara
```

### Running Multiple Engines

You can load multiple engines simultaneously. The `model` field in the OpenAI API selects which engine to use:

```bash
# Use svara (default)
curl -X POST .../v1/audio/speech -d '{"input": "Hello", "model": "svara"}'

# Use melo (CPU fallback)
curl -X POST .../v1/audio/speech -d '{"input": "Hello", "model": "melo"}'
```

### Engine-Specific Voice IDs

| Engine | Voice Format | Examples |
|--------|-------------|----------|
| svara | Speaker tag | `speaker_0` (female), `speaker_1` (male) |
| parler | Preset name or free-text description | `female_calm`, `female_tamil`, `male_clear`, or any description string |
| fish_speech | `default` or custom voice sample ID | `default`, `grandma` (uploaded via voice cloning API) |
| seamless | Speaker index | `default`, `speaker_0`, `speaker_1`, `speaker_2` |
| melo | Language-accent code | `EN-IN` (Indian), `EN-US`, `EN-BR`, `EN-AU` |
| edge_tts | Edge TTS voice name | `en-IN-NeerjaExpressiveNeural`, `ta-IN-PallaviNeural`, `hi-IN-SwaraNeural` |

### Parler Voice Descriptions

The Parler engine uses text descriptions to control voice characteristics. Built-in presets:

| Preset | Description |
|--------|-------------|
| `female_calm` | Calm Indian English female, moderate pace, warm tone |
| `male_clear` | Clear Indian English male, steady pace, friendly tone |
| `female_tamil` | Gentle Tamil female, slow and clear |
| `male_tamil` | Clear Tamil male, moderate pace |
| `female_elderly_friendly` | Warm, gentle, slow  -  ideal for speaking to seniors |

Custom descriptions work too:

```json
{"model": "parler", "voice": "A cheerful young woman speaks quickly with excitement"}
```

### Fish Speech Voice Cloning

Fish Speech (`fish_speech`) supports high-quality voice cloning from short reference samples. Upload a 10-30 second WAV sample via the voice management API, then use the `voice_id` in requests:

```bash
# Upload a reference sample
curl -X POST http://localhost:8200/api/v1/voices/upload \
  -F "voice_id=grandma" \
  -F "name=Grandma" \
  -F "language=ta" \
  -F "file=@grandma_sample.wav"

# Use the cloned voice with Fish Speech
curl -X POST http://localhost:8200/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "fish_speech", "input": "Hello paatti", "voice": "grandma"}' \
  --output speech.mp3
```

Fish Speech supports 80+ languages  -  pass the `language` field for best results when synthesising non-English text.

### SeamlessM4T Voices

SeamlessM4T (`seamless`) uses integer speaker IDs. Available voices: `default` (speaker 0), `speaker_0`, `speaker_1`, `speaker_2`. The engine supports 36 speech output languages  -  specify the `language` field with a BCP-47 code:

```bash
# Tamil TTS with SeamlessM4T
curl -X POST http://localhost:8200/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "seamless", "input": "வணக்கம்", "language": "ta"}' \
  --output speech.mp3
```

### Fish Speech Installation

Fish Speech requires the `fish-speech` package from GitHub (not on PyPI):

```bash
pip install git+https://github.com/fishaudio/fish-speech.git
# Or when installing the tts-service:
pip install -e ".[fish_speech,dev]"
```

Fish Speech is intentionally not part of `.[all]` right now because its current
dependency chain conflicts with `parler-tts`.

The model weights (~6 GB) are downloaded from HuggingFace on first use.

### OpenAI Edge TTS (Pass-Through)

The `edge_tts` engine proxies requests to a remote [openai-edge-tts](https://github.com/travisvn/openai-edge-tts) service, which uses Microsoft Edge TTS voices. No local GPU or model downloads required.

**Setup**: Deploy openai-edge-tts on your network (e.g. via Docker), then point the TTS service to it:

```yaml
engines:
  enabled:
    - edge_tts
  default: edge_tts

  edge_tts:
    base_url: "http://192.168.1.31:6060/v1"   # Your openai-edge-tts endpoint
    default_voice: "en-IN-NeerjaExpressiveNeural"
    default_speed: 0.85
```

Or via environment variable: `EDGE_TTS_URL=http://192.168.1.31:6060/v1`

**Usage**:

```bash
curl -X POST http://localhost:8200/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "edge_tts",
    "input": "Hello, how are you today?",
    "voice": "en-IN-NeerjaExpressiveNeural",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

**Popular voices for Indian languages**:

| Voice ID | Language | Description |
|----------|----------|-------------|
| `en-IN-NeerjaExpressiveNeural` | English (IN) | Expressive Indian English female (default) |
| `en-IN-PrabhatNeural` | English (IN) | Indian English male |
| `ta-IN-PallaviNeural` | Tamil | Tamil female |
| `ta-IN-ValluvarNeural` | Tamil | Tamil male |
| `hi-IN-SwaraNeural` | Hindi | Hindi female |
| `hi-IN-MadhurNeural` | Hindi | Hindi male |
| `te-IN-ShrutiNeural` | Telugu | Telugu female |
| `kn-IN-SapnaNeural` | Kannada | Kannada female |
| `ml-IN-SobhanaNeural` | Malayalam | Malayalam female |

See the full list of voices at the [Edge TTS voice gallery](https://speech.microsoft.com/portal/voicegallery).

## Hardware Requirements

| Engine | Min VRAM | Recommended | CPU Fallback |
|--------|----------|-------------|--------------|
| svara | 8 GB (bf16) | 16 GB | Not practical |
| parler | 4 GB (fp16) | 8 GB | Slow but works |
| fish_speech | 8 GB (bf16) | 16 GB | Not practical |
| seamless | 6 GB (fp16) | 8 GB | Slow but works |
| melo |  -  |  -  | Yes (designed for CPU) |
| edge_tts |  -  |  -  | N/A (remote service) |

**Tested on:**
- NVIDIA DGX Spark (Grace + Blackwell, 128 GB unified)  -  all engines run comfortably
- NVIDIA RTX 4080 (16 GB)  -  svara in bf16, parler in fp16, melo on CPU simultaneously. fish_speech and seamless also fit in bf16/fp16.

## Kubernetes Deployment

```bash
# Build and push to local registry
docker build -t localhost:32000/tts-service:latest .
docker push localhost:32000/tts-service:latest

# Deploy
kubectl apply -f kubernetes/base/pvc.yaml
kubectl apply -f kubernetes/local/deployment.yaml
kubectl apply -f kubernetes/base/service.yaml
```

The service is available at `tts-svc:8200` within the cluster.

## Integration with Cognitive Companion

The Cognitive Companion backend connects to this service via its TTS client. Configure in the Cognitive Companion's `config/settings.yaml`:

```yaml
tts:
  url: "http://tts-service:8200"
  default_voice: "speaker_0"
  default_speed: 0.85
```

The existing `TTSClient` in `backend/integrations/tts.py` already uses the OpenAI-compatible `/v1/audio/speech` endpoint, so it works with this service without code changes.
