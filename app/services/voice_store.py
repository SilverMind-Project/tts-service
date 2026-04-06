"""Voice sample store — manages reference audio files for voice cloning."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VoiceSample:
    """A stored voice reference sample."""

    def __init__(
        self,
        voice_id: str,
        name: str,
        language: str,
        description: str,
        audio_path: Path,
        sample_rate: int = 24000,
    ) -> None:
        self.voice_id = voice_id
        self.name = name
        self.language = language
        self.description = description
        self.audio_path = audio_path
        self.sample_rate = sample_rate

    def load_audio(self) -> tuple[np.ndarray, int]:
        """Load the reference audio as float32 numpy array."""
        import soundfile as sf
        audio, sr = sf.read(str(self.audio_path), dtype="float32")
        return audio, sr


class VoiceStore:
    """Manages voice reference samples stored on disk.

    Directory layout::

        data/voice_samples/
        ├── my_voice/
        │   ├── meta.json        # {name, language, description, sample_rate}
        │   └── reference.wav    # 10-30 second WAV audio sample
        └── grandma_voice/
            ├── meta.json
            └── reference.wav
    """

    def __init__(self, samples_dir: str = "data/voice_samples") -> None:
        self._dir = Path(samples_dir)
        self._samples: dict[str, VoiceSample] = {}

    def load(self) -> None:
        """Scan the samples directory and load metadata."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._samples.clear()

        for voice_dir in sorted(self._dir.iterdir()):
            if not voice_dir.is_dir():
                continue
            meta_path = voice_dir / "meta.json"
            audio_files = list(voice_dir.glob("reference.*"))
            if not meta_path.exists() or not audio_files:
                continue
            try:
                meta = json.loads(meta_path.read_text())
                sample = VoiceSample(
                    voice_id=voice_dir.name,
                    name=meta.get("name", voice_dir.name),
                    language=meta.get("language", "en"),
                    description=meta.get("description", ""),
                    audio_path=audio_files[0],
                    sample_rate=meta.get("sample_rate", 24000),
                )
                self._samples[voice_dir.name] = sample
                logger.info("Loaded voice sample: %s", voice_dir.name)
            except Exception:
                logger.exception("Error loading voice sample from %s", voice_dir)

    def get(self, voice_id: str) -> VoiceSample | None:
        return self._samples.get(voice_id)

    def list_samples(self) -> list[VoiceSample]:
        return list(self._samples.values())

    def save_sample(
        self,
        voice_id: str,
        name: str,
        language: str,
        description: str,
        audio_bytes: bytes,
        filename: str = "reference.wav",
    ) -> VoiceSample:
        """Save a new voice reference sample to disk."""
        voice_dir = self._dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Write audio
        audio_path = voice_dir / filename
        audio_path.write_bytes(audio_bytes)

        # Write metadata
        meta = {
            "name": name,
            "language": language,
            "description": description,
            "sample_rate": 24000,
        }
        (voice_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        sample = VoiceSample(
            voice_id=voice_id,
            name=name,
            language=language,
            description=description,
            audio_path=audio_path,
        )
        self._samples[voice_id] = sample
        return sample

    def delete_sample(self, voice_id: str) -> bool:
        """Delete a voice sample directory."""
        import shutil
        voice_dir = self._dir / voice_id
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
            self._samples.pop(voice_id, None)
            return True
        return False
