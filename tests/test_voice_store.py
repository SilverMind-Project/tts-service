"""Tests for the voice sample store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.voice_store import VoiceStore


@pytest.fixture
def voice_dir(tmp_path: Path) -> Path:
    """Create a temporary voice samples directory with one sample."""
    sample_dir = tmp_path / "test_voice"
    sample_dir.mkdir()
    meta = {"name": "Test Voice", "language": "en", "description": "A test voice"}
    (sample_dir / "meta.json").write_text(json.dumps(meta))
    # Write minimal valid WAV header (silence)
    import soundfile as sf
    import numpy as np

    audio = np.zeros(2400, dtype=np.float32)
    sf.write(str(sample_dir / "reference.wav"), audio, 24000)
    return tmp_path


class TestVoiceStore:
    def test_load_discovers_samples(self, voice_dir: Path):
        store = VoiceStore(samples_dir=str(voice_dir))
        store.load()
        assert len(store.list_samples()) == 1
        assert store.get("test_voice") is not None

    def test_get_missing_voice_returns_none(self, voice_dir: Path):
        store = VoiceStore(samples_dir=str(voice_dir))
        store.load()
        assert store.get("nonexistent") is None

    def test_load_skips_dirs_without_meta(self, tmp_path: Path):
        (tmp_path / "no_meta").mkdir()
        (tmp_path / "no_meta" / "reference.wav").write_bytes(b"dummy")
        store = VoiceStore(samples_dir=str(tmp_path))
        store.load()
        assert len(store.list_samples()) == 0

    def test_load_skips_dirs_without_audio(self, tmp_path: Path):
        sample_dir = tmp_path / "no_audio"
        sample_dir.mkdir()
        (sample_dir / "meta.json").write_text('{"name": "test"}')
        store = VoiceStore(samples_dir=str(tmp_path))
        store.load()
        assert len(store.list_samples()) == 0

    def test_load_creates_missing_dir(self, tmp_path: Path):
        store = VoiceStore(samples_dir=str(tmp_path / "new_dir"))
        store.load()
        assert (tmp_path / "new_dir").exists()

    def test_save_and_retrieve(self, tmp_path: Path):
        store = VoiceStore(samples_dir=str(tmp_path))
        store.load()
        sample = store.save_sample(
            voice_id="new_voice",
            name="New Voice",
            language="ta",
            description="Tamil voice",
            audio_bytes=b"fake audio data",
        )
        assert sample.voice_id == "new_voice"
        assert store.get("new_voice") is not None
        assert (tmp_path / "new_voice" / "meta.json").exists()
        assert (tmp_path / "new_voice" / "reference.wav").exists()

    def test_delete_sample(self, voice_dir: Path):
        store = VoiceStore(samples_dir=str(voice_dir))
        store.load()
        assert store.delete_sample("test_voice") is True
        assert store.get("test_voice") is None
        assert not (voice_dir / "test_voice").exists()

    def test_delete_nonexistent(self, voice_dir: Path):
        store = VoiceStore(samples_dir=str(voice_dir))
        store.load()
        assert store.delete_sample("nonexistent") is False
