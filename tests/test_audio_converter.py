"""Tests for audio format conversion."""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from app.services.audio_converter import AudioConverter, FORMAT_CONTENT_TYPE


@pytest.fixture
def converter() -> AudioConverter:
    return AudioConverter()


class TestToWav:
    def test_produces_valid_wav_header(self, converter: AudioConverter, audio_samples):
        wav = converter.to_wav(audio_samples["sine_440"], 24000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "

    def test_wav_is_readable_by_soundfile(self, converter: AudioConverter, audio_samples):
        import soundfile as sf

        wav = converter.to_wav(audio_samples["sine_440"], 24000)
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
        assert sr == 24000
        assert len(audio) == len(audio_samples["sine_440"])

    def test_silence_produces_valid_wav(self, converter: AudioConverter, audio_samples):
        wav = converter.to_wav(audio_samples["silence"], 24000)
        assert len(wav) > 44  # header + data


class TestToPcm:
    def test_produces_int16_bytes(self, converter: AudioConverter, audio_samples):
        pcm = converter.to_pcm(audio_samples["sine_440"])
        assert len(pcm) == len(audio_samples["sine_440"]) * 2  # 2 bytes per sample

    def test_round_trip_accuracy(self, converter: AudioConverter):
        """PCM conversion should preserve signal within int16 quantization."""
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = converter.to_pcm(original)
        recovered = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        np.testing.assert_allclose(recovered, original, atol=1.0 / 32767)

    def test_empty_audio(self, converter: AudioConverter):
        pcm = converter.to_pcm(np.array([], dtype=np.float32))
        assert pcm == b""


class TestConvert:
    def test_wav_dispatch(self, converter: AudioConverter, audio_samples):
        result = converter.convert(audio_samples["short"], 24000, "wav")
        assert result[:4] == b"RIFF"

    def test_pcm_dispatch(self, converter: AudioConverter, audio_samples):
        result = converter.convert(audio_samples["short"], 24000, "pcm")
        assert len(result) == len(audio_samples["short"]) * 2

    def test_flac_dispatch(self, converter: AudioConverter, audio_samples):
        result = converter.convert(audio_samples["short"], 24000, "flac")
        assert result[:4] == b"fLaC"

    def test_unknown_format_falls_back_to_wav(self, converter: AudioConverter, audio_samples):
        result = converter.convert(audio_samples["short"], 24000, "xyz")
        assert result[:4] == b"RIFF"

    def test_case_insensitive(self, converter: AudioConverter, audio_samples):
        result = converter.convert(audio_samples["short"], 24000, "WAV")
        assert result[:4] == b"RIFF"


class TestContentType:
    def test_known_formats(self):
        assert AudioConverter.content_type("wav") == "audio/wav"
        assert AudioConverter.content_type("mp3") == "audio/mpeg"
        assert AudioConverter.content_type("opus") == "audio/ogg"
        assert AudioConverter.content_type("flac") == "audio/flac"
        assert AudioConverter.content_type("pcm") == "audio/pcm"

    def test_unknown_format_defaults_to_wav(self):
        assert AudioConverter.content_type("xyz") == "audio/wav"
