"""Audio format conversion — WAV, MP3, OGG/Opus, FLAC, raw PCM."""

from __future__ import annotations

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Content-type map
FORMAT_CONTENT_TYPE: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
    "ogg": "audio/ogg",
}


class AudioConverter:
    """Converts float32 numpy audio to various output formats."""

    @staticmethod
    def to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode as 16-bit PCM WAV using soundfile."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @staticmethod
    def to_pcm(audio: np.ndarray) -> bytes:
        """Raw 16-bit signed little-endian PCM."""
        return (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def to_mp3(audio: np.ndarray, sample_rate: int, bitrate: int = 128) -> bytes:
        """Encode as MP3 using pydub (requires ffmpeg)."""
        from pydub import AudioSegment

        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        segment = AudioSegment(
            data=pcm,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        buf = io.BytesIO()
        segment.export(buf, format="mp3", bitrate=f"{bitrate}k")
        return buf.getvalue()

    @staticmethod
    def to_opus(audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode as OGG/Opus using pydub (requires ffmpeg)."""
        from pydub import AudioSegment

        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        segment = AudioSegment(
            data=pcm,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        buf = io.BytesIO()
        segment.export(buf, format="ogg", codec="libopus")
        return buf.getvalue()

    @staticmethod
    def to_flac(audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode as FLAC using soundfile."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
        return buf.getvalue()

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_format: str = "mp3",
    ) -> bytes:
        """Convert audio to the requested format."""
        fmt = output_format.lower().strip()
        if fmt == "wav":
            return self.to_wav(audio, sample_rate)
        if fmt == "mp3":
            return self.to_mp3(audio, sample_rate)
        if fmt in ("opus", "ogg"):
            return self.to_opus(audio, sample_rate)
        if fmt == "flac":
            return self.to_flac(audio, sample_rate)
        if fmt == "pcm":
            return self.to_pcm(audio)
        logger.warning("Unknown format '%s', falling back to WAV", fmt)
        return self.to_wav(audio, sample_rate)

    @staticmethod
    def content_type(output_format: str) -> str:
        return FORMAT_CONTENT_TYPE.get(output_format.lower(), "audio/wav")
