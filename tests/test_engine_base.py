"""Tests for the TTSEngine abstract base class default behaviors."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import MockEngine


class TestSynthesizeStreamDefault:
    """Test the default synthesize_stream() fallback (full synthesis then chunk)."""

    @pytest.mark.asyncio
    async def test_stream_produces_pcm_bytes(self, mock_engine: MockEngine):
        chunks = []
        async for chunk in mock_engine.synthesize_stream("hello", chunk_size=4096):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_total_matches_synthesize(self, mock_engine: MockEngine):
        result = await mock_engine.synthesize("hello")
        expected_pcm = (result.audio * 32767).astype(np.int16).tobytes()

        streamed = b""
        async for chunk in mock_engine.synthesize_stream("hello", chunk_size=4096):
            streamed += chunk

        assert len(streamed) == len(expected_pcm)

    @pytest.mark.asyncio
    async def test_chunk_size_respected(self, mock_engine: MockEngine):
        chunk_size = 1024
        async for chunk in mock_engine.synthesize_stream("hello", chunk_size=chunk_size):
            # All chunks except the last should be exactly chunk_size
            assert len(chunk) <= chunk_size

    @pytest.mark.asyncio
    async def test_small_chunk_size(self, mock_engine: MockEngine):
        chunks = []
        async for chunk in mock_engine.synthesize_stream("hello", chunk_size=64):
            chunks.append(chunk)
        # Should produce many small chunks
        assert len(chunks) > 10
