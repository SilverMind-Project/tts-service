"""Integration tests for the OpenAI-compatible speech API."""

from __future__ import annotations

import io

import pytest


class TestCreateSpeech:
    @pytest.mark.asyncio
    async def test_non_streaming_wav(self, app_client):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "mock", "input": "hello", "response_format": "wav"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_non_streaming_pcm(self, app_client):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "mock", "input": "hello", "response_format": "pcm"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"
        assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_non_streaming_flac(self, app_client):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "mock", "input": "hello", "response_format": "flac"},
        )
        assert resp.status_code == 200
        assert resp.content[:4] == b"fLaC"

    @pytest.mark.asyncio
    async def test_streaming(self, app_client):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "mock", "input": "hello", "stream": True},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"
        assert resp.headers["x-sample-rate"] == "24000"
        assert resp.headers["x-sample-width"] == "16"
        assert resp.headers["x-channels"] == "1"
        assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_unavailable_engine_returns_400(self, app_client):
        # Override the registry to return None
        app_client._transport.app.state.engine_registry._engines.clear()
        app_client._transport.app.state.engine_registry._default = ""

        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "nonexistent", "input": "hello"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_default_model_used(self, app_client):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"input": "hello", "response_format": "pcm"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_content_disposition_header(self, app_client):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "mock", "input": "hello", "response_format": "mp3"},
        )
        assert resp.status_code == 200
        assert "speech.mp3" in resp.headers.get("content-disposition", "")


class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_loaded_engines(self, app_client):
        resp = await app_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "mock"
