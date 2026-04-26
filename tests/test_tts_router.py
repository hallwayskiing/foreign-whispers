"""Tests for POST /api/tts/{video_id} endpoint."""

import json
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def ui_dir(tmp_path):
    (tmp_path / "translations" / "argos").mkdir(parents=True)
    (tmp_path / "tts_audio" / "chatterbox").mkdir(parents=True)
    (tmp_path / "speakers" / "es").mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def client(monkeypatch, ui_dir):
    whisper_mod = types.ModuleType("whisper")
    whisper_mod.load_model = lambda *a, **kw: MagicMock()
    monkeypatch.setitem(sys.modules, "whisper", whisper_mod)

    tts_pkg = types.ModuleType("TTS")
    tts_api_mod = types.ModuleType("TTS.api")
    tts_api_mod.TTS = lambda *a, **kw: MagicMock()
    tts_pkg.api = tts_api_mod
    monkeypatch.setitem(sys.modules, "TTS", tts_pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", tts_api_mod)

    librosa_mod = types.ModuleType("librosa")
    librosa_mod.load = lambda *a, **kw: ([], 24000)
    monkeypatch.setitem(sys.modules, "librosa", librosa_mod)

    soundfile_mod = types.ModuleType("soundfile")
    soundfile_mod.write = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_mod)

    pyrubberband_mod = types.ModuleType("pyrubberband")
    pyrubberband_mod.time_stretch = lambda y, sr, speed: y
    monkeypatch.setitem(sys.modules, "pyrubberband", pyrubberband_mod)

    pydub_mod = types.ModuleType("pydub")
    pydub_mod.AudioSegment = MagicMock()
    monkeypatch.setitem(sys.modules, "pydub", pydub_mod)

    from api.src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", ui_dir)

    from api.src.routers.tts import router

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as c:
        yield c


def _translated_transcript():
    return {
        "text": "Hola mundo",
        "language": "es",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.5, "text": " Hola mundo", "speaker": "SPEAKER_00"},
        ],
    }


def test_tts_returns_audio_path(client, monkeypatch, ui_dir):
    """POST /api/tts/{video_id} returns path to generated WAV."""
    src = ui_dir / "translations" / "argos" / "Test Title.json"
    src.write_text(json.dumps(_translated_transcript()))

    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    calls = []

    def fake_tts(self, source_path, output_path, *, alignment=None, speaker_wav=None, voice_map=None):
        calls.append(
            {
                "source_path": source_path,
                "output_path": output_path,
                "alignment": alignment,
                "speaker_wav": speaker_wav,
                "voice_map": voice_map,
            }
        )
        wav = pathlib.Path(output_path) / "Test Title.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

    monkeypatch.setattr("api.src.services.tts_service.TTSService.text_file_to_speech", fake_tts)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000&alignment=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "G3Eup4mfJdA"
    assert body["audio_path"].endswith(".wav")
    assert body["config"] == "c-0000000"
    assert len(calls) == 1
    assert calls[0]["alignment"] is True
    assert calls[0]["speaker_wav"] is None
    assert calls[0]["voice_map"] == {"SPEAKER_00": "default.wav"}


def test_tts_skips_if_cached(client, monkeypatch, ui_dir):
    """Skip TTS if WAV already exists in config subdirectory."""
    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    config_dir = ui_dir / "tts_audio" / "chatterbox" / "c-0000000"
    config_dir.mkdir(parents=True)
    wav = config_dir / "Test Title.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)

    tts_called = {"count": 0}

    def tracking_tts(self, source_path, output_path, *, alignment=None, speaker_wav=None, voice_map=None):
        tts_called["count"] += 1

    monkeypatch.setattr("api.src.services.tts_service.TTSService.text_file_to_speech", tracking_tts)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000")
    assert resp.status_code == 200
    assert tts_called["count"] == 0


def test_tts_source_not_found(client, monkeypatch, ui_dir):
    """Returns 404 when translated transcript doesn't exist."""
    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: None,
    )

    resp = client.post("/api/tts/NONEXISTENT?config=c-0000000")
    assert resp.status_code == 404


def test_tts_runs_in_threadpool(client, monkeypatch, ui_dir):
    """TTS should run via run_in_executor to avoid blocking the event loop."""
    src = ui_dir / "translations" / "argos" / "Test Title.json"
    src.write_text(json.dumps(_translated_transcript()))

    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    executor_used = {"yes": False}

    def fake_tts(self, source_path, output_path, *, alignment=None, speaker_wav=None, voice_map=None):
        wav = pathlib.Path(output_path) / "Test Title.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

    monkeypatch.setattr("api.src.services.tts_service.TTSService.text_file_to_speech", fake_tts)

    async def tracking_run(executor, fn, *args, **kwargs):
        executor_used["yes"] = True
        return fn(*args, **kwargs)

    monkeypatch.setattr("api.src.routers.tts._run_in_threadpool", tracking_run)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000")
    assert resp.status_code == 200
    assert executor_used["yes"], "TTS should run in a thread pool"


def test_tts_rejects_invalid_config(client, monkeypatch, ui_dir):
    """Config param must match ^c-[0-9a-f]{7}$ to prevent path traversal."""
    resp = client.post("/api/tts/G3Eup4mfJdA?config=../../etc")
    assert resp.status_code == 422


def test_tts_uses_explicit_speaker_wav(client, monkeypatch, ui_dir):
    """Explicit speaker_wav should override automatic speaker resolution."""
    src = ui_dir / "translations" / "argos" / "Test Title.json"
    src.write_text(json.dumps(_translated_transcript()))

    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    calls = []

    def fake_tts(self, source_path, output_path, *, alignment=None, speaker_wav=None, voice_map=None):
        calls.append({"speaker_wav": speaker_wav, "voice_map": voice_map})
        wav = pathlib.Path(output_path) / "Test Title.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

    monkeypatch.setattr("api.src.services.tts_service.TTSService.text_file_to_speech", fake_tts)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000&speaker_wav=es/default.wav")
    assert resp.status_code == 200
    assert calls == [{"speaker_wav": "es/default.wav", "voice_map": None}]
