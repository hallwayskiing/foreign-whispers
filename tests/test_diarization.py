import pytest
from foreign_whispers.diarization import (
    PYANNOTE_PIPELINE,
    _load_pyannote_pipeline_class,
    diarize_audio,
)


def test_returns_empty_without_token():
    result = diarize_audio("/any/path.wav", hf_token=None)
    assert result == []


def test_returns_empty_with_empty_token():
    result = diarize_audio("/any/path.wav", hf_token="")
    assert result == []


def test_returns_empty_when_pyannote_absent(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    result = diarize_audio("/any/path.wav", hf_token="fake-token")
    assert result == []


def test_uses_pyannote_4_pipeline_and_token(monkeypatch):
    calls = {}

    class FakeOutput:
        speaker_diarization = [
            (type("Turn", (), {"start": 1.0, "end": 2.0})(), "SPEAKER_01")
        ]

    class FakePipeline:
        @staticmethod
        def from_pretrained(model_id, token):
            calls["model_id"] = model_id
            calls["token"] = token

            def _run(audio_path):
                calls["audio_path"] = audio_path
                return FakeOutput()

            return _run

    monkeypatch.setitem(__import__("sys").modules, "pyannote.audio", type("Mod", (), {"Pipeline": FakePipeline})())

    result = diarize_audio("/tmp/example.wav", hf_token="hf-test-token")

    assert calls == {
        "model_id": PYANNOTE_PIPELINE,
        "token": "hf-test-token",
        "audio_path": "/tmp/example.wav",
    }
    assert result == [{"start_s": 1.0, "end_s": 2.0, "speaker": "SPEAKER_01"}]


def test_falls_back_to_legacy_itertracks_output(monkeypatch):
    class FakeLegacyOutput:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield (type("Turn", (), {"start": 3.0, "end": 4.5})(), None, "SPEAKER_02")

    class FakePipeline:
        @staticmethod
        def from_pretrained(model_id, token):
            return lambda audio_path: FakeLegacyOutput()

    monkeypatch.setitem(__import__("sys").modules, "pyannote.audio", type("Mod", (), {"Pipeline": FakePipeline})())

    result = diarize_audio("/tmp/example.wav", hf_token="hf-test-token")

    assert result == [{"start_s": 3.0, "end_s": 4.5, "speaker": "SPEAKER_02"}]


def test_returns_none_when_torchcodec_warning_is_emitted(monkeypatch, caplog):
    class FakePipeline:
        pass

    class FakeWarning:
        def __init__(self, message):
            self.message = message

    class FakeCatchWarnings:
        def __enter__(self):
            return [FakeWarning("torchcodec is not installed correctly for this runtime")]

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("foreign_whispers.diarization.importlib.import_module", lambda _: type("Mod", (), {"Pipeline": FakePipeline})())
    monkeypatch.setattr("foreign_whispers.diarization.warnings.catch_warnings", lambda record=True: FakeCatchWarnings())
    monkeypatch.setattr("foreign_whispers.diarization.warnings.simplefilter", lambda *args, **kwargs: None)

    with caplog.at_level("WARNING"):
        result = _load_pyannote_pipeline_class()

    assert result is None
    assert "torchcodec is unavailable for pyannote file decoding" in caplog.text


@pytest.mark.requires_pyannote
def test_real_diarization_returns_speaker_labels(tmp_path):
    """Integration test — requires pyannote.audio, FW_HF_TOKEN, and a real sample."""
    import os

    token = os.environ.get("FW_HF_TOKEN")
    sample = os.environ.get("FW_DIARIZATION_SAMPLE")
    if not token:
        pytest.skip("FW_HF_TOKEN not set")
    if not sample:
        pytest.skip("FW_DIARIZATION_SAMPLE not set")

    result = diarize_audio(sample, hf_token=token)
    assert isinstance(result, list)
    assert result, (
        "Expected non-empty diarization result. "
        "If pyannote 4.x is installed, verify torchcodec is compatible with "
        "the current torch build and FFmpeg shared libraries are available."
    )
    for r in result:
        assert "start_s" in r and "end_s" in r and "speaker" in r
