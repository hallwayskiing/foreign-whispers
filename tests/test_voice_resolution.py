from foreign_whispers.voice_resolution import resolve_speaker_wav


def test_resolve_speaker_wav_prefers_speaker_specific_file(tmp_path):
    speakers = tmp_path / "speakers"
    speakers.mkdir(parents=True)
    (speakers / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (speakers / "es").mkdir(parents=True)
    (speakers / "es" / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (speakers / "es" / "SPEAKER_00.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    result = resolve_speaker_wav(speakers, "es", "SPEAKER_00")
    assert result == "es/SPEAKER_00.wav"


def test_resolve_speaker_wav_falls_back_to_language_default(tmp_path):
    speakers = tmp_path / "speakers"
    speakers.mkdir(parents=True)
    (speakers / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    (speakers / "es").mkdir(parents=True)
    (speakers / "es" / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    result = resolve_speaker_wav(speakers, "es", "SPEAKER_01")
    assert result == "es/default.wav"


def test_resolve_speaker_wav_falls_back_to_global_default(tmp_path):
    speakers = tmp_path / "speakers"
    speakers.mkdir(parents=True)
    (speakers / "default.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    result = resolve_speaker_wav(speakers, "fr", "SPEAKER_00")
    assert result == "default.wav"
