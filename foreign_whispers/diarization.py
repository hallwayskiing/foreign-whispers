"""Speaker diarization using pyannote.audio.

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-community-1 agreement on
Hugging Face and providing an HF token. Returns empty list with a warning if
the dependency is absent or the token is missing.
"""
import copy
import importlib
import logging
import warnings

logger = logging.getLogger(__name__)
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-community-1"
_TORCHCODEC_WARNING_FRAGMENT = "torchcodec is not installed correctly"


def _load_pyannote_pipeline_class():
    """Import pyannote Pipeline and surface torchcodec decoder warnings.

    pyannote 4.x relies on torchcodec for file-based audio decoding. When the
    runtime is missing FFmpeg shared libraries or an incompatible torchcodec
    wheel is installed, pyannote emits a warning during import and file-based
    diarization later fails. Catch that warning early so callers get a direct,
    actionable log message instead of a silent empty result.
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Pipeline = importlib.import_module("pyannote.audio").Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return None

    for warning in caught:
        message = str(warning.message)
        if _TORCHCODEC_WARNING_FRAGMENT in message:
            logger.warning(
                "torchcodec is unavailable for pyannote file decoding. "
                "Install a torchcodec build compatible with the current torch "
                "version and ensure FFmpeg shared libraries are available. %s",
                message,
            )
            return None

    return Pipeline


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    Pipeline = _load_pyannote_pipeline_class()
    if Pipeline is None:
        return []

    try:
        pipeline = Pipeline.from_pretrained(
            PYANNOTE_PIPELINE,
            token=hf_token,
        )
        output = pipeline(audio_path)

        if hasattr(output, "speaker_diarization"):
            return [
                {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
                for turn, speaker in output.speaker_diarization
            ]

        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in output.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def assign_speakers(segments: list[dict], diarization: list[dict]) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, find the diarization interval with maximum temporal
    overlap and copy its speaker label. If diarization is empty, default all
    segments to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        The input list is not mutated.
    """
    labeled_segments = copy.deepcopy(segments)

    if not diarization:
        for segment in labeled_segments:
            segment["speaker"] = "SPEAKER_00"
        return labeled_segments

    for segment in labeled_segments:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        best_speaker = "SPEAKER_00"
        best_overlap = -1.0

        for diar in diarization:
            diar_start = float(diar.get("start_s", 0.0))
            diar_end = float(diar.get("end_s", diar_start))
            overlap = max(0.0, min(seg_end, diar_end) - max(seg_start, diar_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar.get("speaker", "SPEAKER_00")

        segment["speaker"] = best_speaker

    return labeled_segments
