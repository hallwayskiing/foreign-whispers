"""HTTP-agnostic service wrapping TTS engine functions."""

import json
import os
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from api.src.services import tts_engine
from api.src.services.tts_engine import text_file_to_speech as tts_text_file_to_speech


class TTSService:
    """Thin wrapper around the TTS pipeline.

    Accepts *ui_dir* and a pre-loaded *tts_engine* via constructor injection.
    """

    def __init__(self, ui_dir: Path, tts_engine: Any) -> None:
        self.ui_dir = ui_dir
        self.tts_engine = tts_engine

    def text_file_to_speech(
        self,
        source_path: str,
        output_path: str,
        *,
        alignment: bool | None = None,
        speaker_wav: str | None = None,
        voice_map: dict[str, str] | None = None,
    ) -> None:
        """Generate time-aligned TTS audio from a translated JSON transcript."""
        if voice_map:
            self._text_file_to_speech_with_voices(
                source_path,
                output_path,
                alignment=alignment,
                voice_map=voice_map,
            )
            return

        tts_text_file_to_speech(
            source_path,
            output_path,
            self.tts_engine,
            alignment=alignment,
            speaker_wav=speaker_wav,
        )

    def _text_file_to_speech_with_voices(
        self,
        source_path: str,
        output_path: str,
        *,
        alignment: bool | None,
        voice_map: dict[str, str],
    ) -> None:
        """Speaker-aware TTS path implemented at the service layer.

        This mirrors the core segment assembly flow from ``tts_engine`` but
        injects per-segment ``speaker_wav`` values without requiring changes to
        the lower-level engine module.
        """
        engine = self.tts_engine if self.tts_engine is not None else tts_engine._get_tts_engine()
        use_alignment = alignment if alignment is not None else tts_engine._ALIGNMENT_ENABLED

        save_name = pathlib.Path(source_path).stem + ".wav"
        print(f"generating {save_name}...", end="")

        segments = tts_engine.segments_from_file(source_path)
        if not segments:
            text = tts_engine.text_from_file(source_path)
            save_path = pathlib.Path(output_path) / save_name
            engine.tts_to_file(text=text, file_path=str(save_path))
            print("success!")
            return

        offset = tts_engine._compute_speech_offset(source_path)
        if offset > 0:
            print(f" (applying {offset:.1f}s speech offset)", end="")

        with open(source_path) as f:
            es_transcript = json.load(f)
        en_transcript = tts_engine._load_en_transcript(source_path)
        if use_alignment:
            metrics_list, align_map = tts_engine._build_alignment(en_transcript, es_transcript)
        else:
            metrics_list, align_map = [], {}
        aligned_list = list(align_map.values())

        seg_metas = []
        for i, seg in enumerate(segments):
            aligned_seg = align_map.get(i)
            stretch_factor = aligned_seg.stretch_factor if aligned_seg else 1.0
            target_sec = seg["end"] - seg["start"]

            seg_text = seg["text"]
            if aligned_seg is not None:
                from foreign_whispers.alignment import AlignAction

                if aligned_seg.action == AlignAction.REQUEST_SHORTER:
                    en_text = ""
                    en_segs = en_transcript.get("segments", [])
                    if i < len(en_segs):
                        en_text = en_segs[i].get("text", "")
                    seg_text = tts_engine._shorten_segment_text(en_text, seg["text"], target_sec)

            seg_metas.append(
                {
                    "index": i,
                    "speaker": seg.get("speaker", "SPEAKER_00"),
                    "text": seg_text,
                    "start": seg["start"],
                    "target_sec": target_sec,
                    "stretch_factor": stretch_factor,
                    "aligned_seg": aligned_seg,
                }
            )

        raw_wav_map: dict[int, bytes | None] = {}
        worker_count = int(os.getenv("FW_TTS_WORKERS", "1"))

        with tempfile.TemporaryDirectory() as synth_dir:
            def _do_synth(index: int, text: str, speaker_wav: str | None) -> tuple[int, bytes | None]:
                wav_path = str(pathlib.Path(synth_dir) / f"seg_{index}.wav")
                if not text or not text.strip():
                    return index, None
                try:
                    kwargs = {"speaker_wav": speaker_wav} if speaker_wav else {}
                    engine.tts_to_file(text=text, file_path=wav_path, **kwargs)
                    return index, pathlib.Path(wav_path).read_bytes()
                except Exception as exc:
                    print(f"[tts] TTS failed for segment ({exc}), using silence")
                    return index, None

            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        _do_synth,
                        meta["index"],
                        meta["text"],
                        voice_map.get(meta["speaker"]),
                    ): meta["index"]
                    for meta in seg_metas
                }
                for future in as_completed(futures):
                    index, raw_bytes = future.result()
                    raw_wav_map[index] = raw_bytes

        print(f" ({len(segments)} segments synthesized)", end="")

        with tempfile.TemporaryDirectory() as tmpdir:
            combined = AudioSegment.empty()
            cursor_ms = 0
            segment_details = []

            for meta in seg_metas:
                index = meta["index"]
                start_ms = int((meta["start"] + offset) * 1000)

                if start_ms > cursor_ms:
                    combined += AudioSegment.silent(duration=start_ms - cursor_ms)
                    cursor_ms = start_ms

                seg_audio, seg_speed_factor, seg_raw_duration = tts_engine._postprocess_segment(
                    raw_wav_map[index],
                    meta["target_sec"],
                    meta["stretch_factor"],
                    use_alignment,
                    tmpdir,
                )
                seg_audio = self._apply_speaker_tone(seg_audio, meta["speaker"])

                aligned_seg = meta["aligned_seg"]
                segment_details.append(
                    {
                        "index": index,
                        "text": meta["text"],
                        "speaker": meta["speaker"],
                        "speaker_wav": voice_map.get(meta["speaker"]),
                        "target_sec": round(meta["target_sec"], 3),
                        "stretch_factor": round(meta["stretch_factor"], 3),
                        "raw_duration_s": round(seg_raw_duration, 3),
                        "speed_factor": round(seg_speed_factor, 3),
                        "action": aligned_seg.action.value
                        if aligned_seg and hasattr(aligned_seg, "action")
                        else "unknown",
                    }
                )

                if seg_audio is not None:
                    combined += seg_audio
                    cursor_ms += len(seg_audio)

            save_path = pathlib.Path(output_path) / save_name
            combined.export(str(save_path), format="wav")

        stem = pathlib.Path(source_path).stem
        tts_engine._write_align_report(output_path, stem, metrics_list, aligned_list, segment_details)
        print("success!")

    @staticmethod
    def _speaker_index(speaker: str) -> int:
        """Extract a numeric index from labels like SPEAKER_00."""
        try:
            return max(0, int(str(speaker).rsplit("_", 1)[-1]))
        except (ValueError, TypeError):
            return 0

    @classmethod
    def _apply_speaker_tone(cls, segment_audio: AudioSegment | None, speaker: str) -> AudioSegment | None:
        """Adjust pitch by speaker index so higher indices sound deeper."""
        if segment_audio is None:
            return None

        speaker_index = cls._speaker_index(speaker)
        semitone_map = {
            0: 2.5,
            1: -1.5,
            2: -4.5,
        }
        semitones = semitone_map.get(speaker_index, max(-6.0, -4.5 - 1.2 * (speaker_index - 2)))
        if abs(semitones) < 0.01:
            return segment_audio

        pitch_factor = 2 ** (semitones / 12.0)
        shifted_rate = max(1000, int(segment_audio.frame_rate * pitch_factor))
        shifted = segment_audio._spawn(
            segment_audio.raw_data,
            overrides={"frame_rate": shifted_rate},
        ).set_frame_rate(segment_audio.frame_rate)
        target_len = len(segment_audio)
        if len(shifted) < target_len:
            shifted += AudioSegment.silent(duration=target_len - len(shifted))
        elif len(shifted) > target_len:
            shifted = shifted[:target_len]
        return shifted

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        """Find a title by scanning *search_dir* for JSON files."""
        for f in search_dir.glob("*.json"):
            return f.stem
        return None

    def compute_alignment(
        self,
        en_transcript: dict,
        es_transcript: dict,
        silence_regions: list[dict],
        max_stretch: float = 1.4,
    ) -> list:
        """Run global alignment over EN and ES transcripts.

        Returns list[AlignedSegment].  Combines compute_segment_metrics and
        global_align into a single facade call for use by the align router.
        """
        from foreign_whispers.alignment import compute_segment_metrics, global_align
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        return global_align(metrics, silence_regions, max_stretch)
