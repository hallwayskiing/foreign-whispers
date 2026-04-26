"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
)


def _clamp01(value: float) -> float:
    """Clamp *value* to the inclusive ``[0, 1]`` interval."""
    return max(0.0, min(1.0, value))


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    """Return the arithmetic mean, or *default* for empty input."""
    return _stats.mean(values) if values else default


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for a in aligned if a.action == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Return a normalized multi-dimensional dubbing quality scorecard.

    The notebook task suggests richer signals such as TTS→STT round-trip WER
    and embedding similarity. This library stays dependency-light, so the
    scorecard uses stable offline proxies derived from the alignment pipeline:

    - ``timing_accuracy``: combines duration error, severe stretch rate, and drift
    - ``intelligibility``: penalizes aggressive stretch, retries, and failures
    - ``semantic_fidelity``: penalizes retries/failures and over-compressed text
    - ``naturalness``: rewards consistent pacing across segments

    Each dimension is normalized to ``[0, 1]`` where ``1`` is best.
    ``overall_score`` is the arithmetic mean of the four dimensions.
    """
    if align_report is None:
        align_report = clip_evaluation_report(metrics, aligned)

    if not metrics:
        empty = {
            "timing_accuracy": 0.0,
            "intelligibility": 0.0,
            "semantic_fidelity": 0.0,
            "naturalness": 0.0,
            "overall_score": 0.0,
        }
        return empty

    mean_err = float(align_report.get("mean_abs_duration_error_s", 0.0))
    pct_severe = float(align_report.get("pct_severe_stretch", 0.0))
    drift = abs(float(align_report.get("total_cumulative_drift_s", 0.0)))
    timing_penalty = (
        0.5 * _clamp01(mean_err / 1.5) +
        0.3 * _clamp01(pct_severe / 25.0) +
        0.2 * _clamp01(drift / 3.0)
    )
    timing_accuracy = _clamp01(1.0 - timing_penalty)

    intelligibility_scores = []
    semantic_scores = []
    pacing_ratios = []

    for metric, segment in zip(metrics, aligned):
        duration_budget = max(
            segment.scheduled_end - segment.scheduled_start,
            metric.source_duration_s,
            0.001,
        )
        realized_stretch = (
            metric.predicted_tts_s / duration_budget
            if metric.predicted_tts_s > 0 else 1.0
        )
        pacing_ratios.append(realized_stretch)

        if segment.action == AlignAction.FAIL:
            intelligibility = 0.0
        elif segment.action == AlignAction.REQUEST_SHORTER:
            intelligibility = 0.35
        elif realized_stretch <= 1.1:
            intelligibility = 1.0
        elif realized_stretch <= 1.4:
            intelligibility = 1.0 - ((realized_stretch - 1.1) / 0.3) * 0.25
        else:
            intelligibility = 0.75 - min(1.0, (realized_stretch - 1.4) / 0.8) * 0.75
        intelligibility_scores.append(_clamp01(intelligibility))

        compression_ratio = (
            metric.tgt_char_count / max(metric.src_char_count, 1)
        )
        compression_penalty = _clamp01((0.65 - compression_ratio) / 0.65)
        semantic = 1.0 - (0.55 * compression_penalty)
        if segment.action == AlignAction.REQUEST_SHORTER:
            semantic -= 0.25
        elif segment.action == AlignAction.FAIL:
            semantic -= 0.5
        semantic_scores.append(_clamp01(semantic))

    intelligibility_score = _clamp01(_safe_mean(intelligibility_scores))
    semantic_fidelity = _clamp01(_safe_mean(semantic_scores))

    pacing_mean = _safe_mean(pacing_ratios, default=1.0)
    pacing_std = _stats.pstdev(pacing_ratios) if len(pacing_ratios) > 1 else 0.0
    pacing_mad = _safe_mean([abs(ratio - 1.0) for ratio in pacing_ratios])
    naturalness_penalty = (
        0.6 * _clamp01(pacing_mad / 0.35) +
        0.4 * _clamp01(pacing_std / max(0.15, pacing_mean * 0.2))
    )
    naturalness = _clamp01(1.0 - naturalness_penalty)

    overall_score = _safe_mean([
        timing_accuracy,
        intelligibility_score,
        semantic_fidelity,
        naturalness,
    ])

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "intelligibility": round(intelligibility_score, 3),
        "semantic_fidelity": round(semantic_fidelity, 3),
        "naturalness": round(naturalness, 3),
        "overall_score": round(overall_score, 3),
    }
