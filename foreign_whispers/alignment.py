"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.
- ``global_align_dp`` — dynamic-programming search over stretch/gap choices
  that can recover segments the greedy threshold policy gives up on.

No external dependencies — stdlib only.
"""
import dataclasses
import math
import re
import unicodedata
from enum import Enum


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


_SYLLABLE_RATE = 4.5  # syllables per second for Romance languages


def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds using a pause-aware heuristic.

    The old estimate used only ``syllables / 4.5``.  That works tolerably for
    toy strings but underestimates longer utterances because it ignores:

    - extra articulation cost from additional word boundaries
    - clause-level pauses induced by punctuation

    We keep the syllable-rate term as the backbone so policy thresholds remain
    stable, then add small adjustments for multi-word phrasing and punctuation.
    """
    clean = text.strip()
    if not clean:
        return 0.0

    syllables = _count_syllables(clean)
    words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
    comma_like_pauses = len(re.findall(r"[,;:]", clean))
    strong_pauses = len(re.findall(r"[.!?]", clean))

    base_duration = syllables / _SYLLABLE_RATE
    word_boundary_overhead = max(0, len(words) - 2) * 0.04
    pause_overhead = comma_like_pauses * 0.10 + strong_pauses * 0.18

    return base_duration + word_boundary_overhead + pause_overhead


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def _silence_after(silence_regions: list[dict], end_s: float) -> float:
    """Return the first silence span that starts immediately after *end_s*."""
    for region in silence_regions:
        if region.get("label") == "silence" and region["start_s"] >= end_s - 0.1:
            return max(0.0, region["end_s"] - region["start_s"])
    return 0.0


def _candidate_gap_shifts(
    m: SegmentMetrics,
    available_gap_s: float,
    max_stretch: float,
) -> list[float]:
    """Return gap-shift candidates worth evaluating for one segment.

    The useful breakpoints are:

    - ``0``: no silence borrowed
    - enough silence to bring the segment within the requested stretch ceiling
    - enough silence to eliminate stretching entirely
    - the full local silence span
    """
    if available_gap_s <= 0:
        return [0.0]

    needed_for_max_stretch = max(0.0, (m.predicted_tts_s / max_stretch) - m.source_duration_s)
    needed_for_natural_fit = max(0.0, m.overflow_s)

    candidates = {0.0, round(min(available_gap_s, needed_for_max_stretch), 3)}
    candidates.add(round(min(available_gap_s, needed_for_natural_fit), 3))
    candidates.add(round(available_gap_s, 3))

    return sorted(candidate for candidate in candidates if candidate >= 0.0)


def _alignment_option(
    m: SegmentMetrics,
    gap_shift_s: float,
    max_stretch: float,
) -> tuple[AlignAction, float, tuple[int, int, int, int]]:
    """Score one alignment choice for dynamic programming.

    Returns the chosen action, the applied stretch factor, and a lexicographic
    cost tuple. Lower is better.
    """
    effective_duration = m.source_duration_s + gap_shift_s
    if effective_duration <= 0:
        effective_stretch = math.inf
    else:
        effective_stretch = m.predicted_tts_s / effective_duration

    if gap_shift_s > 0 and effective_stretch <= max_stretch:
        action = AlignAction.GAP_SHIFT
        stretch = 1.0 if effective_stretch <= 1.1 else effective_stretch
    elif effective_stretch <= 1.1:
        action = AlignAction.ACCEPT
        stretch = 1.0
    elif effective_stretch <= max_stretch:
        action = AlignAction.MILD_STRETCH
        stretch = effective_stretch
    elif effective_stretch <= 2.5:
        action = AlignAction.REQUEST_SHORTER
        stretch = 1.0
    else:
        action = AlignAction.FAIL
        stretch = 1.0

    cost = (
        1 if action == AlignAction.FAIL else 0,
        1 if stretch > 1.4 else 0,
        1 if action == AlignAction.REQUEST_SHORTER else 0,
        int(round(max(0.0, stretch - 1.0) * 1000)),
    )
    return action, stretch, cost


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(silence_regions, m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


def global_align_dp(
    metrics: list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch: float = 1.4,
) -> list[AlignedSegment]:
    """Search for a lower-cost alignment plan than ``global_align``.

    Unlike the greedy threshold policy, this optimizer can combine a partial
    gap shift with stretching. That lets it rescue segments that would
    otherwise fall into ``REQUEST_SHORTER`` when nearby silence is sufficient
    to bring the required stretch back within the same safe ceiling used by
    ``global_align``.

    The search is dynamic programming over segment prefixes. For each segment
    we evaluate a small set of gap-shift candidates and keep the lexicographically
    best plan under this objective:

    1. minimize ``FAIL`` segments
    2. minimize severe stretches (``stretch_factor > 1.4``)
    3. minimize cumulative drift beyond a 3.0s soft budget
    4. minimize ``REQUEST_SHORTER`` segments
    5. minimize total cumulative drift
    6. minimize the number of gap-shifts
    7. minimize residual stretch penalty

    By default ``max_stretch`` stays at ``1.4`` so the optimizer does not
    trade translation retries for more aggressive time-stretching. Callers can
    opt into a looser ceiling explicitly if they want to explore that tradeoff.
    """
    if not metrics:
        return []

    drift_budget_ms = 3000
    states: dict[int, tuple[tuple[int, int, int, int, int, int, int], list[tuple[AlignAction, float, float]]]] = {
        0: ((0, 0, 0, 0, 0, 0, 0), [])
    }

    for m in metrics:
        available_gap_s = _silence_after(silence_regions, m.source_end)
        next_states: dict[int, tuple[tuple[int, int, int, int, int, int, int], list[tuple[AlignAction, float, float]]]] = {}

        for drift_ms, (prefix_cost, prefix_plan) in states.items():
            for gap_shift_s in _candidate_gap_shifts(m, available_gap_s, max_stretch):
                action, stretch, option_cost = _alignment_option(m, gap_shift_s, max_stretch)
                next_drift_ms = drift_ms + int(round(gap_shift_s * 1000))
                transition_cost = (
                    option_cost[0],
                    option_cost[1],
                    max(0, next_drift_ms - drift_budget_ms) - max(0, drift_ms - drift_budget_ms),
                    option_cost[2],
                    next_drift_ms - drift_ms,
                    1 if gap_shift_s > 0 else 0,
                    option_cost[3],
                )
                total_cost = tuple(a + b for a, b in zip(prefix_cost, transition_cost))
                candidate_plan = prefix_plan + [(action, gap_shift_s, stretch)]

                incumbent = next_states.get(next_drift_ms)
                if incumbent is None or total_cost < incumbent[0]:
                    next_states[next_drift_ms] = (total_cost, candidate_plan)

        states = next_states

    _, best_plan = min(states.values(), key=lambda item: item[0])

    aligned: list[AlignedSegment] = []
    cumulative_drift = 0.0
    for m, (action, gap_shift, stretch) in zip(metrics, best_plan):
        sched_start = m.source_start + cumulative_drift
        sched_end = sched_start + m.source_duration_s + gap_shift
        aligned.append(AlignedSegment(
            index=m.index,
            original_start=m.source_start,
            original_end=m.source_end,
            scheduled_start=sched_start,
            scheduled_end=sched_end,
            text=m.translated_text,
            action=action,
            gap_shift_s=gap_shift,
            stretch_factor=stretch,
        ))
        cumulative_drift += gap_shift

    return aligned
