# tests/test_agents.py — renamed module is now foreign_whispers.reranking
from foreign_whispers.reranking import (
    get_shorter_translations,
    analyze_failures,
    TranslationCandidate,
    FailureAnalysis,
)


def test_get_shorter_translations_returns_shorter_candidates():
    result = get_shorter_translations(
        "The military wants to carry out a large number of operations right now.",
        "Los militares quieren llevar a cabo una gran cantidad de operaciones en este momento.",
        3.0,
    )
    assert result
    assert all(isinstance(candidate, TranslationCandidate) for candidate in result)
    assert all(len(candidate.text) < len("Los militares quieren llevar a cabo una gran cantidad de operaciones en este momento.") for candidate in result)


def test_analyze_failures_returns_dataclass():
    result = analyze_failures({"mean_abs_duration_error_s": 0.5})
    assert isinstance(result, FailureAnalysis)
    assert result.failure_category == "ok"


def test_analyze_failures_detects_overflow():
    result = analyze_failures({"pct_severe_stretch": 30})
    assert result.failure_category == "duration_overflow"


def test_analyze_failures_detects_drift():
    result = analyze_failures({"total_cumulative_drift_s": 5.0})
    assert result.failure_category == "cumulative_drift"


def test_analyze_failures_detects_stretch_quality():
    result = analyze_failures({"mean_abs_duration_error_s": 1.2})
    assert result.failure_category == "stretch_quality"
