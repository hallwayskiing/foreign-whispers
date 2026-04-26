# tests/test_evaluation.py
from foreign_whispers.alignment import compute_segment_metrics, global_align, global_align_dp
from foreign_whispers.evaluation import clip_evaluation_report, dubbing_scorecard


def _make_transcripts(src_dur=3.0, tgt_chars=30):
    en = {"segments": [{"start": 0.0, "end": src_dur, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": src_dur, "text": "x" * tgt_chars}]}
    return en, es


def test_report_keys():
    en, es = _make_transcripts()
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert set(report.keys()) == {
        "mean_abs_duration_error_s",
        "pct_severe_stretch",
        "n_gap_shifts",
        "n_translation_retries",
        "total_cumulative_drift_s",
    }


def test_report_no_issues_for_easy_segment():
    en, es = _make_transcripts(src_dur=3.0, tgt_chars=15)  # 1s predicted, 3s budget
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert report["n_gap_shifts"] == 0
    assert report["n_translation_retries"] == 0
    assert report["total_cumulative_drift_s"] == 0.0


def test_report_counts_retries_for_hard_segment():
    # 1s budget, 9 syllables (ba*9) → ~2.0s predicted → REQUEST_SHORTER
    en = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": 1.0, "text": "ba" * 9}]}
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert report["n_translation_retries"] == 1


def test_report_empty_inputs():
    report = clip_evaluation_report([], [])
    assert report["mean_abs_duration_error_s"] == 0.0
    assert report["n_gap_shifts"] == 0


def test_report_counts_final_alignment_actions():
    en = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": 1.0, "text": "ba" * 9}]}
    silence = [{"start_s": 1.0, "end_s": 2.1, "label": "silence"}]

    metrics = compute_segment_metrics(en, es)
    aligned = global_align_dp(metrics, silence_regions=silence)
    report = clip_evaluation_report(metrics, aligned)

    assert report["n_translation_retries"] == 0
    assert report["n_gap_shifts"] == 1


def test_report_dp_without_silence_does_not_introduce_severe_stretch():
    en = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": 1.0, "text": "ba" * 9}]}

    metrics = compute_segment_metrics(en, es)
    aligned = global_align_dp(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)

    assert report["pct_severe_stretch"] == 0.0
    assert report["n_translation_retries"] == 1


def test_dubbing_scorecard_keys_and_bounds():
    en, es = _make_transcripts(src_dur=3.0, tgt_chars=15)
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    scorecard = dubbing_scorecard(metrics, aligned, report)

    assert set(scorecard.keys()) == {
        "timing_accuracy",
        "intelligibility",
        "semantic_fidelity",
        "naturalness",
        "overall_score",
    }
    assert all(0.0 <= value <= 1.0 for value in scorecard.values())


def test_dubbing_scorecard_penalizes_bad_alignment():
    good_en = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "Hello world"},
        {"start": 1.0, "end": 2.0, "text": "Nice to meet you"},
    ]}
    good_es = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "ba" * 4},
        {"start": 1.0, "end": 2.0, "text": "ba" * 4},
    ]}
    bad_en = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "This is a very long sentence"},
        {"start": 1.0, "end": 2.0, "text": "Another long sentence"},
    ]}
    bad_es = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "ba" * 18},
        {"start": 1.0, "end": 2.0, "text": "ba" * 16},
    ]}

    good_metrics = compute_segment_metrics(good_en, good_es)
    good_aligned = global_align(good_metrics, silence_regions=[])
    good_score = dubbing_scorecard(
        good_metrics,
        good_aligned,
        clip_evaluation_report(good_metrics, good_aligned),
    )

    bad_metrics = compute_segment_metrics(bad_en, bad_es)
    bad_aligned = global_align(bad_metrics, silence_regions=[])
    bad_score = dubbing_scorecard(
        bad_metrics,
        bad_aligned,
        clip_evaluation_report(bad_metrics, bad_aligned),
    )

    assert good_score["timing_accuracy"] > bad_score["timing_accuracy"]
    assert good_score["intelligibility"] > bad_score["intelligibility"]
    assert good_score["naturalness"] > bad_score["naturalness"]
    assert good_score["overall_score"] > bad_score["overall_score"]


def test_dubbing_scorecard_penalizes_aggressive_compression():
    en = {"segments": [{"start": 0.0, "end": 3.0, "text": "This sentence carries several important details"}]}
    es = {"segments": [{"start": 0.0, "end": 3.0, "text": "breve"}]}

    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    scorecard = dubbing_scorecard(
        metrics,
        aligned,
        clip_evaluation_report(metrics, aligned),
    )

    assert scorecard["semantic_fidelity"] < 0.8
