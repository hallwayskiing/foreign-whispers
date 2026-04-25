"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import difflib
import logging
import math
import os
import re
import unicodedata
from collections import Counter

import requests

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.
    """
    source = _normalize_text(source_text)
    baseline = _normalize_text(baseline_es)
    prev_context = _normalize_text(context_prev)
    next_context = _normalize_text(context_next)
    if not baseline:
        return []

    budget_chars = max(1, int(round(target_duration_s * 15.0)))
    candidates: dict[str, TranslationCandidate] = {}

    for candidate in _generate_rule_based_candidates(
        source,
        baseline,
        budget_chars,
        prev_context,
        next_context,
    ):
        candidates.setdefault(candidate.text, candidate)

    for candidate in _generate_multi_backend_candidates(
        source,
        baseline,
        budget_chars,
        prev_context,
        next_context,
    ):
        candidates.setdefault(candidate.text, candidate)

    for candidate in _generate_llm_candidates(
        source,
        baseline,
        budget_chars,
        prev_context,
        next_context,
    ):
        candidates.setdefault(candidate.text, candidate)

    ranked = [
        candidate
        for candidate in candidates.values()
        if candidate.text and len(candidate.text) < len(baseline)
    ]

    in_budget = [candidate for candidate in ranked if candidate.char_count <= budget_chars]
    if in_budget:
        ranked = in_budget

    ranked.sort(
        key=lambda candidate: _candidate_score(
            candidate.text,
            baseline,
            source,
            target_duration_s,
            prev_context,
            next_context,
        )
    )
    logger.info(
        "get_shorter_translations produced %d candidates for %.1fs budget (%d chars baseline).",
        len(ranked),
        target_duration_s,
        len(baseline),
    )
    return ranked


_SPANISH_COMPRESSION_MAP = {
    "en este momento": "ahora",
    "en ese momento": "entonces",
    "debido a que": "porque",
    "a fin de": "para",
    "con el fin de": "para",
    "de hecho": "",
    "la verdad es que": "",
    "es importante destacar que": "",
    "hay que tener en cuenta que": "",
    "por otra parte": "ademas",
    "por otro lado": "ademas",
    "sin embargo": "pero",
    "no obstante": "pero",
    "con respecto a": "sobre",
    "con relacion a": "sobre",
    "una gran cantidad de": "muchos",
    "un gran numero de": "muchos",
    "llevar a cabo": "hacer",
    "dar comienzo": "empezar",
}

_SPANISH_FILLERS = [
    "realmente",
    "basicamente",
    "practicamente",
    "en realidad",
    "por supuesto",
    "de alguna manera",
    "de algun modo",
]

_ENGLISH_COMPRESSION_MAP = {
    "right now": "now",
    "at this moment": "now",
    "in this moment": "now",
    "a large number of": "many",
    "a great deal of": "much",
    "carry out": "do",
    "in order to": "to",
    "due to the fact that": "because",
}

_ENGLISH_FILLERS = [
    "really",
    "actually",
    "basically",
    "just",
]

_SPANISH_STOPWORDS = {
    "a", "al", "ante", "con", "de", "del", "el", "en", "es", "la", "las",
    "lo", "los", "para", "por", "que", "se", "su", "sus", "un", "una", "y",
}


def _generate_rule_based_candidates(
    source_text: str,
    baseline_es: str,
    budget_chars: int,
    context_prev: str,
    context_next: str,
) -> list[TranslationCandidate]:
    candidates: list[TranslationCandidate] = []

    def add(text: str, rationale: str) -> None:
        final_text = _finalize_candidate(
            text,
            source_text=source_text,
            budget_chars=budget_chars,
            context_prev=context_prev,
            context_next=context_next,
        )
        if final_text and len(final_text) < len(baseline_es):
            candidates.append(
                _build_candidate(final_text, "rule-based", rationale)
            )

    light = _replace_phrases(baseline_es, _SPANISH_COMPRESSION_MAP)
    light = _strip_fillers(light, _SPANISH_FILLERS)
    add(light, "Rule-based truncation of verbose Spanish phrases.")

    medium = _drop_context_overlap(light, context_prev, context_next)
    add(medium, "Rule-based truncation plus overlap removal from adjacent segments.")

    return _dedupe_candidates(candidates)


def _generate_multi_backend_candidates(
    source_text: str,
    baseline_es: str,
    budget_chars: int,
    context_prev: str,
    context_next: str,
) -> list[TranslationCandidate]:
    candidates: list[TranslationCandidate] = []

    shortened_source = _shorten_english_source(source_text)
    if not shortened_source or shortened_source == source_text:
        return []

    argos_text = _translate_with_argos(shortened_source)
    if argos_text:
        final_argos = _finalize_candidate(
            argos_text,
            source_text=source_text,
            budget_chars=budget_chars,
            context_prev=context_prev,
            context_next=context_next,
        )
        candidates.append(
            _build_candidate(
                final_argos,
                "argos",
                "ArgosTranslate candidate from a rule-compressed English source.",
            )
        )

    marian_text = _translate_with_marian(shortened_source)
    if marian_text:
        final_marian = _finalize_candidate(
            marian_text,
            source_text=source_text,
            budget_chars=budget_chars,
            context_prev=context_prev,
            context_next=context_next,
        )
        candidates.append(
            _build_candidate(
                final_marian,
                "marian",
                "MarianMT candidate from a rule-compressed English source.",
            )
        )

    return _dedupe_candidates(
        [
            candidate
            for candidate in candidates
            if candidate.text and len(candidate.text) < len(baseline_es)
        ]
    )


def _generate_llm_candidates(
    source_text: str,
    baseline_es: str,
    budget_chars: int,
    context_prev: str,
    context_next: str,
) -> list[TranslationCandidate]:
    llm_text = _generate_with_local_llm(
        source_text=source_text,
        baseline_es=baseline_es,
        budget_chars=budget_chars,
        context_prev=context_prev,
        context_next=context_next,
    )
    if not llm_text:
        return []

    final_llm = _finalize_candidate(
        llm_text,
        source_text=source_text,
        budget_chars=budget_chars,
        context_prev=context_prev,
        context_next=context_next,
    )
    if not final_llm or len(final_llm) >= len(baseline_es):
        return []

    return [
        _build_candidate(
            final_llm,
            "llm",
            "Local LLM candidate tuned to the duration budget.",
        )
    ]


def _candidate_score(
    candidate_text: str,
    baseline_es: str,
    source_text: str,
    target_duration_s: float,
    context_prev: str,
    context_next: str,
) -> float:
    predicted_duration = len(candidate_text) / 15.0
    duration_error = (predicted_duration - target_duration_s) ** 2
    semantic_penalty = 1.0 - _semantic_similarity(candidate_text, baseline_es)
    context_penalty = _context_overlap_penalty(candidate_text, context_prev, context_next)
    source_penalty = 0.0 if _matches_source_punctuation(candidate_text, source_text) else 0.1
    over_budget_penalty = 0.0
    budget_chars = max(1, int(round(target_duration_s * 15.0)))
    if len(candidate_text) > budget_chars:
        over_budget_penalty = ((len(candidate_text) - budget_chars) / budget_chars) * 0.5
    return duration_error + (semantic_penalty * 0.9) + context_penalty + source_penalty + over_budget_penalty


def _semantic_similarity(a: str, b: str) -> float:
    a_tokens = _content_tokens(a)
    b_tokens = _content_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = sum((Counter(a_tokens) & Counter(b_tokens)).values())
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    token_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    seq_ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return (token_f1 * 0.7) + (seq_ratio * 0.3)


def _context_overlap_penalty(candidate_text: str, context_prev: str, context_next: str) -> float:
    cand_tokens = _content_tokens(candidate_text)
    if not cand_tokens:
        return 1.0
    prev_tokens = set(_content_tokens(context_prev))
    next_tokens = set(_content_tokens(context_next))
    overlap = sum(token in prev_tokens or token in next_tokens for token in cand_tokens)
    return (overlap / len(cand_tokens)) * 0.2


def _matches_source_punctuation(candidate_text: str, source_text: str) -> bool:
    if ("?" in source_text) != ("?" in candidate_text):
        return False
    if ("!" in source_text) != ("!" in candidate_text):
        return False
    return True


def _translate_with_argos(text: str) -> str:
    if not text:
        return ""
    try:
        import argostranslate.translate

        return _normalize_text(argostranslate.translate.translate(text, "en", "es"))
    except Exception as exc:
        logger.debug("ArgosTranslate candidate generation skipped: %s", exc)
        return ""


_MARIAN_CACHE: dict[str, object] = {}


def _translate_with_marian(text: str) -> str:
    if not text:
        return ""
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except Exception as exc:
        logger.debug("MarianMT unavailable: %s", exc)
        return ""

    try:
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer = _MARIAN_CACHE.get("tokenizer")
        model = _MARIAN_CACHE.get("model")
        if tokenizer is None or model is None:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            _MARIAN_CACHE["tokenizer"] = tokenizer
            _MARIAN_CACHE["model"] = model
        encoded = tokenizer([text], return_tensors="pt", truncation=True)
        generated = model.generate(**encoded, num_beams=4, max_new_tokens=96)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return _normalize_text(decoded[0] if decoded else "")
    except Exception as exc:
        logger.debug("MarianMT candidate generation failed: %s", exc)
        return ""


def _generate_with_local_llm(
    *,
    source_text: str,
    baseline_es: str,
    budget_chars: int,
    context_prev: str,
    context_next: str,
) -> str:
    base_url = _get_vllm_base_url()
    if not base_url:
        return ""

    prompt = (
        "Produce one shorter Spanish translation that preserves meaning.\n"
        f"Source: {source_text}\n"
        f"Baseline Spanish: {baseline_es}\n"
        f"Previous segment: {context_prev}\n"
        f"Next segment: {context_next}\n"
        f"Hard limit: {budget_chars} characters.\n"
        "Return only the Spanish text."
    )

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": min(128, max(24, math.ceil(budget_chars * 1.2))),
            },
            timeout=(3, 20),
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _normalize_text(content)
    except Exception as exc:
        logger.debug("Local LLM candidate generation failed: %s", exc)
        return ""


def _get_vllm_base_url() -> str:
    env_url = os.getenv("FW_VLLM_BASE_URL", "").rstrip("/")
    if env_url:
        return env_url
    try:
        from api.src.core.config import settings

        return settings.vllm_base_url.rstrip("/")
    except Exception:
        return ""


def _shorten_english_source(text: str) -> str:
    result = _normalize_text(text).lower()
    for source, target in _ENGLISH_COMPRESSION_MAP.items():
        result = re.sub(rf"\b{re.escape(source)}\b", target, result)
    result = _strip_fillers(result, _ENGLISH_FILLERS)
    return _normalize_text(result)


def _finalize_candidate(
    text: str,
    *,
    source_text: str,
    budget_chars: int,
    context_prev: str,
    context_next: str,
) -> str:
    text = _normalize_text(text)
    text = _drop_context_overlap(text, context_prev, context_next)
    text = _shape_candidate_to_source(text, source_text)
    return _normalize_text(text)


def _replace_phrases(text: str, replacements: dict[str, str]) -> str:
    result = text
    for source, target in replacements.items():
        result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.IGNORECASE)
    return _normalize_text(result)


def _strip_fillers(text: str, fillers: list[str]) -> str:
    result = text
    for filler in fillers:
        result = re.sub(
            rf"(?:,\s*)?\b{re.escape(filler)}\b(?:,\s*)?",
            " ",
            result,
            flags=re.IGNORECASE,
        )
    return _normalize_text(result)


def _drop_context_overlap(text: str, context_prev: str, context_next: str) -> str:
    words = _normalize_text(text).split()
    prev_words = _normalize_text(context_prev).lower().split()
    next_words = _normalize_text(context_next).lower().split()
    lowered = [word.lower() for word in words]

    for size in range(min(4, len(lowered), len(prev_words)), 1, -1):
        if lowered[:size] == prev_words[-size:]:
            words = words[size:]
            lowered = [word.lower() for word in words]
            break

    for size in range(min(4, len(lowered), len(next_words)), 1, -1):
        if lowered[-size:] == next_words[:size]:
            words = words[:-size]
            break

    return _normalize_text(" ".join(words))


def _shape_candidate_to_source(text: str, source_text: str) -> str:
    text = _normalize_text(text)
    if not text:
        return ""
    if "?" in source_text and "?" not in text:
        text = text.rstrip(".") + "?"
    if "!" in source_text and "!" not in text:
        text = text.rstrip(".") + "!"
    if "?" not in source_text:
        text = text.rstrip("?")
    if "!" not in source_text:
        text = text.rstrip("!")
    return _normalize_text(text)


def _dedupe_candidates(candidates: list[TranslationCandidate]) -> list[TranslationCandidate]:
    deduped: dict[str, TranslationCandidate] = {}
    for candidate in candidates:
        if not candidate.text:
            continue
        key = _canonical_candidate_key(candidate.text)
        existing = deduped.get(key)
        if existing is None or len(candidate.brevity_rationale) < len(existing.brevity_rationale):
            deduped[key] = candidate
    return list(deduped.values())


def _build_candidate(text: str, source: str, rationale: str) -> TranslationCandidate:
    return TranslationCandidate(
        text=text,
        char_count=len(text),
        brevity_rationale=f"[source:{source}] {rationale}",
    )


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?]){2,}", r"\1", text)
    return text.strip(" ,")


def _content_tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]+", text)]
    return [token for token in tokens if token not in _SPANISH_STOPWORDS]


def _canonical_candidate_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", _normalize_text(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized

