"""Scoring episodes against hidden relevance judgments.

Grading runs after the fact, on recorded episodes. Nothing here is reachable
from the agent loop, and re-scoring a finished run -- with a different
threshold or a different k -- never requires calling the model again.

Unjudged documents are treated as non-relevant, the standard pooled-judgment
assumption. ``judged@k`` is reported alongside every result so the cost of that
assumption stays visible.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

from .types import Episode, Judgments

#: Two attempts within this nDCG distance are treated as equal quality, so
#: floating-point noise is not counted as a bad retry.
QUALITY_EPSILON = 1e-9


# --------------------------------------------------------------------------
# Per-result-set IR metrics
# --------------------------------------------------------------------------


def score_result_set(
    doc_ids: Sequence[str],
    grades: dict[str, int],
    relevant_threshold: int,
    max_grade: int,
    k: int,
) -> dict[str, float]:
    """Standard IR metrics for one ranked list."""
    top = list(doc_ids[:k])
    relevant_total = sum(1 for g in grades.values() if g >= relevant_threshold)
    hits = [1 if grades.get(doc_id, 0) >= relevant_threshold else 0 for doc_id in top]

    first_rank = next((i + 1 for i, hit in enumerate(hits) if hit), 0)
    dcg = sum(
        (2 ** grades.get(doc_id, 0) - 1) / math.log2(i + 2)
        for i, doc_id in enumerate(top)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades)
    )
    judged = sum(1 for doc_id in top if doc_id in grades)

    return {
        f"success@{k}": float(any(hits)),
        f"ndcg@{k}": (dcg / idcg) if idcg > 0 else 0.0,
        f"recall@{k}": (sum(hits) / relevant_total) if relevant_total else 0.0,
        f"precision@{k}": (sum(hits) / len(top)) if top else 0.0,
        f"mrr@{k}": (1.0 / first_rank) if first_rank else 0.0,
        f"judged@{k}": (judged / len(top)) if top else 0.0,
        "first_relevant_rank": float(first_rank),
        "relevant_in_collection": float(relevant_total),
        "returned": float(len(top)),
    }


# --------------------------------------------------------------------------
# Per-episode metrics
# --------------------------------------------------------------------------


def grade_episode(episode: Episode, judgments: Judgments, k: int) -> Episode:
    """Score every attempt, then derive the episode-level benchmark metrics.

    Mutates and returns ``episode``.
    """
    grades = judgments.for_query(episode.query_id)
    threshold = judgments.relevant_threshold
    relevant = judgments.relevant_docs(episode.query_id)

    for attempt in episode.attempts:
        attempt.scores = score_result_set(
            attempt.result_set.doc_ids, grades, threshold, judgments.max_grade, k
        )

    success_flags = [bool(a.scores[f"success@{k}"]) for a in episode.attempts]
    quality = [a.scores[f"ndcg@{k}"] for a in episode.attempts]
    shots_used = len(episode.attempts)

    first_success = next(
        (i + 1 for i, flag in enumerate(success_flags) if flag), None
    )
    submitted_scores = (
        episode.attempts[episode.submitted_shot - 1].scores
        if episode.submitted_shot
        else {}
    )

    # Retries: transitions between consecutive attempts.
    bad_retries = sum(
        1
        for i in range(1, shots_used)
        if quality[i] < quality[i - 1] - QUALITY_EPSILON
    )

    metrics: dict[str, Any] = dict(episode.metrics)
    metrics.update(
        {
            "k": k,
            "relevant_threshold": threshold,
            "shots_used": shots_used,
            # Primary: did the *submitted* result set contain a relevant doc.
            "success": bool(submitted_scores.get(f"success@{k}", 0.0)),
            # Trajectory: did any attempt succeed, whether or not it was kept.
            "success_any": bool(first_success),
            "shots_to_success": first_success,
            "first_attempt_success": bool(success_flags[0]) if success_flags else False,
            # Among episodes whose first attempt failed, did a later one land.
            "recovered": (
                bool(first_success and first_success > 1)
                if success_flags and not success_flags[0]
                else None
            ),
            "retries": max(0, shots_used - 1),
            "bad_retries": bad_retries,
            # Searches spent after the need was already satisfied.
            "oversearch_shots": (
                shots_used - first_success if first_success else 0
            ),
            "oversearched": bool(first_success and shots_used > first_success),
            "quality_trajectory": [round(q, 4) for q in quality],
            "stop_reason": episode.stop_reason.value,
            "agent_confident": episode.agent_confident,
        }
    )
    for name, value in submitted_scores.items():
        metrics[f"submitted_{name}"] = value

    # Calibration: was the agent right that it had found something.
    if episode.agent_confident is not None:
        metrics["confidence_correct"] = episode.agent_confident == metrics["success"]

    # Self-assessment: of the documents the agent named as satisfying the need,
    # how many actually were.
    nominated = [str(d) for d in metrics.get("nominated_doc_ids", [])]
    if nominated:
        correct = sum(1 for doc_id in nominated if doc_id in relevant)
        metrics["stop_precision"] = correct / len(nominated)
    else:
        metrics["stop_precision"] = None

    episode.metrics = metrics
    return episode


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(episodes: Iterable[Episode], k: int, max_shots: int) -> dict[str, Any]:
    """Roll graded episodes up into the benchmark's reported numbers."""
    eps = [e for e in episodes if e.error is None and e.attempts]
    errored = [e for e in episodes if e.error is not None]
    total = len(eps)
    if not total:
        return {"episodes": 0, "errored": len(errored)}

    def mean(values: Sequence[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    successes = [e for e in eps if e.metrics["success_any"]]
    shots_to_success = [e.metrics["shots_to_success"] for e in successes]
    first_failed = [e for e in eps if not e.metrics["first_attempt_success"]]
    recovered = [e for e in first_failed if e.metrics["success_any"]]
    retries = sum(e.metrics["retries"] for e in eps)
    bad_retries = sum(e.metrics["bad_retries"] for e in eps)
    stop_precisions = [
        e.metrics["stop_precision"] for e in eps if e.metrics["stop_precision"] is not None
    ]
    confidences = [
        e.metrics["confidence_correct"] for e in eps if "confidence_correct" in e.metrics
    ]

    solved_at = {
        shot: sum(1 for s in shots_to_success if s == shot) / total
        for shot in range(1, max_shots + 1)
    }

    return {
        "episodes": total,
        "errored": len(errored),
        # -- primary metric -------------------------------------------------
        # Mean shots to the first attempt that satisfied Success@k, over
        # episodes that ever succeeded. Read it next to success_rate: a system
        # that only solves easy queries gets a flattering mean.
        "shots_to_success": mean(shots_to_success),
        # Same number with failures charged max_shots + 1, so one scalar orders
        # systems without hiding the failures.
        "shots_to_success_censored": mean(
            shots_to_success + [max_shots + 1] * (total - len(successes))
        ),
        "shots_to_success_median": (
            statistics.median(shots_to_success) if shots_to_success else None
        ),
        "solved_at_shot": solved_at,
        # -- guardrails -----------------------------------------------------
        f"success@{k}": mean([float(e.metrics["success"]) for e in eps]),
        f"success_any@{k}": len(successes) / total,
        "recovery_rate": (len(recovered) / len(first_failed)) if first_failed else None,
        "bad_retry_rate": (bad_retries / retries) if retries else None,
        "oversearch_rate": (
            sum(1 for e in successes if e.metrics["oversearched"]) / len(successes)
            if successes
            else None
        ),
        # -- diagnostics ----------------------------------------------------
        "mean_shots_used": mean([float(e.metrics["shots_used"]) for e in eps]),
        f"ndcg@{k}": mean([e.metrics.get(f"submitted_ndcg@{k}", 0.0) for e in eps]),
        f"recall@{k}": mean([e.metrics.get(f"submitted_recall@{k}", 0.0) for e in eps]),
        f"mrr@{k}": mean([e.metrics.get(f"submitted_mrr@{k}", 0.0) for e in eps]),
        f"judged@{k}": mean([e.metrics.get(f"submitted_judged@{k}", 0.0) for e in eps]),
        "stop_precision": mean(stop_precisions),
        "confidence_accuracy": mean([float(c) for c in confidences]),
        "aux_calls_per_episode": mean([float(len(e.aux_calls)) for e in eps]),
        "stop_reasons": _counts(e.stop_reason.value for e in eps),
        "tokens": _token_totals(eps),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _token_totals(episodes: Sequence[Episode]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for episode in episodes:
        for key, value in episode.usage.items():
            totals[key] = totals.get(key, 0) + value
    return totals


__all__ = ["QUALITY_EPSILON", "aggregate", "grade_episode", "score_result_set"]
