"""Metrics: per-result-set IR scores and the episode-level benchmark numbers."""

from __future__ import annotations

import pytest

from shots_to_success.grading import aggregate, grade_episode, score_result_set
from shots_to_success.types import (
    Attempt,
    Episode,
    Hit,
    Judgments,
    ResultSet,
    StopReason,
)

GRADES = {"a": 2, "b": 1, "c": 0, "d": 2}
JUDGMENTS = Judgments(
    grades={"q1": GRADES}, relevant_threshold=2, max_grade=2
)


def score(doc_ids, k=10):
    return score_result_set(doc_ids, GRADES, 2, 2, k)


def test_success_requires_a_document_at_or_above_the_threshold():
    # "b" is graded 1, below the threshold of 2, so it is not a success.
    assert score(["b", "c"])["success@10"] == 0.0
    assert score(["b", "a"])["success@10"] == 1.0


def test_metrics_of_a_perfect_and_an_empty_ranking():
    perfect = score(["a", "d", "b"])
    assert perfect["mrr@10"] == 1.0
    assert perfect["recall@10"] == 1.0
    assert perfect["first_relevant_rank"] == 1.0

    empty = score([])
    assert empty["success@10"] == 0.0
    assert empty["ndcg@10"] == 0.0
    assert empty["judged@10"] == 0.0


def test_cutoff_is_respected():
    ranking = ["c", "c", "c", "a"]
    assert score(ranking, k=3)["success@3"] == 0.0
    assert score(ranking, k=10)["success@10"] == 1.0


def test_judged_reports_pool_coverage():
    # Two of three results carry a judgment; the unjudged one counts as
    # non-relevant, and judged@k is how visible that assumption is.
    assert score(["a", "c", "unseen"])["judged@10"] == pytest.approx(2 / 3)


def test_unjudged_query_scores_zero_rather_than_crashing():
    assert score_result_set(["a"], {}, 2, 2, 10)["success@10"] == 0.0


# -- episode level ---------------------------------------------------------


def episode(*rankings, submitted=None, confident=None, nominated=None):
    ep = Episode(
        query_id="q1", query="q", condition="feedback", adapter="x", dataset="tiny"
    )
    for i, ranking in enumerate(rankings, start=1):
        ep.attempts.append(
            Attempt(
                shot=i,
                tool="search",
                tool_input={},
                result_set=ResultSet(
                    hits=[Hit(doc_id=d, score=1.0) for d in ranking]
                ),
            )
        )
    ep.submitted_shot = submitted if submitted is not None else len(rankings)
    ep.stop_reason = StopReason.SUBMITTED
    ep.agent_confident = confident
    if nominated is not None:
        ep.metrics["nominated_doc_ids"] = nominated
    return grade_episode(ep, JUDGMENTS, 10)


def test_shots_to_success_is_the_first_successful_attempt():
    ep = episode(["c"], ["b"], ["a"])
    assert ep.metrics["shots_to_success"] == 3
    assert ep.metrics["success"] is True
    assert ep.metrics["recovered"] is True


def test_a_failed_episode_has_no_shots_to_success():
    ep = episode(["c"], ["b"])
    assert ep.metrics["shots_to_success"] is None
    assert ep.metrics["success"] is False
    assert ep.metrics["recovered"] is False


def test_oversearch_counts_searches_after_the_need_was_met():
    ep = episode(["a"], ["c"], ["a"])
    assert ep.metrics["shots_to_success"] == 1
    assert ep.metrics["oversearched"] is True
    assert ep.metrics["oversearch_shots"] == 2


def test_a_retry_that_lowers_quality_is_a_bad_retry():
    ep = episode(["a", "d"], ["c"])
    assert ep.metrics["bad_retries"] == 1
    ep = episode(["c"], ["a"])
    assert ep.metrics["bad_retries"] == 0


def test_identical_retries_are_not_counted_as_bad():
    ep = episode(["a"], ["a"])
    assert ep.metrics["bad_retries"] == 0


def test_success_is_scored_on_the_submitted_attempt_not_the_best_one():
    # Attempt 1 succeeded but the agent submitted attempt 2, which did not.
    ep = episode(["a"], ["c"], submitted=2)
    assert ep.metrics["success_any"] is True
    assert ep.metrics["success"] is False


def test_stop_precision_scores_the_documents_the_agent_named():
    ep = episode(["a", "c"], nominated=["a", "c"])
    assert ep.metrics["stop_precision"] == 0.5
    assert episode(["a"], nominated=[]).metrics["stop_precision"] is None


def test_confidence_calibration_is_recorded():
    assert episode(["c"], confident=True).metrics["confidence_correct"] is False
    assert episode(["a"], confident=True).metrics["confidence_correct"] is True


# -- aggregation -----------------------------------------------------------


def test_aggregate_rolls_up_the_headline_numbers():
    episodes = [
        episode(["a"]),              # solved on shot 1
        episode(["c"], ["a"]),       # recovered on shot 2
        episode(["c"], ["c"], ["c"]),  # never solved
        episode(["a"], ["c"]),       # solved then wandered off, submitted a miss
    ]
    metrics = aggregate(episodes, k=10, max_shots=3)

    assert metrics["episodes"] == 4
    assert metrics["success_any@10"] == 0.75
    assert metrics["success@10"] == 0.5  # the wandering episode submitted a miss
    assert metrics["shots_to_success"] == pytest.approx((1 + 2 + 1) / 3)
    # Failures are charged max_shots + 1 in the censored variant.
    assert metrics["shots_to_success_censored"] == pytest.approx((1 + 2 + 1 + 4) / 4)
    assert metrics["recovery_rate"] == 0.5   # one of two first-attempt failures
    assert metrics["oversearch_rate"] == pytest.approx(1 / 3)


def test_aggregate_survives_an_empty_or_errored_run():
    assert aggregate([], k=10, max_shots=3)["episodes"] == 0
    broken = Episode(
        query_id="q1", query="q", condition="c", adapter="a", dataset="d"
    )
    broken.error = "boom"
    assert aggregate([broken], k=10, max_shots=3)["errored"] == 1
