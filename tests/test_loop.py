"""The multi-shot loop: budget, stopping, and what counts as a shot."""

from __future__ import annotations

import pytest
from conftest import FakeClient, call, text, turn

from shots_to_success.agent.loop import EpisodeRunner, select_tools
from shots_to_success.config import CONDITIONS, RunConfig
from shots_to_success.types import Query, StopReason

QUERY = Query(query_id="q1", text="red chair")


def runner(adapter, script, condition="results_only", **overrides):
    config = RunConfig(
        dataset="tiny", adapter="bm25", condition=condition, max_shots=3, **overrides
    )
    return EpisodeRunner(
        FakeClient(script),
        adapter,
        config.condition_spec(),
        config,
        "A document is relevant if it is the item asked for.",
    )


def test_agent_stops_after_one_good_search(adapter):
    episode = runner(
        adapter,
        [
            turn(call("search", query="red chair")),
            turn(call("submit_results", satisfying_doc_ids=["1"], reason="found it", confident=True)),
        ],
    ).run(QUERY)

    assert episode.stop_reason is StopReason.SUBMITTED
    assert episode.shots_used == 1
    assert episode.submitted_shot == 1
    assert episode.agent_confident is True
    assert episode.metrics["nominated_doc_ids"] == ["1"]


def test_budget_is_enforced_and_last_attempt_is_auto_submitted(adapter):
    script = [turn(call("search", query=f"attempt {i}")) for i in range(4)]
    episode = runner(adapter, script).run(QUERY)

    # Three searches land; the fourth is refused rather than silently allowed.
    assert episode.shots_used == 3
    assert episode.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert episode.submitted_shot == 3
    refused = episode.transcript[3]["tool_calls"][0]["result"]
    assert "budget exhausted" in refused


def test_submitting_before_searching_is_refused_not_fatal(adapter):
    episode = runner(
        adapter,
        [
            turn(call("submit_results", satisfying_doc_ids=[], reason="giving up", confident=False)),
            turn(call("search", query="red chair")),
            turn(call("submit_results", satisfying_doc_ids=["1"], reason="ok", confident=True)),
        ],
    ).run(QUERY)

    assert "have not run a search yet" in episode.transcript[0]["tool_calls"][0]["result"]
    assert episode.stop_reason is StopReason.SUBMITTED
    assert episode.shots_used == 1


def test_ending_the_turn_without_submitting_still_submits_the_last_search(adapter):
    episode = runner(
        adapter,
        [
            turn(call("search", query="red chair")),
            turn(text("I think that is enough."), stop="end_turn"),
        ],
    ).run(QUERY)

    assert episode.stop_reason is StopReason.ENDED_TURN
    assert episode.submitted_shot == 1


def test_never_searching_is_recorded_as_such(adapter):
    episode = runner(adapter, [turn(text("I cannot help."), stop="end_turn")]).run(QUERY)

    assert episode.stop_reason is StopReason.NO_SEARCH
    assert episode.submitted_shot is None
    assert episode.shots_used == 0


def test_parallel_searches_in_one_turn_each_cost_a_shot(adapter):
    episode = runner(
        adapter,
        [
            turn(
                call("search", block_id="a", query="red chair"),
                call("search", block_id="b", query="crimson chair"),
            ),
            turn(call("submit_results", satisfying_doc_ids=["1"], reason="ok", confident=True)),
        ],
    ).run(QUERY)

    assert episode.shots_used == 2
    assert [a.shot for a in episode.attempts] == [1, 2]


def test_auxiliary_calls_do_not_spend_the_budget(adapter):
    episode = runner(
        adapter,
        [
            turn(call("autocomplete", prefix="red")),
            turn(call("facets", query="chair", field="product_class")),
            turn(call("search", query="red chair")),
            turn(call("submit_results", satisfying_doc_ids=["1"], reason="ok", confident=True)),
        ],
        condition="tooling",
    ).run(QUERY)

    assert episode.shots_used == 1
    assert [c.tool for c in episode.aux_calls] == ["autocomplete", "facets"]


def test_a_tool_that_raises_is_reported_to_the_agent_not_the_run(adapter):
    broken = next(spec for spec in adapter.tools() if spec.name == "search")
    original = broken.handler
    try:
        object.__setattr__(
            adapter, "tools", lambda: [_boom(spec) for spec in [broken]]
        )
        episode = runner(
            adapter,
            [turn(call("search", query="x")), turn(text("giving up"), stop="end_turn")],
        ).run(QUERY)
        assert episode.error is None
        assert "failed" in episode.transcript[0]["tool_calls"][0]["result"]
    finally:
        broken.handler = original


def _boom(spec):
    import dataclasses

    def explode(_args):
        raise RuntimeError("backend down")

    return dataclasses.replace(spec, handler=explode)


def test_max_turns_stops_a_runaway_agent(adapter):
    script = [turn(call("autocomplete", prefix="r")) for _ in range(50)]
    episode = runner(adapter, script, condition="tooling", max_turns=5).run(QUERY)

    assert episode.stop_reason is StopReason.MAX_TURNS
    assert episode.turns == 5


def test_usage_is_accumulated_across_turns(adapter):
    episode = runner(
        adapter,
        [
            turn(call("search", query="red chair")),
            turn(call("submit_results", satisfying_doc_ids=["1"], reason="ok", confident=True)),
        ],
    ).run(QUERY)

    assert episode.usage["input_tokens"] == 20
    assert episode.usage["output_tokens"] == 10


# -- condition wiring ------------------------------------------------------


def test_feedback_track_conditions_share_one_tool_contract(adapter):
    results_only = {s.name for s in select_tools(adapter, CONDITIONS["results_only"])}
    feedback = {s.name for s in select_tools(adapter, CONDITIONS["feedback"])}
    assert results_only == feedback == {"search"}


def test_tooling_condition_opens_the_tool_contract(adapter):
    names = {s.name for s in select_tools(adapter, CONDITIONS["tooling"])}
    assert names == {"search", "autocomplete", "facets", "describe_index", "explain_result"}


def test_diagnostics_appear_only_under_the_feedback_condition(adapter):
    scripts = lambda: [  # noqa: E731
        turn(call("search", query="red chair")),
        turn(call("submit_results", satisfying_doc_ids=["1"], reason="ok", confident=True)),
    ]
    plain = runner(adapter, scripts(), condition="results_only").run(QUERY)
    rich = runner(adapter, scripts(), condition="feedback").run(QUERY)

    assert "Retrieval diagnostics" not in plain.transcript[0]["tool_calls"][0]["result"]
    assert "Retrieval diagnostics" in rich.transcript[0]["tool_calls"][0]["result"]
    assert plain.attempts[0].diagnostics_shown is False
    assert rich.attempts[0].diagnostics_shown is True


def test_feedback_track_prompts_are_byte_identical(adapter):
    plain = runner(adapter, [], condition="results_only")
    rich = runner(adapter, [], condition="feedback")
    assert plain.system_prompt == rich.system_prompt


def test_the_loop_cannot_see_judgments():
    """The agent path must not import the grader or the judgment type. This is
    the property that makes the hidden labels hidden, so it is asserted rather
    than assumed -- an import added here would silently break the benchmark."""
    import ast
    import inspect

    from shots_to_success.agent import loop, prompts

    forbidden = {"grading", "Judgments"}
    for module in (loop, prompts):
        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[-1])
                imported.update(alias.name for alias in node.names)
        assert not (imported & forbidden), (
            f"{module.__name__} imports {imported & forbidden}"
        )


# -- fatal vs recoverable failures -----------------------------------------


class _ExplodingMessages:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def create(self, **_kwargs):
        raise self.exc


class _ExplodingClient:
    def __init__(self, exc: Exception) -> None:
        self.messages = _ExplodingMessages(exc)


def _run_against(adapter, exc):
    from shots_to_success.agent.loop import EpisodeRunner

    config = RunConfig(dataset="tiny", adapter="bm25", condition="results_only")
    return EpisodeRunner(
        _ExplodingClient(exc), adapter, config.condition_spec(), config, "rubric"
    ).run(QUERY)


def test_missing_credentials_aborts_the_run_rather_than_failing_every_episode(adapter):
    from shots_to_success.agent.loop import FatalRunError

    exc = TypeError("Could not resolve authentication method. Expected one of api_key")
    with pytest.raises(FatalRunError, match="No Anthropic credentials"):
        _run_against(adapter, exc)


def test_an_unrelated_error_fails_only_that_episode(adapter):
    episode = _run_against(adapter, ValueError("something odd"))
    assert episode.stop_reason is StopReason.ERROR
    assert "something odd" in episode.error
