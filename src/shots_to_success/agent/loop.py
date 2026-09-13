"""The multi-shot search loop.

One episode = one information need, one agent, one search budget. The agent
searches, reads results, and decides after each attempt whether to stop or
retry. It stops by calling ``submit_results``; if it burns the whole budget
without stopping, the last attempt is submitted for it.

This module deliberately has no access to relevance judgments. It cannot import
them, is never handed them, and its output -- an :class:`Episode` -- is graded
afterwards by :mod:`shots_to_success.grading`. That separation is what makes
the hidden labels actually hidden.
"""

from __future__ import annotations

import random
import time
from typing import Any, Mapping

from ..adapters.base import SearchAdapter
from ..config import Condition, RunConfig
from ..types import (
    Attempt,
    AuxCall,
    Episode,
    Query,
    StopReason,
    ToolKind,
    ToolSpec,
)
from .prompts import (
    SUBMIT_TOOL,
    render_information_need,
    render_system_prompt,
    render_tool_result,
    system_prompt_sha,
)

SUBMIT = SUBMIT_TOOL["name"]


class FatalRunError(RuntimeError):
    """A failure that would repeat on every episode, so the run should stop.

    Bad credentials, a model the account cannot reach, a revoked key. Recording
    these as hundreds of individually failed episodes wastes time and buries
    the one line that says what is wrong.
    """



def select_tools(adapter: SearchAdapter, condition: Condition) -> list[ToolSpec]:
    """Which of the adapter's tools this condition exposes.

    Retrieval tools are always available. Auxiliary tools require
    ``auxiliary_tools`` -- widening that moves a run into the Tooling Track.
    Tools marked ``feedback_only`` additionally require feedback to be on.
    """
    selected: list[ToolSpec] = []
    for spec in adapter.tools():
        if spec.kind is ToolKind.RETRIEVAL:
            selected.append(spec)
            continue
        if not condition.auxiliary_tools:
            continue
        if spec.feedback_only and not condition.feedback:
            continue
        selected.append(spec)
    return selected


class EpisodeRunner:
    """Drives one agent through one information need."""

    def __init__(
        self,
        client: Any,
        adapter: SearchAdapter,
        condition: Condition,
        config: RunConfig,
        rubric: str,
    ) -> None:
        self.client = client
        self.adapter = adapter
        self.condition = condition
        self.config = config
        self.tool_specs = select_tools(adapter, condition)
        self.tools_by_name = {spec.name: spec for spec in self.tool_specs}
        self.api_tools = [spec.to_api_tool() for spec in self.tool_specs] + [SUBMIT_TOOL]
        self.system_prompt = render_system_prompt(condition, config.k, rubric)
        self.system_prompt_sha = system_prompt_sha(self.system_prompt)

    # -- API ------------------------------------------------------------

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            # Cache breakpoint on the system block covers tools + system, which
            # are byte-identical across every episode in a run.
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": self.config.effort},
            tools=self.api_tools,
            messages=messages,
        )

    def _create_with_retry(self, messages: list[dict[str, Any]]) -> Any:
        import anthropic

        delay = 2.0
        last: Exception | None = None
        for _ in range(max(1, self.config.max_retries)):
            try:
                return self._create(messages)
            except TypeError as exc:
                # The SDK raises TypeError, not an APIError, when it cannot
                # resolve credentials at all.
                if "authentication" in str(exc).lower():
                    raise FatalRunError(
                        "No Anthropic credentials found. Set ANTHROPIC_API_KEY "
                        "or sign in with `ant auth login`."
                    ) from exc
                raise
            except (
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
            ) as exc:
                raise FatalRunError(
                    f"Anthropic API rejected the credentials: {exc}"
                ) from exc
            except anthropic.NotFoundError as exc:
                raise FatalRunError(
                    f"Model {self.config.model!r} was not found for this "
                    f"account: {exc}"
                ) from exc
            except (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ) as exc:
                last = exc
                time.sleep(delay + random.random())
                delay = min(delay * 2, 30.0)
            except anthropic.APIStatusError as exc:
                if exc.status_code < 500:
                    raise
                last = exc
                time.sleep(delay + random.random())
                delay = min(delay * 2, 30.0)
        assert last is not None
        raise last

    # -- episode ---------------------------------------------------------

    def run(self, query: Query) -> Episode:
        episode = Episode(
            query_id=query.query_id,
            query=query.text,
            condition=self.condition.name,
            adapter=self.adapter.name,
            dataset=self.config.dataset,
        )
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": render_information_need(query)}
        ]
        try:
            self._drive(episode, messages)
        except FatalRunError:
            # Credentials, permissions, a bad model name: every remaining
            # episode would fail the same way, so stop the run instead of
            # recording hundreds of identical failures.
            raise
        except Exception as exc:  # noqa: BLE001 - one bad episode must not kill a run
            episode.error = f"{type(exc).__name__}: {exc}"
            episode.stop_reason = StopReason.ERROR
        episode.wall_ms = (time.perf_counter() - started) * 1000
        self._finalize(episode)
        return episode

    def _drive(self, episode: Episode, messages: list[dict[str, Any]]) -> None:
        while episode.turns < self.config.max_turns:
            response = self._create_with_retry(messages)
            episode.turns += 1
            _accumulate_usage(episode, response)

            if response.stop_reason == "refusal":
                episode.error = "model refused"
                episode.stop_reason = StopReason.ERROR
                return

            assistant_text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})

            if not tool_uses:
                episode.transcript.append(
                    {"turn": episode.turns, "assistant": assistant_text}
                )
                episode.stop_reason = (
                    StopReason.ENDED_TURN if episode.attempts else StopReason.NO_SEARCH
                )
                episode.stop_rationale = assistant_text
                return

            tool_results: list[dict[str, Any]] = []
            calls: list[dict[str, Any]] = []
            stop = False
            for block in tool_uses:
                text, is_error, submitted = self._dispatch(episode, block)
                calls.append({"tool": block.name, "input": dict(block.input), "result": text})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": text,
                        **({"is_error": True} if is_error else {}),
                    }
                )
                stop = stop or submitted

            episode.transcript.append(
                {"turn": episode.turns, "assistant": assistant_text, "tool_calls": calls}
            )
            messages.append({"role": "user", "content": tool_results})
            if stop:
                return

        episode.stop_reason = StopReason.MAX_TURNS

    def _dispatch(self, episode: Episode, block: Any) -> tuple[str, bool, bool]:
        """Run one tool call. Returns ``(result_text, is_error, should_stop)``."""
        name = block.name
        args: Mapping[str, Any] = dict(block.input or {})

        if name == SUBMIT:
            if not episode.attempts:
                return (
                    "Error: you have not run a search yet. Run at least one "
                    "search before submitting.",
                    True,
                    False,
                )
            episode.stop_reason = StopReason.SUBMITTED
            episode.stop_rationale = str(args.get("reason") or "")
            episode.agent_confident = bool(args.get("confident"))
            episode.submitted_shot = len(episode.attempts)
            nominated = args.get("satisfying_doc_ids") or []
            episode.metrics["nominated_doc_ids"] = [str(d) for d in nominated]
            return "Submitted. The episode is complete.", False, True

        spec = self.tools_by_name.get(name)
        if spec is None:
            return f"Error: unknown tool {name!r}.", True, False

        if spec.kind is ToolKind.RETRIEVAL and episode.shots_used >= self.condition.max_shots:
            return (
                f"Error: search budget exhausted "
                f"({episode.shots_used} of {self.condition.max_shots} used). "
                "Call submit_results now to submit your last search.",
                True,
                False,
            )

        started = time.perf_counter()
        try:
            response = spec.handler(args)
        except Exception as exc:  # noqa: BLE001 - a broken tool is the adapter's bug, not the run's
            return f"Error: tool {name!r} failed: {type(exc).__name__}: {exc}", True, False
        latency_ms = (time.perf_counter() - started) * 1000

        if spec.kind is ToolKind.RETRIEVAL:
            if response.error is None and response.result_set is None:
                return (
                    f"Error: tool {name!r} returned no result set.",
                    True,
                    False,
                )
            if response.result_set is not None:
                episode.attempts.append(
                    Attempt(
                        shot=len(episode.attempts) + 1,
                        tool=name,
                        tool_input=dict(args),
                        result_set=response.result_set,
                        latency_ms=latency_ms,
                        diagnostics_shown=bool(
                            self.condition.feedback and response.diagnostics
                        ),
                    )
                )
        else:
            episode.aux_calls.append(
                AuxCall(
                    tool=name,
                    tool_input=dict(args),
                    latency_ms=latency_ms,
                    after_shot=episode.shots_used,
                )
            )

        return (
            render_tool_result(response, self.condition),
            response.error is not None,
            False,
        )

    def _finalize(self, episode: Episode) -> None:
        if episode.submitted_shot is None and episode.attempts:
            episode.submitted_shot = len(episode.attempts)
            if episode.stop_reason in (StopReason.ERROR, StopReason.MAX_TURNS):
                pass  # keep the real failure reason
            elif episode.shots_used >= self.condition.max_shots:
                episode.stop_reason = StopReason.BUDGET_EXHAUSTED
        if not episode.attempts and episode.stop_reason is StopReason.ERROR and not episode.error:
            episode.stop_reason = StopReason.NO_SEARCH


def run_static_episode(
    adapter: SearchAdapter,
    condition: Condition,
    config: RunConfig,
    query: Query,
) -> Episode:
    """The static baseline: run the original query once, no model involved."""
    episode = Episode(
        query_id=query.query_id,
        query=query.text,
        condition=condition.name,
        adapter=adapter.name,
        dataset=config.dataset,
    )
    started = time.perf_counter()
    try:
        result_set = adapter.baseline(query.text, config.k)
        episode.attempts.append(
            Attempt(
                shot=1,
                tool=adapter.primary_tool().name,
                tool_input={"query": query.text, "k": config.k},
                result_set=result_set,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
        episode.submitted_shot = 1
        episode.stop_reason = StopReason.SUBMITTED
        episode.stop_rationale = "static baseline: single unmodified query"
    except Exception as exc:  # noqa: BLE001
        episode.error = f"{type(exc).__name__}: {exc}"
        episode.stop_reason = StopReason.ERROR
    episode.wall_ms = (time.perf_counter() - started) * 1000
    return episode


def _accumulate_usage(episode: Episode, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, field, None)
        if isinstance(value, int):
            episode.usage[field] = episode.usage.get(field, 0) + value


__all__ = [
    "EpisodeRunner",
    "FatalRunError",
    "run_static_episode",
    "select_tools",
]
