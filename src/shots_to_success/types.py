"""Core data model shared by datasets, adapters, the agent loop, and the grader.

Two rules govern this module:

1. Nothing here that an adapter or the agent loop can see carries relevance
   judgments. Judgments live in :class:`Judgments`, which only the grader loads.
2. Everything an episode records is JSON-serializable, so a run is a plain
   JSONL file that can be re-scored later without re-running the model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

# --------------------------------------------------------------------------
# Dataset side
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Document:
    """One indexable item in the corpus."""

    doc_id: str
    title: str
    text: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Query:
    """An information need adapted from a benchmark query."""

    query_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Judgments:
    """Hidden relevance labels. Never passed to an adapter or to the model.

    ``grades`` maps ``query_id -> doc_id -> graded relevance`` (higher is
    better). ``relevant_threshold`` is the grade at or above which a document
    counts as "judged relevant" for the binary Success@k family of metrics.
    ``max_grade`` is used as the ideal gain when computing nDCG.
    """

    grades: dict[str, dict[str, int]]
    relevant_threshold: int
    max_grade: int
    grade_names: dict[int, str] = field(default_factory=dict)

    def for_query(self, query_id: str) -> dict[str, int]:
        return self.grades.get(query_id, {})

    def relevant_docs(self, query_id: str) -> set[str]:
        return {
            doc_id
            for doc_id, grade in self.grades.get(query_id, {}).items()
            if grade >= self.relevant_threshold
        }


# --------------------------------------------------------------------------
# Adapter side: what a search backend returns
# --------------------------------------------------------------------------


class ToolKind(str, Enum):
    """Whether calling a tool spends a shot.

    ``RETRIEVAL`` tools produce a ranked result set that can be submitted and
    graded; each call consumes one unit of the search budget. ``AUXILIARY``
    tools (autocomplete, facets, explain, schema inspection) inform query
    construction but return nothing gradable, so they are free.
    """

    RETRIEVAL = "retrieval"
    AUXILIARY = "auxiliary"


@dataclass(slots=True)
class Hit:
    """One ranked result.

    Adapters normalize into this shape so that result rendering is identical
    across backends -- result rendering is a controlled variable.
    """

    doc_id: str
    score: float
    title: str = ""
    snippet: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Hit":
        return cls(
            doc_id=str(payload["doc_id"]),
            score=float(payload.get("score", 0.0)),
            title=payload.get("title", ""),
            snippet=payload.get("snippet", ""),
            extra=dict(payload.get("extra") or {}),
        )


@dataclass(slots=True)
class ResultSet:
    """A ranked list of hits produced by one retrieval tool call."""

    hits: list[Hit]
    query: str = ""
    total: int | None = None

    @property
    def doc_ids(self) -> list[str]:
        return [hit.doc_id for hit in self.hits]

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "total": self.total,
            "hits": [hit.to_json() for hit in self.hits],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ResultSet":
        return cls(
            hits=[Hit.from_json(h) for h in payload.get("hits", [])],
            query=payload.get("query", ""),
            total=payload.get("total"),
        )


@dataclass(slots=True)
class ToolResponse:
    """What an adapter tool handler returns.

    ``body`` is always rendered to the model. ``diagnostics`` is rendered only
    in feedback-enabled conditions -- this is the single lever the Feedback
    Track turns on and off. ``result_set`` must be set by ``RETRIEVAL`` tools
    and is what the grader scores.
    """

    body: Any = None
    result_set: ResultSet | None = None
    diagnostics: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class ToolSpec:
    """A tool an adapter exposes to the agent.

    ``input_schema`` is a JSON Schema object passed straight through to the
    model. ``handler`` receives the parsed tool input and returns a
    :class:`ToolResponse`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    kind: ToolKind
    handler: Callable[[Mapping[str, Any]], ToolResponse]
    # Tools that only make sense when retrieval feedback is being tested.
    # Hidden in conditions with feedback disabled so the tool contract stays
    # comparable.
    feedback_only: bool = False

    def to_api_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# --------------------------------------------------------------------------
# Episode side: what a run records
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Attempt:
    """One retrieval tool call -- one "shot"."""

    shot: int  # 1-indexed
    tool: str
    tool_input: dict[str, Any]
    result_set: ResultSet
    latency_ms: float = 0.0
    diagnostics_shown: bool = False
    scores: dict[str, float] = field(default_factory=dict)  # filled by the grader

    def to_json(self) -> dict[str, Any]:
        return {
            "shot": self.shot,
            "tool": self.tool,
            "tool_input": self.tool_input,
            "result_set": self.result_set.to_json(),
            "latency_ms": round(self.latency_ms, 2),
            "diagnostics_shown": self.diagnostics_shown,
            "scores": self.scores,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Attempt":
        return cls(
            shot=int(payload["shot"]),
            tool=payload["tool"],
            tool_input=dict(payload.get("tool_input") or {}),
            result_set=ResultSet.from_json(payload.get("result_set") or {}),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            diagnostics_shown=bool(payload.get("diagnostics_shown", False)),
            scores=dict(payload.get("scores") or {}),
        )


@dataclass(slots=True)
class AuxCall:
    """One auxiliary tool call. Recorded but does not consume budget."""

    tool: str
    tool_input: dict[str, Any]
    latency_ms: float = 0.0
    after_shot: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_input": self.tool_input,
            "latency_ms": round(self.latency_ms, 2),
            "after_shot": self.after_shot,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "AuxCall":
        return cls(
            tool=payload["tool"],
            tool_input=dict(payload.get("tool_input") or {}),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            after_shot=int(payload.get("after_shot", 0)),
        )


class StopReason(str, Enum):
    SUBMITTED = "submitted"  # agent called submit_results
    BUDGET_EXHAUSTED = "budget_exhausted"  # ran out of shots, last set auto-submitted
    ENDED_TURN = "ended_turn"  # model stopped talking without submitting
    NO_SEARCH = "no_search"  # model never issued a retrieval call
    MAX_TURNS = "max_turns"
    ERROR = "error"


@dataclass(slots=True)
class Episode:
    """The full record of one agent's attempt at one information need."""

    query_id: str
    query: str
    condition: str
    adapter: str
    dataset: str
    attempts: list[Attempt] = field(default_factory=list)
    aux_calls: list[AuxCall] = field(default_factory=list)
    submitted_shot: int | None = None  # which attempt became the final result set
    stop_reason: StopReason = StopReason.ERROR
    stop_rationale: str = ""
    agent_confident: bool | None = None
    turns: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    wall_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)  # filled by the grader

    @property
    def shots_used(self) -> int:
        return len(self.attempts)

    @property
    def submitted(self) -> ResultSet | None:
        if self.submitted_shot is None:
            return None
        return self.attempts[self.submitted_shot - 1].result_set

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "condition": self.condition,
            "adapter": self.adapter,
            "dataset": self.dataset,
            "shots_used": self.shots_used,
            "submitted_shot": self.submitted_shot,
            "stop_reason": self.stop_reason.value,
            "stop_rationale": self.stop_rationale,
            "agent_confident": self.agent_confident,
            "turns": self.turns,
            "usage": self.usage,
            "wall_ms": round(self.wall_ms, 2),
            "error": self.error,
            "attempts": [a.to_json() for a in self.attempts],
            "aux_calls": [c.to_json() for c in self.aux_calls],
            "metrics": self.metrics,
            "transcript": self.transcript,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Episode":
        """Rebuild a recorded episode. Lets a finished run be re-graded or
        re-aggregated without calling the model again."""
        return cls(
            query_id=str(payload["query_id"]),
            query=payload.get("query", ""),
            condition=payload.get("condition", ""),
            adapter=payload.get("adapter", ""),
            dataset=payload.get("dataset", ""),
            attempts=[Attempt.from_json(a) for a in payload.get("attempts", [])],
            aux_calls=[AuxCall.from_json(c) for c in payload.get("aux_calls", [])],
            submitted_shot=payload.get("submitted_shot"),
            stop_reason=StopReason(payload.get("stop_reason", "error")),
            stop_rationale=payload.get("stop_rationale", ""),
            agent_confident=payload.get("agent_confident"),
            turns=int(payload.get("turns", 0)),
            usage=dict(payload.get("usage") or {}),
            transcript=list(payload.get("transcript") or []),
            error=payload.get("error"),
            wall_ms=float(payload.get("wall_ms", 0.0)),
            metrics=dict(payload.get("metrics") or {}),
        )


def documents_to_json(docs: Iterable[Document]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(d) for d in docs]


__all__ = [
    "Attempt",
    "AuxCall",
    "Document",
    "Episode",
    "Hit",
    "Judgments",
    "Query",
    "ResultSet",
    "StopReason",
    "ToolKind",
    "ToolResponse",
    "ToolSpec",
    "documents_to_json",
]
