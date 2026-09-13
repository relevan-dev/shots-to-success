"""Run configuration and test conditions.

The benchmark's central claim is that differences between conditions are
attributable to the condition alone. That only holds if everything else is
pinned and recorded, so :class:`RunConfig` doubles as the run manifest: it is
hashed and written next to the episodes, and :func:`RunConfig.controlled`
lists the variables that must match for two runs to be comparable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Condition:
    """One test condition.

    ``agentic=False`` is the static baseline: the original benchmark query is
    executed once, with no model in the loop at all.
    """

    name: str
    description: str
    agentic: bool = True
    #: Render adapter diagnostics (matched terms, filter effects, score
    #: contributors) after each search. The Feedback Track's only lever.
    feedback: bool = False
    #: Expose auxiliary tools (autocomplete, facets, explain). Widening this
    #: changes the tool contract, which moves a run into the Tooling Track.
    auxiliary_tools: bool = False
    max_shots: int = 3

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: Built-in conditions. The first three form the Feedback Track comparison and
#: differ only in `feedback`; `tooling` additionally opens the tool contract.
CONDITIONS: dict[str, Condition] = {
    "static": Condition(
        name="static",
        description="Original benchmark query executed once. No model, no retries.",
        agentic=False,
        feedback=False,
        auxiliary_tools=False,
        max_shots=1,
    ),
    "results_only": Condition(
        name="results_only",
        description="Agent sees results and snippets, then may stop or retry.",
        agentic=True,
        feedback=False,
        auxiliary_tools=False,
    ),
    "feedback": Condition(
        name="feedback",
        description=(
            "Agent sees results plus retrieval diagnostics: matched and missing "
            "terms, filter behavior, and score contributors."
        ),
        agentic=True,
        feedback=True,
        auxiliary_tools=False,
    ),
    "tooling": Condition(
        name="tooling",
        description=(
            "Feedback plus the adapter's auxiliary tools (autocomplete, facets, "
            "explain). Tooling Track: the tool contract itself varies."
        ),
        agentic=True,
        feedback=True,
        auxiliary_tools=True,
    ),
}


@dataclass(slots=True)
class RunConfig:
    """Everything that defines a run. Serialized as the run manifest."""

    dataset: str
    adapter: str
    condition: str
    # --- controlled variables -------------------------------------------
    model: str = "claude-opus-5"
    effort: str = "high"  # low | medium | high | xhigh | max
    max_shots: int = 3
    k: int = 10  # result set depth the agent sees and the grader scores
    max_turns: int = 24  # hard stop on model turns, guards runaway loops
    max_tokens: int = 8000
    prompt_version: str = "v1"
    relevant_threshold: int | None = None  # None -> dataset default
    #: Skip queries with no judged-relevant document. Those episodes cannot
    #: reach Success@k no matter what the system does, so including them only
    #: adds noise. Recorded in the manifest because it changes the episode set.
    require_judged: bool = True
    #: Which queries become episodes. "all" keeps every judged query; "hard"
    #: keeps only those the adapter's own single-shot baseline fails, which is
    #: where a multi-shot agent has anything to prove. Because "hard" is
    #: computed from the adapter, two runs on different adapters get different
    #: episode sets -- pin `episode_set` to compare across backends.
    difficulty: str = "all"
    #: Path to a JSON list of query ids. Overrides `difficulty` and makes the
    #: episode set an explicit, shareable artifact.
    episode_set: str = ""
    # --- run plumbing (not a controlled variable) ------------------------
    limit: int | None = None
    query_ids: list[str] = field(default_factory=list)
    seed: int = 0
    concurrency: int = 4
    max_retries: int = 4
    run_id: str = ""
    out_dir: str = "runs"
    data_dir: str = "data"
    resume: bool = True
    include_transcript: bool = True
    dataset_options: dict[str, Any] = field(default_factory=dict)
    adapter_options: dict[str, Any] = field(default_factory=dict)

    def condition_spec(self) -> Condition:
        try:
            base = CONDITIONS[self.condition]
        except KeyError:
            raise KeyError(
                f"unknown condition {self.condition!r}; "
                f"known: {', '.join(sorted(CONDITIONS))}"
            ) from None
        # The CLI's --max-shots wins so budget stays identical across a sweep.
        return dataclasses.replace(base, max_shots=self.max_shots)

    def controlled(self) -> dict[str, Any]:
        """The variables that must match for two runs to be comparable.

        Note that ``condition`` is deliberately absent: it is the variable
        under test. ``adapter`` is present because the Feedback Track fixes
        the backend; Tooling and System Track comparisons intentionally drop
        it, which :func:`compare_controlled` reports rather than enforces.
        """
        return {
            "dataset": self.dataset,
            "adapter": self.adapter,
            "model": self.model,
            "effort": self.effort,
            "max_shots": self.max_shots,
            "k": self.k,
            "prompt_version": self.prompt_version,
            "relevant_threshold": self.relevant_threshold,
            "require_judged": self.require_judged,
            "difficulty": self.difficulty,
            "episode_set": self.episode_set,
            "dataset_options": self.dataset_options,
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.controlled(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_json(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["condition_spec"] = self.condition_spec().to_json()
        payload["fingerprint"] = self.fingerprint()
        return payload


def compare_controlled(configs: list[RunConfig]) -> list[str]:
    """Return human-readable warnings for any controlled variable that drifted."""

    warnings: list[str] = []
    if len(configs) < 2:
        return warnings
    reference = configs[0].controlled()
    for key in reference:
        values = {json.dumps(c.controlled()[key], sort_keys=True, default=str) for c in configs}
        if len(values) > 1:
            warnings.append(
                f"controlled variable {key!r} differs across runs: "
                + ", ".join(sorted(values))
            )
    return warnings


__all__ = ["CONDITIONS", "Condition", "RunConfig", "compare_controlled"]
