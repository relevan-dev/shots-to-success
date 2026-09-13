"""Agent instructions and result rendering.

Both are controlled variables. The system prompt is versioned and hashed into
the run manifest, and the renderer emits exactly the same fields in every
condition -- the only thing a condition changes is whether the adapter's
diagnostics block is appended. Keep it that way: any conditional text added
here that is not gated on a declared condition flag silently breaks
comparability between runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..config import Condition
from ..types import Query, ToolResponse

PROMPT_VERSION = "v1"

_SYSTEM = """\
You are testing a search system by using it to satisfy an information need.

You will be given one information need and a set of search tools over a
document collection. Find documents in that collection that satisfy the need.

# Relevance standard

{rubric}

# Search budget

You have {max_shots} {search_word}. Every call to a retrieval tool spends one, \
whether or not it returns anything useful.{aux_clause}

When your budget is gone, further retrieval calls will be refused and your last \
search will be submitted automatically.

# After each search

Look at the top {k} results and decide: stop, or search again.

Stop when the top results already contain at least one document that meets the \
relevance standard above. Searching further at that point does not improve your \
result and counts against you.

Search again when the results contain nothing that meets the standard. Before \
you do, say what specifically went wrong -- wrong vocabulary, too broad, too \
narrow, a filter that excluded what you wanted, a concept the collection does \
not use -- and make the next query address that. A retry that only rephrases \
the previous query without a reason wastes a shot.

# Finishing

Call `submit_results` to stop. The results of your most recent search become \
your final answer, so your last search should be your best one. Never call \
`submit_results` before you have run at least one search.
"""

_AUX_CLAUSE = (
    "\n\nThe collection also exposes tools that do not retrieve documents "
    "(query suggestions, field value counts, index schema, score explanation). "
    "Those are free and do not count against your budget -- use them to build a "
    "better query before spending a search."
)

SUBMIT_TOOL: dict[str, Any] = {
    "name": "submit_results",
    "description": (
        "Stop searching and submit the results of your most recent search as "
        "your final answer. Call this as soon as those results contain at least "
        "one document meeting the relevance standard."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "satisfying_doc_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "doc_ids from your most recent search that you believe meet "
                    "the relevance standard. Empty if you are submitting without "
                    "having found any."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One or two sentences on why you are stopping now.",
            },
            "confident": {
                "type": "boolean",
                "description": (
                    "True if you believe the submitted results satisfy the need; "
                    "false if you are submitting because you are out of options."
                ),
            },
        },
        "required": ["satisfying_doc_ids", "reason", "confident"],
    },
}


def render_system_prompt(condition: Condition, k: int, rubric: str) -> str:
    return _SYSTEM.format(
        rubric=rubric.strip(),
        max_shots=condition.max_shots,
        search_word="search" if condition.max_shots == 1 else "searches",
        aux_clause=_AUX_CLAUSE if condition.auxiliary_tools else "",
        k=k,
    )


def system_prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def render_information_need(query: Query) -> str:
    """The episode's opening user message.

    Only the query text crosses this boundary. Dataset metadata such as WANDS'
    ``query_class`` is annotator context, not something a searcher would have,
    so it stays out.
    """
    return f"Information need: {query.text}"


def render_tool_result(response: ToolResponse, condition: Condition) -> str:
    """What the model sees back from a tool call.

    Identical in every condition except the diagnostics block, which is the
    Feedback Track's single lever.
    """
    if response.error:
        return f"Error: {response.error}"
    parts: list[str] = []
    if response.body is not None:
        parts.append(_dumps(response.body))
    if condition.feedback and response.diagnostics:
        parts.append(
            "Retrieval diagnostics:\n" + _dumps(_prune(response.diagnostics))
        )
    return "\n\n".join(parts) if parts else "(no content)"


def _prune(value: Any) -> Any:
    """Drop null and empty entries so diagnostics read as signal, not schema."""
    if isinstance(value, dict):
        return {
            key: _prune(item)
            for key, item in value.items()
            if item not in (None, [], {}, "")
        }
    if isinstance(value, list):
        return [_prune(item) for item in value]
    return value


def _dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


__all__ = [
    "PROMPT_VERSION",
    "SUBMIT_TOOL",
    "render_information_need",
    "render_system_prompt",
    "render_tool_result",
    "system_prompt_sha",
]
