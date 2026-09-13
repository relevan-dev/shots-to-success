"""Reading runs back: comparison tables, re-grading, and trajectory inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .config import RunConfig
from .grading import aggregate, grade_episode
from .recording import load_run
from .types import Episode, Judgments

#: Column label, metric key template, formatter.
COLUMNS: list[tuple[str, str, str]] = [
    ("episodes", "episodes", "int"),
    ("shots->success", "shots_to_success", "f2"),
    ("censored", "shots_to_success_censored", "f2"),
    ("success@{k}", "success@{k}", "pct"),
    ("recovery", "recovery_rate", "pct"),
    ("bad retry", "bad_retry_rate", "pct"),
    ("oversearch", "oversearch_rate", "pct"),
    ("nDCG@{k}", "ndcg@{k}", "f3"),
    ("recall@{k}", "recall@{k}", "f3"),
    ("judged@{k}", "judged@{k}", "pct"),
    ("shots used", "mean_shots_used", "f2"),
]


def _format(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "int":
        return str(int(value))
    if kind == "f2":
        return f"{value:.2f}"
    return f"{value:.3f}"


def render_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    widths = {
        header: max(len(header), *(len(row.get(header, "")) for row in rows))
        for header in headers
    }
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append(
            "  ".join(row.get(header, "").ljust(widths[header]) for header in headers)
        )
    return "\n".join(lines)


def compare(run_dirs: Sequence[str | Path]) -> str:
    """Render one or more runs side by side, flagging any drift between them."""
    loaded = []
    for run_dir in run_dirs:
        manifest, episodes = load_run(run_dir)
        summary_path = Path(run_dir) / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary.get("metrics", {})
        else:
            config = manifest["config"]
            metrics = aggregate(
                [Episode.from_json(e) for e in episodes],
                config["k"],
                config["max_shots"],
            )
        loaded.append((Path(run_dir).name, manifest, metrics))

    k = loaded[0][1]["config"]["k"]
    headers = ["run"] + [label.format(k=k) for label, _, _ in COLUMNS]
    rows = []
    for name, manifest, metrics in loaded:
        row = {"run": f"{manifest['config']['condition']} ({name})"}
        for label, key, kind in COLUMNS:
            row[label.format(k=k)] = _format(metrics.get(key.format(k=k)), kind)
        rows.append(row)

    out = [render_table(rows, headers)]

    warnings = _drift_warnings([m for _, m, _ in loaded])
    if warnings:
        out.append("")
        out.append("Comparability warnings:")
        out.extend(f"  ! {w}" for w in warnings)

    out.append("")
    out.append("Stop reasons:")
    for name, manifest, metrics in loaded:
        reasons = metrics.get("stop_reasons") or {}
        detail = ", ".join(f"{key}={value}" for key, value in reasons.items())
        out.append(f"  {manifest['config']['condition']:<14} {detail or '-'}")
    return "\n".join(out)


def _drift_warnings(manifests: Sequence[dict[str, Any]]) -> list[str]:
    """Flag controlled variables that differ between runs being compared."""
    if len(manifests) < 2:
        return []
    warnings: list[str] = []
    keys = list(RunConfig(dataset="", adapter="", condition="").controlled())
    for key in keys:
        values = {
            json.dumps(m["config"].get(key), sort_keys=True, default=str)
            for m in manifests
        }
        if len(values) > 1:
            warnings.append(f"{key} differs: {', '.join(sorted(values))}")

    sets = {m.get("episode_set_sha") for m in manifests if m.get("episode_set_sha")}
    if len(sets) > 1:
        warnings.append(
            "runs scored different episode sets: " + ", ".join(sorted(s for s in sets if s))
        )

    prompts = {m.get("system_prompt_sha") for m in manifests if m.get("system_prompt_sha")}
    conditions = {m["config"]["condition"] for m in manifests}
    if len(prompts) > 1 and conditions <= {"static", "results_only", "feedback"}:
        # Feedback Track runs must share a byte-identical prompt; only the
        # Tooling Track is expected to change the instructions.
        warnings.append(
            "system prompt differs between Feedback Track runs: "
            + ", ".join(sorted(p for p in prompts if p))
        )
    return warnings


def regrade(
    run_dir: str | Path, judgments: Judgments, k: int
) -> tuple[list[Episode], dict[str, Any]]:
    """Re-score a finished run under different judgments or a different k.

    No model calls: the recorded result sets are all the grader needs.
    """
    manifest, payloads = load_run(run_dir)
    episodes = [Episode.from_json(p) for p in payloads]
    for episode in episodes:
        episode.metrics = {
            key: value
            for key, value in episode.metrics.items()
            if key == "nominated_doc_ids"
        }
        grade_episode(episode, judgments, k)
    metrics = aggregate(episodes, k, manifest["config"]["max_shots"])
    return episodes, metrics


def inspect(run_dir: str | Path, query_id: str | None = None, failures: bool = False) -> str:
    """Human-readable trajectory for one or more episodes."""
    _, payloads = load_run(run_dir)
    episodes = [Episode.from_json(p) for p in payloads]
    if query_id:
        episodes = [e for e in episodes if e.query_id == query_id]
    if failures:
        episodes = [e for e in episodes if not e.metrics.get("success")]
    if not episodes:
        return "no matching episodes"

    out: list[str] = []
    for episode in episodes:
        out.append("=" * 72)
        out.append(
            f"[{episode.query_id}] {episode.query}  "
            f"({episode.condition}, {episode.stop_reason.value})"
        )
        relevant_total = (
            episode.attempts[0].scores.get("relevant_in_collection")
            if episode.attempts
            else None
        )
        out.append(
            f"  success={episode.metrics.get('success')}  "
            f"shots_to_success={episode.metrics.get('shots_to_success')}  "
            f"shots_used={episode.metrics.get('shots_used')}  "
            f"relevant_in_collection={int(relevant_total or 0)}"
        )
        for aux in episode.aux_calls:
            out.append(f"  (free) {aux.tool} {json.dumps(aux.tool_input)}")
        for attempt in episode.attempts:
            marker = "*" if attempt.shot == episode.submitted_shot else " "
            k_keys = [key for key in attempt.scores if key.startswith("success@")]
            k = k_keys[0].split("@")[1] if k_keys else "10"
            out.append(
                f" {marker}shot {attempt.shot}: {attempt.tool} "
                f"{json.dumps(attempt.tool_input, ensure_ascii=False)}"
            )
            out.append(
                f"     -> success={int(attempt.scores.get(f'success@{k}', 0))} "
                f"ndcg={attempt.scores.get(f'ndcg@{k}', 0):.3f} "
                f"first_rel_rank={int(attempt.scores.get('first_relevant_rank', 0))} "
                f"top={', '.join(h.title[:32] for h in attempt.result_set.hits[:3])}"
            )
        if episode.stop_rationale:
            out.append(f"  stop: {episode.stop_rationale}")
        if episode.error:
            out.append(f"  ERROR: {episode.error}")
    return "\n".join(out)


__all__ = ["compare", "inspect", "regrade", "render_table"]
