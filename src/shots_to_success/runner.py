"""Run orchestration: build the world, run episodes, grade, record."""

from __future__ import annotations

import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import random
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from .adapters import load_adapter
from .adapters.base import SearchAdapter
from .agent.loop import EpisodeRunner, run_static_episode
from .agent.prompts import PROMPT_VERSION
from .config import RunConfig
from .datasets import load_dataset
from .datasets.base import Dataset
from .grading import aggregate, grade_episode
from .recording import RunRecorder
from .types import Episode, Judgments, Query


def episode_set_sha(queries: list[Query]) -> str:
    """Identifies the exact episode set, so a comparison can prove two runs
    scored the same queries."""
    blob = "\n".join(q.query_id for q in queries)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def default_run_id(config: RunConfig) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{config.dataset}-{config.adapter}-{config.condition}-{stamp}"


def select_queries(
    dataset: Dataset,
    adapter: SearchAdapter,
    judgments: Judgments,
    config: RunConfig,
    say: Callable[[str], None] = lambda _: None,
) -> list[Query]:
    """Pick the episode set.

    The episode set is a controlled variable, so this is deterministic and the
    resulting query ids are written into the run directory.

    Queries with no judged-relevant document are dropped by default: no system
    can reach Success@k on them, so they only add a constant offset and noise.
    Sub-sampling is a seeded random sample rather than a head slice, so a
    ``--limit`` run is representative of the dataset rather than of its first
    few query ids.
    """
    queries = dataset.queries()
    by_id = {q.query_id: q for q in queries}

    if config.episode_set:
        wanted = json.loads(Path(config.episode_set).read_text(encoding="utf-8"))
        missing = [qid for qid in wanted if qid not in by_id]
        if missing:
            raise KeyError(f"episode set references unknown query ids: {missing[:5]}")
        return [by_id[qid] for qid in wanted]

    if config.query_ids:
        wanted_ids = set(config.query_ids)
        return [q for q in queries if q.query_id in wanted_ids]

    if config.require_judged:
        queries = [q for q in queries if judgments.relevant_docs(q.query_id)]

    if config.difficulty == "hard":
        queries = _hard_subset(queries, adapter, judgments, config, say)
    elif config.difficulty != "all":
        raise ValueError(f"unknown difficulty {config.difficulty!r}; use all or hard")

    if config.limit is not None and config.limit < len(queries):
        queries = random.Random(config.seed).sample(queries, config.limit)
        queries.sort(key=lambda q: (len(q.query_id), q.query_id))
    return queries


def _hard_subset(
    queries: list[Query],
    adapter: SearchAdapter,
    judgments: Judgments,
    config: RunConfig,
    say: Callable[[str], None],
) -> list[Query]:
    """Queries the adapter's single unmodified query fails to satisfy.

    Cheap -- one baseline search per query, no model. On an easy dataset this
    is what keeps Shots to Success measurable instead of saturated.
    """
    from .grading import score_result_set

    say(f"selecting hard subset from {len(queries)} queries (single-shot baseline)")
    hard: list[Query] = []
    for query in queries:
        result_set = adapter.baseline(query.text, config.k)
        scores = score_result_set(
            result_set.doc_ids,
            judgments.for_query(query.query_id),
            judgments.relevant_threshold,
            judgments.max_grade,
            config.k,
        )
        if not scores[f"success@{config.k}"]:
            hard.append(query)
    say(f"  {len(hard)} of {len(queries)} queries fail the single-shot baseline")
    return hard


def build_world(config: RunConfig) -> tuple[Dataset, SearchAdapter, Judgments]:
    """Load the dataset, index the corpus, and load the hidden judgments."""
    dataset = load_dataset(config.dataset, config.data_dir, **config.dataset_options)
    dataset.prepare()
    if config.relevant_threshold is not None:
        dataset.options["relevant_threshold"] = config.relevant_threshold
    judgments = dataset.judgments()

    adapter = load_adapter(config.adapter, **config.adapter_options)
    cache_key = hashlib.sha256(
        json.dumps(
            {"dataset": dataset.describe(), "adapter_version": adapter.version},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:12]
    adapter.build(dataset.corpus(), dataset_name=dataset.name, cache_key=cache_key)
    return dataset, adapter, judgments


def run(
    config: RunConfig,
    progress: Callable[[str], None] | None = None,
    world: tuple[Dataset, SearchAdapter, Judgments] | None = None,
) -> dict[str, Any]:
    """Execute a full run and return its summary.

    ``world`` lets a sweep share one built index across conditions instead of
    re-indexing the corpus for each.
    """
    say = progress or (lambda message: print(message, file=sys.stderr))
    condition = config.condition_spec()
    dataset, adapter, judgments = world or build_world(config)
    queries = select_queries(dataset, adapter, judgments, config, say)

    run_id = config.run_id or default_run_id(config)
    config.run_id = run_id
    run_dir = Path(config.out_dir) / run_id

    client = None
    episode_runner: EpisodeRunner | None = None
    if condition.agentic:
        import anthropic

        client = anthropic.Anthropic(max_retries=config.max_retries)
        episode_runner = EpisodeRunner(
            client, adapter, condition, config, dataset.relevance_rubric()
        )

    with RunRecorder(config, run_dir) as recorder:
        done: set[str] = recorder.completed_query_ids() if config.resume else set()
        if done:
            say(f"resuming {run_id}: {len(done)} episodes already recorded")
        pending = [q for q in queries if q.query_id not in done]

        recorder.write_manifest(
            {
                "dataset_describe": dataset.describe(),
                "adapter_describe": adapter.describe(),
                "prompt_version": PROMPT_VERSION,
                "system_prompt": episode_runner.system_prompt if episode_runner else None,
                "system_prompt_sha": (
                    episode_runner.system_prompt_sha if episode_runner else None
                ),
                "tools_exposed": (
                    [t["name"] for t in episode_runner.api_tools]
                    if episode_runner
                    else [adapter.primary_tool().name]
                ),
                "episodes_planned": len(queries),
                "episode_set_sha": episode_set_sha(queries),
                "judged_queries": len(judgments.grades),
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        (run_dir / "episode_set.json").write_text(
            json.dumps([q.query_id for q in queries], indent=1), encoding="utf-8"
        )
        say(
            f"run {run_id}: {len(pending)} episodes to run "
            f"({condition.name}, budget {condition.max_shots}, k={config.k})"
        )

        counter = {"done": 0, "success": 0}
        lock = threading.Lock()

        def one(query: Query) -> Episode:
            if episode_runner is not None:
                episode = episode_runner.run(query)
            else:
                episode = run_static_episode(adapter, condition, config, query)
            grade_episode(episode, judgments, config.k)
            recorder.write(episode)
            with lock:
                counter["done"] += 1
                counter["success"] += int(bool(episode.metrics.get("success")))
                if counter["done"] % 5 == 0 or counter["done"] == len(pending):
                    rate = counter["success"] / counter["done"]
                    say(
                        f"  {counter['done']}/{len(pending)} episodes, "
                        f"success@{config.k}={rate:.0%}"
                    )
            return episode

        episodes: list[Episode] = []
        workers = 1 if not condition.agentic else max(1, config.concurrency)
        if workers == 1:
            episodes = [one(query) for query in pending]
        else:
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for future in futures.as_completed(
                    [pool.submit(one, query) for query in pending]
                ):
                    episodes.append(future.result())

        # Aggregate over everything on disk, so a resumed run summarizes the
        # whole thing rather than only this invocation's slice.
        all_episodes = episodes + _rehydrate(recorder, done)
        summary = {
            "run_id": run_id,
            "condition": condition.to_json(),
            "config": config.to_json(),
            "metrics": aggregate(all_episodes, config.k, condition.max_shots),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        recorder.write_summary(summary)

    if world is None:
        adapter.close()
    say(f"wrote {run_dir}")
    return summary


def _rehydrate(recorder: RunRecorder, wanted: set[str]) -> list[Episode]:
    """Load previously recorded episodes so a resumed run summarizes in full."""
    from .recording import read_episodes_json

    out: list[Episode] = []
    for payload in read_episodes_json(recorder.episodes_path):
        if payload["query_id"] in wanted:
            out.append(Episode.from_json(payload))
            wanted.discard(payload["query_id"])
    return out


__all__ = [
    "build_world",
    "default_run_id",
    "episode_set_sha",
    "run",
    "select_queries",
]
