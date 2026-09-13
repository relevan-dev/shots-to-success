"""Command line entry point: `sts`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import adapters as adapters_module
from . import datasets as datasets_module
from .config import CONDITIONS, RunConfig
from .datasets import load_dataset
from .grading import aggregate
from .recording import load_run
from .report import compare, inspect, regrade
from .runner import run as run_benchmark
from .types import Episode


def _kv(values: list[str] | None) -> dict[str, object]:
    """Parse repeated ``--opt key=value`` flags, JSON-decoding where possible."""
    out: dict[str, object] = {}
    for item in values or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"bad option {item!r}; expected key=value")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sts",
        description="Shots to Success: how many search attempts an agent needs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show built-in datasets, adapters, conditions")

    prepare = sub.add_parser("prepare", help="download a dataset's source files")
    prepare.add_argument("--dataset", default="wands")
    prepare.add_argument("--data-dir", default="data")
    prepare.add_argument("--opt", action="append", metavar="KEY=VALUE")

    run = sub.add_parser("run", help="run a benchmark condition")
    run.add_argument("--dataset", default="wands", help="name or 'pkg.mod:Class'")
    run.add_argument("--adapter", default="bm25", help="name or 'pkg.mod:Class'")
    run.add_argument(
        "--condition",
        default="feedback",
        help=f"one of: {', '.join(CONDITIONS)}",
    )
    run.add_argument("--model", default="claude-opus-5")
    run.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="thinking depth; a controlled variable, keep it fixed across a sweep",
    )
    run.add_argument("--max-shots", type=int, default=3, help="search budget")
    run.add_argument("--k", type=int, default=10, help="result depth shown and scored")
    run.add_argument("--limit", type=int, help="run a seeded random sample of queries")
    run.add_argument("--query-id", action="append", dest="query_ids")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--max-turns", type=int, default=24)
    run.add_argument("--max-tokens", type=int, default=8000)
    run.add_argument("--relevant-threshold", type=int)
    run.add_argument(
        "--include-unjudged",
        action="store_true",
        help="keep queries with no judged-relevant document (they cannot succeed)",
    )
    run.add_argument(
        "--difficulty",
        default="all",
        choices=["all", "hard"],
        help="'hard' keeps only queries the adapter's single-shot baseline fails",
    )
    run.add_argument(
        "--episode-set",
        help="JSON file of query ids; pins the episode set across adapters",
    )
    run.add_argument("--run-id")
    run.add_argument("--out-dir", default="runs")
    run.add_argument("--data-dir", default="data")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--no-transcript", action="store_true")
    run.add_argument("--opt", action="append", metavar="KEY=VALUE", help="dataset option")
    run.add_argument(
        "--adapter-opt", action="append", metavar="KEY=VALUE", help="adapter option"
    )

    sweep = sub.add_parser(
        "sweep",
        help="run several conditions over one pinned episode set, then compare",
    )
    sweep.add_argument("--dataset", default="wands")
    sweep.add_argument("--adapter", default="bm25")
    sweep.add_argument(
        "--conditions",
        default="static,results_only,feedback",
        help="comma-separated; the episode set is shared across all of them",
    )
    sweep.add_argument("--model", default="claude-opus-5")
    sweep.add_argument(
        "--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"]
    )
    sweep.add_argument("--max-shots", type=int, default=3)
    sweep.add_argument("--k", type=int, default=10)
    sweep.add_argument("--difficulty", default="all", choices=["all", "hard"])
    sweep.add_argument("--episode-set")
    sweep.add_argument("--limit", type=int)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--concurrency", type=int, default=4)
    sweep.add_argument("--relevant-threshold", type=int)
    sweep.add_argument("--run-prefix", help="run ids become <prefix>-<condition>")
    sweep.add_argument("--out-dir", default="runs")
    sweep.add_argument("--data-dir", default="data")
    sweep.add_argument("--no-resume", action="store_true")
    sweep.add_argument("--opt", action="append", metavar="KEY=VALUE")
    sweep.add_argument("--adapter-opt", action="append", metavar="KEY=VALUE")

    episodes = sub.add_parser(
        "episodes", help="materialize an episode set to a file, for pinning"
    )
    episodes.add_argument("--dataset", default="wands")
    episodes.add_argument("--adapter", default="bm25")
    episodes.add_argument("--difficulty", default="hard", choices=["all", "hard"])
    episodes.add_argument("--k", type=int, default=10)
    episodes.add_argument("--limit", type=int)
    episodes.add_argument("--seed", type=int, default=0)
    episodes.add_argument("--relevant-threshold", type=int)
    episodes.add_argument("--data-dir", default="data")
    episodes.add_argument("-o", "--out", required=True)
    episodes.add_argument("--opt", action="append", metavar="KEY=VALUE")
    episodes.add_argument("--adapter-opt", action="append", metavar="KEY=VALUE")

    report = sub.add_parser("report", help="compare one or more runs")
    report.add_argument("run_dirs", nargs="+")
    report.add_argument("--json", action="store_true")

    show = sub.add_parser("inspect", help="print episode trajectories from a run")
    show.add_argument("run_dir")
    show.add_argument("--query-id")
    show.add_argument("--failures", action="store_true")

    again = sub.add_parser(
        "regrade", help="re-score a finished run under different judgments"
    )
    again.add_argument("run_dir")
    again.add_argument("--relevant-threshold", type=int)
    again.add_argument("--k", type=int)
    again.add_argument("--data-dir", default="data")

    return parser


def main(argv: list[str] | None = None) -> int:
    from .agent.loop import FatalRunError

    try:
        return _dispatch(build_parser().parse_args(argv))
    except FatalRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:

    if args.command == "list":
        print("datasets:")
        for name, cls in sorted(datasets_module.BUILTIN.items()):
            note = "  [recommended: see docs/DATASETS.md]" if name == "trec-tot" else ""
            summary = (cls.__doc__ or "").splitlines()[0]
            print(f"  {name:<12} {summary}{note}")
        print("  <any>        pkg.module:ClassName (your own Dataset subclass)")
        print("\nadapters:")
        for name, cls in sorted(adapters_module.BUILTIN.items()):
            print(f"  {name:<12} {(cls.__doc__ or '').splitlines()[0]}")
        print("  <any>        pkg.module:ClassName (your own SearchAdapter subclass)")
        print("\nconditions:")
        for name, condition in CONDITIONS.items():
            print(f"  {name:<14} {condition.description}")
        return 0

    if args.command == "prepare":
        dataset = load_dataset(args.dataset, args.data_dir, **_kv(args.opt))
        dataset.prepare()
        queries = dataset.queries()
        judgments = dataset.judgments()
        judged = sum(1 for q in queries if judgments.relevant_docs(q.query_id))
        print(
            f"{dataset.name}: {len(queries)} queries "
            f"({judged} with at least one judged-relevant document at "
            f"threshold {judgments.relevant_threshold}), "
            f"{sum(1 for _ in dataset.corpus())} documents"
        )
        return 0

    if args.command == "run":
        config = RunConfig(
            dataset=args.dataset,
            adapter=args.adapter,
            condition=args.condition,
            model=args.model,
            effort=args.effort,
            max_shots=args.max_shots,
            k=args.k,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            relevant_threshold=args.relevant_threshold,
            require_judged=not args.include_unjudged,
            difficulty=args.difficulty,
            episode_set=args.episode_set or "",
            limit=args.limit,
            query_ids=args.query_ids or [],
            seed=args.seed,
            concurrency=args.concurrency,
            run_id=args.run_id or "",
            out_dir=args.out_dir,
            data_dir=args.data_dir,
            resume=not args.no_resume,
            include_transcript=not args.no_transcript,
            dataset_options=_kv(args.opt),
            adapter_options=_kv(args.adapter_opt),
        )
        summary = run_benchmark(config)
        print(json.dumps(summary["metrics"], indent=2, default=str))
        return 0

    if args.command == "episodes":
        from .runner import build_world, episode_set_sha, select_queries

        config = RunConfig(
            dataset=args.dataset,
            adapter=args.adapter,
            condition="static",
            k=args.k,
            limit=args.limit,
            seed=args.seed,
            relevant_threshold=args.relevant_threshold,
            difficulty=args.difficulty,
            data_dir=args.data_dir,
            dataset_options=_kv(args.opt),
            adapter_options=_kv(args.adapter_opt),
        )
        dataset, adapter, judgments = build_world(config)
        queries = select_queries(
            dataset, adapter, judgments, config, lambda m: print(m, file=sys.stderr)
        )
        Path(args.out).write_text(
            json.dumps([q.query_id for q in queries], indent=1), encoding="utf-8"
        )
        print(
            f"wrote {args.out}: {len(queries)} episodes, "
            f"set {episode_set_sha(queries)}"
        )
        return 0

    if args.command == "sweep":
        import datetime as dt

        from .runner import build_world, episode_set_sha, select_queries

        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            raise SystemExit(f"unknown condition(s): {', '.join(unknown)}")

        prefix = args.run_prefix or (
            f"{args.dataset}-{args.adapter}-"
            + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        shared = dict(
            dataset=args.dataset,
            adapter=args.adapter,
            model=args.model,
            effort=args.effort,
            max_shots=args.max_shots,
            k=args.k,
            relevant_threshold=args.relevant_threshold,
            limit=args.limit,
            seed=args.seed,
            concurrency=args.concurrency,
            out_dir=args.out_dir,
            data_dir=args.data_dir,
            resume=not args.no_resume,
            dataset_options=_kv(args.opt),
            adapter_options=_kv(args.adapter_opt),
        )

        # Build the index once and pin the episode set before anything runs, so
        # every condition is scored on exactly the same queries.
        world = build_world(RunConfig(condition="static", **shared))
        dataset, adapter, judgments = world
        set_path = args.episode_set
        if not set_path:
            selection = RunConfig(
                condition="static", difficulty=args.difficulty, **shared
            )
            queries = select_queries(
                dataset, adapter, judgments, selection,
                lambda m: print(m, file=sys.stderr),
            )
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            set_path = str(out_dir / f"{prefix}-episodes.json")
            Path(set_path).write_text(
                json.dumps([q.query_id for q in queries], indent=1), encoding="utf-8"
            )
            print(
                f"episode set {episode_set_sha(queries)}: "
                f"{len(queries)} queries -> {set_path}",
                file=sys.stderr,
            )

        run_dirs = []
        for condition in conditions:
            print(f"\n=== {condition} ===", file=sys.stderr)
            config = RunConfig(
                condition=condition,
                episode_set=set_path,
                run_id=f"{prefix}-{condition}",
                **shared,
            )
            run_benchmark(config, world=world)
            run_dirs.append(str(Path(args.out_dir) / f"{prefix}-{condition}"))
        adapter.close()
        print()
        print(compare(run_dirs))
        return 0

    if args.command == "report":
        if args.json:
            out = []
            for run_dir in args.run_dirs:
                manifest, episodes = load_run(run_dir)
                config = manifest["config"]
                out.append(
                    {
                        "run": Path(run_dir).name,
                        "condition": config["condition"],
                        "metrics": aggregate(
                            [Episode.from_json(e) for e in episodes],
                            config["k"],
                            config["max_shots"],
                        ),
                    }
                )
            print(json.dumps(out, indent=2, default=str))
        else:
            print(compare(args.run_dirs))
        return 0

    if args.command == "inspect":
        print(inspect(args.run_dir, args.query_id, args.failures))
        return 0

    if args.command == "regrade":
        manifest, _ = load_run(args.run_dir)
        config = manifest["config"]
        options = dict(config.get("dataset_options") or {})
        if args.relevant_threshold is not None:
            options["relevant_threshold"] = args.relevant_threshold
        dataset = load_dataset(config["dataset"], args.data_dir, **options)
        k = args.k or config["k"]
        _, metrics = regrade(args.run_dir, dataset.judgments(), k)
        print(json.dumps(metrics, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
