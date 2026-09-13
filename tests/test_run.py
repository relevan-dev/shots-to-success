"""Run plumbing: episode selection, recording, resume, and re-grading."""

from __future__ import annotations

import json

import pytest

from shots_to_success.config import RunConfig, compare_controlled
from shots_to_success.grading import grade_episode
from shots_to_success.recording import RunRecorder, load_run
from shots_to_success.report import compare, inspect, regrade
from shots_to_success.runner import episode_set_sha, run_static_episode, select_queries
from shots_to_success.types import Episode


@pytest.fixture
def judgments(dataset):
    return dataset.judgments()


def config(**overrides):
    base = dict(dataset="tiny", adapter="bm25", condition="static", k=10, max_shots=3)
    base.update(overrides)
    return RunConfig(**base)


# -- episode selection -----------------------------------------------------


def test_unjudged_queries_are_dropped_by_default(dataset, adapter, judgments):
    judgments.grades["q2"] = {}  # q2 now has nothing relevant to find
    queries = select_queries(dataset, adapter, judgments, config())
    assert [q.query_id for q in queries] == ["q1"]


def test_include_unjudged_keeps_them(dataset, adapter, judgments):
    judgments.grades["q2"] = {}
    queries = select_queries(dataset, adapter, judgments, config(require_judged=False))
    assert len(queries) == 2


def test_hard_subset_keeps_only_baseline_failures(dataset, adapter, judgments):
    # Both tiny queries are solved by a single search, so "hard" is empty.
    queries = select_queries(dataset, adapter, judgments, config(difficulty="hard"))
    assert queries == []


def test_episode_set_file_pins_the_queries(dataset, adapter, judgments, tmp_path):
    path = tmp_path / "set.json"
    path.write_text(json.dumps(["q2"]))
    queries = select_queries(
        dataset, adapter, judgments, config(episode_set=str(path))
    )
    assert [q.query_id for q in queries] == ["q2"]


def test_unknown_query_id_in_an_episode_set_is_an_error(
    dataset, adapter, judgments, tmp_path
):
    path = tmp_path / "set.json"
    path.write_text(json.dumps(["nope"]))
    with pytest.raises(KeyError):
        select_queries(dataset, adapter, judgments, config(episode_set=str(path)))


def test_episode_set_sha_identifies_the_query_set(dataset):
    queries = dataset.queries()
    assert episode_set_sha(queries) == episode_set_sha(list(queries))
    assert episode_set_sha(queries) != episode_set_sha(queries[:1])


# -- static baseline -------------------------------------------------------


def test_static_baseline_runs_one_unmodified_query(dataset, adapter, judgments):
    cfg = config()
    episode = run_static_episode(
        adapter, cfg.condition_spec(), cfg, dataset.queries()[0]
    )
    grade_episode(episode, judgments, 10)
    assert episode.shots_used == 1
    assert episode.attempts[0].tool_input["query"] == "red chair"
    assert episode.metrics["success"] is True


# -- recording -------------------------------------------------------------


def test_episodes_round_trip_through_jsonl(dataset, adapter, judgments, tmp_path):
    cfg = config(out_dir=str(tmp_path))
    original = run_static_episode(adapter, cfg.condition_spec(), cfg, dataset.queries()[0])
    grade_episode(original, judgments, 10)

    with RunRecorder(cfg, tmp_path / "r1") as recorder:
        recorder.write_manifest({"note": "test"})
        recorder.write(original)

    manifest, payloads = load_run(tmp_path / "r1")
    assert manifest["note"] == "test"
    restored = Episode.from_json(payloads[0])
    assert restored.to_json() == original.to_json()


def test_resume_skips_already_recorded_queries(dataset, adapter, judgments, tmp_path):
    cfg = config(out_dir=str(tmp_path))
    episode = run_static_episode(adapter, cfg.condition_spec(), cfg, dataset.queries()[0])
    with RunRecorder(cfg, tmp_path / "r2") as recorder:
        recorder.write(episode)
    with RunRecorder(cfg, tmp_path / "r2") as recorder:
        assert recorder.completed_query_ids() == {"q1"}


def test_a_truncated_final_line_does_not_break_loading(tmp_path):
    run_dir = tmp_path / "r3"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}")
    (run_dir / "episodes.jsonl").write_text(
        json.dumps({"query_id": "q1", "attempts": []}) + "\n{\"query_id\": \"q2\""
    )
    _, payloads = load_run(run_dir)
    assert [p["query_id"] for p in payloads] == ["q1"]


def test_transcripts_can_be_omitted(dataset, adapter, judgments, tmp_path):
    cfg = config(out_dir=str(tmp_path), include_transcript=False)
    episode = run_static_episode(adapter, cfg.condition_spec(), cfg, dataset.queries()[0])
    episode.transcript = [{"turn": 1}]
    with RunRecorder(cfg, tmp_path / "r4") as recorder:
        recorder.write_manifest({"config": cfg.to_json()})
        recorder.write(episode)
    _, payloads = load_run(tmp_path / "r4")
    assert "transcript" not in payloads[0]


# -- re-grading and reporting ----------------------------------------------


def test_regrade_rescores_without_calling_a_model(dataset, adapter, judgments, tmp_path):
    cfg = config(out_dir=str(tmp_path))
    with RunRecorder(cfg, tmp_path / "r5") as recorder:
        recorder.write_manifest({"config": cfg.to_json()})
        for query in dataset.queries():
            episode = run_static_episode(adapter, cfg.condition_spec(), cfg, query)
            grade_episode(episode, judgments, 10)
            recorder.write(episode)
    # Re-scoring the same run at k=1 is a pure function of what was recorded.
    _, metrics = regrade(tmp_path / "r5", judgments, k=1)
    assert metrics["episodes"] == 2
    assert "success@1" in metrics


def test_report_flags_a_drifting_controlled_variable(tmp_path):
    for name, effort in (("a", "high"), ("b", "low")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        cfg = config(condition="results_only", effort=effort)
        (run_dir / "manifest.json").write_text(
            json.dumps({"config": cfg.to_json(), "episode_set_sha": "same"})
        )
        (run_dir / "episodes.jsonl").write_text("")
    output = compare([tmp_path / "a", tmp_path / "b"])
    assert "effort differs" in output


def test_report_flags_different_episode_sets(tmp_path):
    for name, sha in (("a", "aaa"), ("b", "bbb")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps({"config": config().to_json(), "episode_set_sha": sha})
        )
        (run_dir / "episodes.jsonl").write_text("")
    assert "different episode sets" in compare([tmp_path / "a", tmp_path / "b"])


def test_compare_controlled_ignores_the_condition_under_test():
    runs = [config(condition="results_only"), config(condition="feedback")]
    assert compare_controlled(runs) == []


def test_inspect_renders_a_trajectory(dataset, adapter, judgments, tmp_path):
    cfg = config(out_dir=str(tmp_path))
    with RunRecorder(cfg, tmp_path / "r6") as recorder:
        recorder.write_manifest({"config": cfg.to_json()})
        episode = run_static_episode(adapter, cfg.condition_spec(), cfg, dataset.queries()[0])
        grade_episode(episode, judgments, 10)
        recorder.write(episode)
    output = inspect(tmp_path / "r6")
    assert "red chair" in output
    assert "shot 1" in output
