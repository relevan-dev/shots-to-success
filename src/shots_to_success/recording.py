"""Run artifacts on disk.

A run is a directory:

    runs/<run_id>/
        manifest.json    every controlled variable, plus dataset/adapter provenance
        episodes.jsonl   one graded episode per line, written as they finish
        summary.json     aggregate metrics

Episodes are written incrementally so a long run is resumable and so a crashed
run still yields whatever it finished.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator

from .config import RunConfig
from .types import Episode


class RunRecorder:
    """Append-only JSONL writer for one run."""

    def __init__(self, config: RunConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = run_dir / "episodes.jsonl"
        self._lock = threading.Lock()
        self._handle = self.episodes_path.open("a", encoding="utf-8")

    def completed_query_ids(self) -> set[str]:
        """Query ids already recorded, so a resumed run skips them."""
        return {episode["query_id"] for episode in read_episodes_json(self.episodes_path)}

    def write_manifest(self, extra: dict[str, Any]) -> None:
        payload = {"config": self.config.to_json(), **extra}
        (self.run_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def write(self, episode: Episode) -> None:
        payload = episode.to_json()
        if not self.config.include_transcript:
            payload.pop("transcript", None)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_episodes_json(path: Path) -> Iterator[dict[str, Any]]:
    """Stream recorded episodes as dicts. Tolerates a truncated final line."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_run(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a finished run's manifest and episodes."""
    path = Path(run_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; is {path} a run directory?")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = list(read_episodes_json(path / "episodes.jsonl"))
    return manifest, episodes


__all__ = ["RunRecorder", "load_run", "read_episodes_json"]
