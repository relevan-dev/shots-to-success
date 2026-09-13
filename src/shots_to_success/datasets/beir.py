"""BEIR adapter: 15+ standard IR datasets behind one loader.

BEIR distributes every dataset in the same three-file shape, so one loader
covers TREC-COVID, NFCorpus, FiQA, SciFact, SCIDOCS, ArguAna, Touche, DBPedia,
Quora, HotpotQA, NQ, FEVER, Climate-FEVER, MS MARCO, and SciDocs:

    corpus.jsonl    {"_id", "title", "text", "metadata"}
    queries.jsonl   {"_id", "text", "metadata"}
    qrels/test.tsv  query-id  corpus-id  score

Pick one with ``--opt subset=fiqa``.

https://github.com/beir-cellar/beir
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..types import Document, Judgments, Query
from .base import Dataset

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

#: Subsets that download and index comfortably on a laptop. Larger BEIR
#: datasets (msmarco, hotpotqa, fever, dbpedia-entity, nq, climate-fever) work
#: the same way but need a real search backend behind the adapter.
SMALL_SUBSETS = (
    "nfcorpus",
    "scifact",
    "arguana",
    "scidocs",
    "fiqa",
    "trec-covid",
    "quora",
    "webis-touche2020",
)

#: What "relevant" means, per subset. Shown to the agent verbatim, identical
#: across conditions. Generic fallback used for anything not listed.
RUBRICS: dict[str, str] = {
    "nfcorpus": (
        "This is a collection of PubMed medical abstracts, searched with "
        "questions written by non-specialists. A document satisfies the "
        "information need if it reports evidence that answers the question. "
        "The collection uses clinical and biochemical terminology, while the "
        "question may use everyday language for the same concept."
    ),
    "fiqa": (
        "This is a collection of financial discussion posts. A document "
        "satisfies the information need if it directly answers the question "
        "asked. Posts on the same general financial topic that do not answer "
        "the question do not count."
    ),
    "scifact": (
        "This is a collection of scientific paper abstracts. A document "
        "satisfies the information need if it contains evidence that supports "
        "or refutes the claim, not merely if it discusses the same topic."
    ),
    "scidocs": (
        "This is a collection of scientific paper abstracts. A document "
        "satisfies the information need if it is a paper that the given paper "
        "would cite or that addresses the same specific contribution."
    ),
    "trec-covid": (
        "This is a collection of COVID-19 research abstracts. A document "
        "satisfies the information need if it reports findings that answer the "
        "question, not merely if it mentions the same terms."
    ),
    "arguana": (
        "This is a collection of debate arguments. A document satisfies the "
        "information need if it is a counterargument to the argument given."
    ),
    "webis-touche2020": (
        "This is a collection of debate portal arguments. A document satisfies "
        "the information need if it argues a position on the controversial "
        "question asked."
    ),
    "quora": (
        "This is a collection of questions. A document satisfies the "
        "information need if it is a duplicate of the question asked -- the "
        "same question phrased differently."
    ),
}


class BeirDataset(Dataset):
    """A BEIR dataset. Choose the subset with ``--opt subset=<name>``."""

    name: ClassVar[str] = "beir"
    version: ClassVar[str] = "1"
    #: BEIR qrels are mostly binary; grade 1 and above is relevant.
    default_relevant_threshold: ClassVar[int] = 1

    def __init__(self, data_dir: str | Path = "data", **options: Any) -> None:
        super().__init__(data_dir, **options)
        self.subset = str(options.get("subset", "fiqa"))
        if "/" in self.subset:
            raise ValueError(
                "nested BEIR subsets such as cqadupstack are not supported; "
                f"got {self.subset!r}"
            )
        self.split = str(options.get("split", "test"))
        # Each subset gets its own directory so several can coexist, and its
        # own index cache, because `describe()` feeds the adapter's cache key.
        self.data_dir = Path(data_dir) / self.name / self.subset

    # -- source files ----------------------------------------------------

    def prepare(self) -> None:
        if (self.data_dir / "corpus.jsonl").exists():
            return
        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{self.subset}.zip"
        archive = self.data_dir.parent / f"{self.subset}.zip"
        print(f"[beir] downloading {url}", file=sys.stderr)
        try:
            urllib.request.urlretrieve(url, archive)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"could not download BEIR subset {self.subset!r} from {url}. "
                f"Known small subsets: {', '.join(SMALL_SUBSETS)}. ({exc})"
            ) from exc
        print(f"[beir] extracting {archive.name}", file=sys.stderr)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(self.data_dir.parent)
        archive.unlink()
        if not (self.data_dir / "corpus.jsonl").exists():
            raise RuntimeError(
                f"{self.subset}.zip did not contain the expected layout at "
                f"{self.data_dir}"
            )

    def _jsonl(self, filename: str) -> Iterator[dict[str, Any]]:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run "
                f"`sts prepare --dataset beir --opt subset={self.subset}` first."
            )
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # -- benchmark surface -----------------------------------------------

    def queries(self) -> list[Query]:
        # Only queries carrying judgments in this split are episodes; BEIR
        # ships the full query file regardless of split.
        judged = set(self.judgments().grades)
        out = [
            Query(query_id=str(row["_id"]), text=(row.get("text") or "").strip())
            for row in self._jsonl("queries.jsonl")
            if str(row["_id"]) in judged and (row.get("text") or "").strip()
        ]
        out.sort(key=lambda q: q.query_id)
        return out

    def corpus(self) -> Iterator[Document]:
        for row in self._jsonl("corpus.jsonl"):
            metadata = row.get("metadata") or {}
            yield Document(
                doc_id=str(row["_id"]),
                title=(row.get("title") or "").strip(),
                text=(row.get("text") or "").strip(),
                fields={
                    "description": (row.get("text") or "").strip(),
                    **{
                        key: value
                        for key, value in metadata.items()
                        if isinstance(value, (str, int, float))
                    },
                },
            )

    def judgments(self) -> Judgments:
        path = self.data_dir / "qrels" / f"{self.split}.tsv"
        if not path.exists():
            available = sorted(
                p.stem for p in (self.data_dir / "qrels").glob("*.tsv")
            ) if (self.data_dir / "qrels").exists() else []
            raise FileNotFoundError(
                f"no qrels for split {self.split!r} at {path}"
                + (f"; available: {', '.join(available)}" if available else "")
            )
        grades: dict[str, dict[str, int]] = {}
        max_grade = 1
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or (i == 0 and parts[0] == "query-id"):
                    continue
                query_id, doc_id, score = parts[0], parts[1], parts[2]
                try:
                    grade = int(float(score))
                except ValueError:
                    continue
                grades.setdefault(query_id, {})[doc_id] = grade
                max_grade = max(max_grade, grade)
        threshold = int(
            self.options.get("relevant_threshold", self.default_relevant_threshold)
        )
        return Judgments(
            grades=grades,
            relevant_threshold=threshold,
            max_grade=max_grade,
        )

    def relevance_rubric(self) -> str:
        return RUBRICS.get(self.subset, super().relevance_rubric())

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["subset"] = self.subset
        info["split"] = self.split
        return info


__all__ = ["BeirDataset", "SMALL_SUBSETS"]
