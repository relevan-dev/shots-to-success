"""TREC Tip-of-the-Tongue (2023): known-item retrieval.

A searcher has seen a film but cannot remember its name, and describes it from
memory -- verbosely, vaguely, and often wrongly. The corpus is ~232k Wikipedia
pages for audiovisual works, and exactly one of them is the answer.

Two properties make this a good fit for a multi-shot search benchmark:

* A single BM25 query answers roughly one query in ten, so there is room for
  reformulation to show an effect.
* There is exactly one correct document per query and it is labelled, so
  scoring an unjudged result as non-relevant is simply true. Topical datasets
  with shallow judgment pools penalize an agent for surfacing documents the
  original pooling never saw -- which is the behavior this benchmark exists to
  reward.

Data (no usage agreement required): https://trec-tot.github.io/guidelines-2023
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..types import Document, Judgments, Query
from .base import Dataset

ARCHIVE_URL = "https://surfdrive.surf.nl/files/index.php/s/FaEK4xc6Xp2JcAJ/download"
ARCHIVE_NAME = "TREC-ToT.zip"
ROOT = "TREC-TOT"

#: Infobox parameters worth exposing as structured fields. These mirror what a
#: half-remembering searcher actually reaches for.
INFOBOX_FIELDS = ("country", "language", "genre", "director", "starring")
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]")
_MARKUP = re.compile(r"<[^>]+>|['\{\}]")
_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")
_TEMPLATE_HEAD = re.compile(r"^\s*(?:ubl|plainlist|flatlist|hlist|unbulleted list|nowrap)\s*\|", re.I)


class TrecTotDataset(Dataset):
    """TREC ToT 2023. Choose the split with ``--opt split=dev|train|train+dev``."""

    name: ClassVar[str] = "trec-tot"
    version: ClassVar[str] = "1"
    default_relevant_threshold: ClassVar[int] = 1

    def __init__(self, data_dir: str | Path = "data", **options: Any) -> None:
        super().__init__(data_dir, **options)
        self.split = str(options.get("split", "dev"))
        #: The corpus is 4.3 GB of full article text. Indexing a prefix keeps
        #: the reference backend usable on a laptop; it is part of the index
        #: definition, so it is reported in the manifest.
        self.max_text_chars = int(options.get("max_text_chars", 4000))

    @property
    def archive(self) -> Path:
        return self.data_dir / ARCHIVE_NAME

    # -- source files ----------------------------------------------------

    def prepare(self) -> None:
        if self.archive.exists() and self.archive.stat().st_size > 900_000_000:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"[trec-tot] downloading {ARCHIVE_URL} (~950 MB)", file=sys.stderr)
        tmp = self.archive.with_suffix(".part")
        urllib.request.urlretrieve(ARCHIVE_URL, tmp)
        tmp.replace(self.archive)

    def _open(self, member: str) -> Iterator[dict[str, Any]]:
        if not self.archive.exists():
            raise FileNotFoundError(
                f"{self.archive} is missing. Run "
                "`sts prepare --dataset trec-tot` first (~950 MB download)."
            )
        with zipfile.ZipFile(self.archive) as zf:
            with zf.open(f"{ROOT}/{member}") as handle:
                for raw in handle:
                    line = raw.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)

    def _splits(self) -> list[str]:
        return [part.strip() for part in self.split.split("+") if part.strip()]

    # -- benchmark surface -----------------------------------------------

    def queries(self) -> list[Query]:
        out: list[Query] = []
        for split in self._splits():
            for row in self._open(f"{split}/queries.jsonl"):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                # Only `text` crosses this boundary. The query records also
                # carry `wikipedia_id`, `wikipedia_url` and `imdb_url`, which
                # identify the answer -- putting any of them in metadata would
                # leak the label into the episode.
                out.append(
                    Query(
                        query_id=str(row["id"]),
                        text=text,
                        metadata={"split": split},
                    )
                )
        out.sort(key=lambda q: int(q.query_id))
        return out

    def corpus(self) -> Iterator[Document]:
        for row in self._open("corpus.jsonl"):
            text = (row.get("text") or "")[: self.max_text_chars]
            fields: dict[str, Any] = {"description": text[:1200]}

            classes = row.get("wikidata_classes") or []
            labels = [c[1] for c in classes if isinstance(c, list) and len(c) > 1]
            if labels:
                fields["type"] = labels[0]

            infoboxes = row.get("infoboxes") or []
            params = infoboxes[0].get("params", {}) if infoboxes else {}
            for key in INFOBOX_FIELDS:
                value = _clean_markup(params.get(key))
                if value:
                    fields[key] = value
            year = _first_year(params)
            if year:
                fields["year"] = year
                fields["decade"] = f"{year[:3]}0s"

            yield Document(
                doc_id=str(row["doc_id"]),
                title=(row.get("page_title") or "").strip(),
                text=text,
                fields=fields,
            )

    def judgments(self) -> Judgments:
        grades: dict[str, dict[str, int]] = {}
        for split in self._splits():
            with zipfile.ZipFile(self.archive) as zf:
                with zf.open(f"{ROOT}/{split}/qrel.txt") as handle:
                    for raw in handle:
                        parts = raw.decode("utf-8").split()
                        if len(parts) < 4:
                            continue
                        query_id, doc_id, grade = parts[0], parts[2], int(parts[3])
                        if grade > 0:
                            grades.setdefault(query_id, {})[doc_id] = grade
        threshold = int(
            self.options.get("relevant_threshold", self.default_relevant_threshold)
        )
        return Judgments(grades=grades, relevant_threshold=threshold, max_grade=1)

    def relevance_rubric(self) -> str:
        return (
            "This is a collection of Wikipedia pages for films and other "
            "audiovisual works. Someone is describing a film they have seen but "
            "cannot name. Exactly one page in the collection is the film they "
            "mean, and it satisfies the information need; every other page, "
            "however similar, does not. Expect the description to be vague, out "
            "of order, and partly wrong -- remembered details such as the "
            "decade, an actor, or the ending are often mistaken, so treat them "
            "as hints rather than constraints."
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["split"] = self.split
        info["max_text_chars"] = self.max_text_chars
        return info


def _clean_markup(value: Any) -> str:
    """Strip wiki markup: ``[[Ventura Pons]]`` becomes ``Ventura Pons``."""
    if not isinstance(value, str) or not value.strip():
        return ""
    text = _WIKI_LINK.sub(r"\1", value)
    text = _MARKUP.sub(" ", text)
    # Infobox values are often wrapped in list templates ("ubl|a|b|c").
    text = _TEMPLATE_HEAD.sub("", text).replace("|", ", ")
    return " ".join(text.split()).strip(" ,")[:120]


def _first_year(params: dict[str, Any]) -> str:
    for key in ("released", "release_date", "first_aired", "date"):
        match = _YEAR.search(str(params.get(key) or ""))
        if match:
            return match.group(1)
    return ""


__all__ = ["TrecTotDataset"]
