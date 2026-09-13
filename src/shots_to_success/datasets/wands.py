"""WANDS (Wayfair ANnotation DataSet) adapter.

~480 queries over ~43k products with three-level graded judgments. Source
files are tab-separated despite the ``.csv`` extension.

https://github.com/wayfair/WANDS
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..types import Document, Judgments, Query
from .base import Dataset

BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
FILES = ("product.csv", "query.csv", "label.csv")

#: WANDS ships ordinal labels; map them to graded relevance for nDCG.
GRADES = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
GRADE_NAMES = {2: "Exact", 1: "Partial", 0: "Irrelevant"}


class WandsDataset(Dataset):
    """WANDS product search."""

    name: ClassVar[str] = "wands"
    version: ClassVar[str] = "1"
    #: Default to Exact-only. "Partial" in WANDS covers same-category items
    #: that do not satisfy the need, so counting them as success would make
    #: Success@10 trivially easy for a category-matching baseline.
    default_relevant_threshold: ClassVar[int] = 2

    def __init__(self, data_dir: str | Path = "data", **options: Any) -> None:
        super().__init__(data_dir, **options)
        #: Cap indexed description length. Recorded in the manifest because it
        #: is part of the index definition, not a display choice.
        self.max_description_chars = int(options.get("max_description_chars", 4000))

    # -- source files ----------------------------------------------------

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            target = self.data_dir / filename
            if target.exists() and target.stat().st_size > 0:
                continue
            url = f"{BASE_URL}/{filename}"
            print(f"[wands] downloading {url}", file=sys.stderr)
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)

    def _rows(self, filename: str) -> Iterator[dict[str, str]]:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `sts prepare --dataset wands` first."
            )
        # Product descriptions contain unescaped quotes; QUOTE_NONE keeps the
        # tab-delimited parse honest instead of swallowing whole columns.
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(
                handle, delimiter="\t", quoting=csv.QUOTE_NONE
            )

    # -- benchmark surface -----------------------------------------------

    def queries(self) -> list[Query]:
        out: list[Query] = []
        for row in self._rows("query.csv"):
            query_id = (row.get("query_id") or "").strip()
            text = (row.get("query") or "").strip()
            if not query_id or not text:
                continue
            out.append(
                Query(
                    query_id=query_id,
                    text=text,
                    # query_class is the annotator's product category. It is
                    # metadata about the query, not a judgment, but it is not
                    # shown to the agent either -- a real user would not have it.
                    metadata={"query_class": (row.get("query_class") or "").strip()},
                )
            )
        out.sort(key=lambda q: int(q.query_id))
        return out

    def corpus(self) -> Iterator[Document]:
        for row in self._rows("product.csv"):
            doc_id = (row.get("product_id") or "").strip()
            if not doc_id:
                continue
            name = _clean(row.get("product_name"))
            description = _clean(row.get("product_description"))
            if len(description) > self.max_description_chars:
                description = description[: self.max_description_chars]
            features = _clean(row.get("product_features"))
            product_class = _clean(row.get("product_class"))
            category = _clean(row.get("category hierarchy"))
            yield Document(
                doc_id=doc_id,
                title=name,
                text=" ".join(x for x in (description, features) if x),
                fields={
                    "product_class": product_class,
                    "category": category,
                    "description": description,
                    "features": features,
                    "rating": _to_float(row.get("average_rating")),
                    "review_count": _to_int(row.get("review_count")),
                },
            )

    def judgments(self) -> Judgments:
        grades: dict[str, dict[str, int]] = {}
        for row in self._rows("label.csv"):
            query_id = (row.get("query_id") or "").strip()
            doc_id = (row.get("product_id") or "").strip()
            label = (row.get("label") or "").strip()
            if not query_id or not doc_id or label not in GRADES:
                continue
            grades.setdefault(query_id, {})[doc_id] = GRADES[label]
        threshold = int(
            self.options.get("relevant_threshold", self.default_relevant_threshold)
        )
        return Judgments(
            grades=grades,
            relevant_threshold=threshold,
            max_grade=max(GRADES.values()),
            grade_names=GRADE_NAMES,
        )

    def relevance_rubric(self) -> str:
        return (
            "This is a furniture and home goods catalog. A product satisfies "
            "the information need only if it is the specific kind of item the "
            "shopper asked for. Products from a related, parent, or "
            "complementary category do not count -- a chair is not a stool, a "
            "pillow cover is not a pillow, and a table lamp is not a floor lamp."
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["max_description_chars"] = self.max_description_chars
        return info


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.replace("\\n", " ").split())


def _to_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(value: str | None) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


__all__ = ["WandsDataset"]
