"""Shared fixtures: a tiny corpus and a scripted stand-in for the Claude client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import pytest

from shots_to_success.adapters.bm25 import Bm25Adapter
from shots_to_success.datasets.base import Dataset
from shots_to_success.types import Document, Judgments, Query

CORPUS = [
    ("1", "red velvet accent chair", "Accent Chairs", "a plush red chair for the study"),
    ("2", "blue linen accent chair", "Accent Chairs", "a calm blue chair"),
    ("3", "oak dining table", "Dining Tables", "solid oak table seats six"),
    ("4", "red table lamp", "Table Lamps", "a small red lamp for a side table"),
    ("5", "chair cushion set", "Cushions", "cushions that fit most chairs"),
]


class TinyDataset(Dataset):
    """Five documents, two queries, hand-written judgments."""

    name = "tiny"
    default_relevant_threshold = 2

    def queries(self) -> list[Query]:
        return [
            Query(query_id="q1", text="red chair"),
            Query(query_id="q2", text="oak table"),
        ]

    def corpus(self) -> Iterator[Document]:
        for doc_id, title, product_class, text in CORPUS:
            yield Document(
                doc_id=doc_id,
                title=title,
                text=text,
                fields={
                    "product_class": product_class,
                    "category": f"Furniture / {product_class}",
                    "description": text,
                    "rating": 4.0,
                },
            )

    def judgments(self) -> Judgments:
        return Judgments(
            grades={"q1": {"1": 2, "2": 1, "5": 1, "4": 0}, "q2": {"3": 2}},
            relevant_threshold=2,
            max_grade=2,
        )


@pytest.fixture
def dataset() -> TinyDataset:
    return TinyDataset(data_dir="/nonexistent")


@pytest.fixture
def adapter(dataset: TinyDataset) -> Bm25Adapter:
    built = Bm25Adapter(use_cache=False)
    built.build(dataset.corpus(), dataset_name="tiny")
    return built


# -- a scripted stand-in for anthropic.Anthropic ---------------------------


@dataclass
class FakeBlock:
    type: str
    text: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    id: str = "tu_0"


@dataclass
class FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 5
    cache_read_input_tokens: int = 0


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str = "tool_use"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self, script: list[FakeResponse]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.script:
            # Ran off the end of the script: behave like a model that stopped.
            return FakeResponse(content=[FakeBlock(type="text", text="done")],
                                stop_reason="end_turn")
        return self.script.pop(0)


class FakeClient:
    """Replays a fixed list of responses in place of the real API."""

    def __init__(self, script: list[FakeResponse]) -> None:
        self.messages = FakeMessages(script)


def text(value: str) -> FakeBlock:
    return FakeBlock(type="text", text=value)


def call(name: str, block_id: str = "tu_0", **args: Any) -> FakeBlock:
    return FakeBlock(type="tool_use", name=name, input=args, id=block_id)


def turn(*blocks: FakeBlock, stop: str = "tool_use") -> FakeResponse:
    return FakeResponse(content=list(blocks), stop_reason=stop)
