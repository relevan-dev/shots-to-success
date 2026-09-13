"""Dataset adapter interface.

A dataset supplies three things: an information need per episode, a corpus to
search, and hidden relevance judgments used only by the grader. Implement this
class to adapt any IR dataset into benchmark episodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..types import Document, Judgments, Query


class Dataset(ABC):
    """Adapts an IR dataset into benchmark episodes."""

    #: Short name used on the command line and recorded in the manifest.
    name: ClassVar[str]
    #: Bumped whenever loading changes in a way that alters episodes.
    version: ClassVar[str] = "1"
    #: Grade at or above which a document counts as judged-relevant.
    default_relevant_threshold: ClassVar[int] = 1

    def __init__(self, data_dir: str | Path = "data", **options: Any) -> None:
        self.data_dir = Path(data_dir) / self.name
        self.options = options

    def prepare(self) -> None:
        """Download or materialize source files. Must be idempotent."""

    @abstractmethod
    def queries(self) -> list[Query]:
        """Information needs, in a stable order."""

    @abstractmethod
    def corpus(self) -> Iterator[Document]:
        """The searchable collection. Streamed so large corpora stay cheap."""

    @abstractmethod
    def judgments(self) -> Judgments:
        """Hidden relevance labels. Only the grader calls this."""

    def relevance_rubric(self) -> str:
        """What "satisfies the information need" means for this collection.

        Shown to the agent verbatim and identical across conditions. It aligns
        the agent's target with the hidden judgments -- without it the agent
        optimizes a different bar than the grader scores. It states a standard,
        never anything about specific documents.
        """
        return (
            "A document satisfies the information need if someone with that "
            "need would accept it as a correct result."
        )

    def describe(self) -> dict[str, Any]:
        """Goes into the run manifest; anything affecting episodes belongs here."""
        return {
            "name": self.name,
            "version": self.version,
            "options": self.options,
            "relevance_rubric": self.relevance_rubric(),
        }


__all__ = ["Dataset"]
