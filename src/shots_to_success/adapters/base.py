"""Search adapter interface -- the benchmark's extension point.

An adapter wraps a search backend and exposes it to the agent as a set of
tools. The harness never assumes anything about how retrieval works; it only
needs each tool to declare whether calling it spends a shot and, for retrieval
tools, to hand back a ranked :class:`~shots_to_success.types.ResultSet`.

Implementing one:

    from shots_to_success.adapters import SearchAdapter
    from shots_to_success.types import (
        Hit, ResultSet, ToolKind, ToolResponse, ToolSpec,
    )

    class MyAdapter(SearchAdapter):
        name = "my-engine"

        def build(self, corpus, *, dataset_name, cache_key=""):
            ...  # ingest, or no-op if the backend is already populated

        def tools(self):
            return [ToolSpec(
                name="search",
                description="Full-text search over the product catalog.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                kind=ToolKind.RETRIEVAL,
                handler=self._search,
            )]

        def _search(self, args):
            hits = [Hit(doc_id=d, score=s) for d, s in my_backend(args["query"])]
            return ToolResponse(
                body={"hits": len(hits)},
                result_set=ResultSet(hits=hits, query=args["query"]),
                diagnostics={"matched_terms": [...]},  # feedback conditions only
            )

Register it by passing ``--adapter mypkg.module:MyAdapter`` -- no changes to
this repository are required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Sequence

from ..types import Document, ResultSet, ToolKind, ToolSpec


class SearchAdapter(ABC):
    """Wraps a search backend and exposes it as agent tools."""

    #: Short name used on the command line and recorded in the manifest.
    name: ClassVar[str]
    #: Bumped whenever behavior changes in a way that alters results.
    version: ClassVar[str] = "1"

    def __init__(self, **options: Any) -> None:
        self.options = options

    # -- lifecycle -------------------------------------------------------

    def build(
        self,
        corpus: Iterable[Document],
        *,
        dataset_name: str,
        cache_key: str = "",
    ) -> None:
        """Ingest the corpus.

        ``cache_key`` is a stable hash of the dataset configuration; adapters
        that persist an index can use it to skip a rebuild. Default is a
        no-op so adapters pointing at an already-populated backend do not have
        to pretend to index anything.
        """

    def close(self) -> None:
        """Release connections or file handles."""

    # -- tool contract ---------------------------------------------------

    @abstractmethod
    def tools(self) -> Sequence[ToolSpec]:
        """Every tool this backend exposes.

        The harness filters this list by condition -- auxiliary tools are
        hidden unless the condition allows them -- so return the full surface
        and let the condition decide.
        """

    def retrieval_tools(self) -> list[ToolSpec]:
        return [t for t in self.tools() if t.kind is ToolKind.RETRIEVAL]

    def primary_tool(self) -> ToolSpec:
        """The retrieval tool used for the static baseline.

        Defaults to the first retrieval tool declared.
        """
        retrieval = self.retrieval_tools()
        if not retrieval:
            raise ValueError(f"adapter {self.name!r} declares no retrieval tool")
        return retrieval[0]

    def baseline(self, query: str, k: int) -> ResultSet:
        """Run the unmodified benchmark query once, for the static baseline.

        Override if the primary tool's schema is not ``{"query", "k"}``.
        """
        tool = self.primary_tool()
        response = tool.handler({"query": query, "k": k})
        if response.result_set is None:
            raise ValueError(
                f"retrieval tool {tool.name!r} returned no result set"
            )
        return response.result_set

    # -- provenance ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Goes into the run manifest. Include anything affecting ranking."""
        return {
            "name": self.name,
            "version": self.version,
            "options": self.options,
            "tools": [
                {"name": t.name, "kind": t.kind.value, "feedback_only": t.feedback_only}
                for t in self.tools()
            ],
        }


__all__ = ["SearchAdapter"]
