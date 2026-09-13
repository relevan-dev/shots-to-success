"""Template: wrapping a hosted search service as a benchmark adapter.

This is the shape most real backends take -- an HTTP API you do not control,
with a search endpoint and whatever extra endpoints the provider happens to
expose. Copy this file, point it at your service, and run it with:

    sts run --adapter examples.http_adapter:HttpSearchAdapter \
        --adapter-opt base_url=https://search.example.com \
        --condition tooling

Nothing needs to be registered and no benchmark code changes.

The three things that matter:

1. ``build`` is a no-op here. The service already has the corpus, so the
   adapter's job is to confirm that, not to ingest anything. If your service
   does need ingest, do it here and make it idempotent.

2. Every endpoint the provider exposes becomes a tool. The one that returns
   ranked documents is ``ToolKind.RETRIEVAL`` and spends a shot; the
   autocomplete and facet endpoints are ``ToolKind.AUXILIARY`` and are free.
   That split is what keeps "shots to success" meaningful when a provider
   gives the agent extra query-construction help.

3. Whatever your engine can say about *why* it returned these results --
   matched terms, analyzer output, filter counts, score explanation -- goes in
   ``ToolResponse.diagnostics``. The harness shows it only in feedback
   conditions, which is what the Feedback Track measures.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from shots_to_success.adapters import SearchAdapter
from shots_to_success.types import (
    Document,
    Hit,
    ResultSet,
    ToolKind,
    ToolResponse,
    ToolSpec,
)


class HttpSearchAdapter(SearchAdapter):
    """A hosted search service with search and autocomplete endpoints."""

    name: ClassVar[str] = "http-example"
    version: ClassVar[str] = "1"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.base_url = str(options.get("base_url", "http://localhost:8080")).rstrip("/")
        self.index = str(options.get("index", "products"))
        self.timeout = float(options.get("timeout", 20.0))
        self.api_key = options.get("api_key")

    # -- lifecycle -------------------------------------------------------

    def build(
        self,
        corpus: Iterable[Document],
        *,
        dataset_name: str,
        cache_key: str = "",
    ) -> None:
        """The service owns the corpus, so just check it is reachable.

        A no-op build is a legitimate implementation. What is not legitimate is
        silently searching a corpus that differs from the dataset's -- the
        judgments would no longer line up with the documents. Verify instead.
        """
        health = self._get("/health", {})
        indexed = health.get("document_count")
        if indexed is None:
            return
        expected = sum(1 for _ in corpus)
        if indexed != expected:
            raise RuntimeError(
                f"{self.name}: index holds {indexed} documents but dataset "
                f"{dataset_name!r} has {expected}. The judgments would not "
                "match the corpus; reindex before benchmarking."
            )

    # -- tool contract ---------------------------------------------------

    def tools(self) -> Sequence[ToolSpec]:
        return [
            ToolSpec(
                name="search",
                description=(
                    "Full-text search over the catalog. Returns ranked results "
                    "with a title and snippet."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "description": "Results to return."},
                        "category": {
                            "type": "string",
                            "description": "Optional category filter.",
                        },
                    },
                    "required": ["query"],
                },
                kind=ToolKind.RETRIEVAL,
                handler=self._search,
            ),
            ToolSpec(
                name="autocomplete",
                description=(
                    "Query completions for a prefix, from the provider's "
                    "suggester. Free -- it does not spend a search."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "prefix": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["prefix"],
                },
                kind=ToolKind.AUXILIARY,
                handler=self._autocomplete,
            ),
        ]

    # -- handlers --------------------------------------------------------

    def _search(self, args: Mapping[str, Any]) -> ToolResponse:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResponse(error="`query` must be a non-empty string.")
        params = {"q": query, "size": int(args.get("k") or 10), "index": self.index}
        if args.get("category"):
            params["category"] = str(args["category"])

        try:
            payload = self._get("/search", params)
        except Exception as exc:  # noqa: BLE001
            # Return the failure to the agent rather than raising: a search
            # that errors is a legitimate thing for it to recover from.
            return ToolResponse(error=f"search failed: {type(exc).__name__}: {exc}")

        hits = [
            Hit(
                doc_id=str(row["id"]),
                score=float(row.get("score", 0.0)),
                title=row.get("title", ""),
                snippet=row.get("snippet", ""),
                extra={"category": row.get("category", "")},
            )
            for row in payload.get("hits", [])
        ]
        return ToolResponse(
            body={
                "query": query,
                "total_matches": payload.get("total"),
                "results": [
                    {
                        "rank": i + 1,
                        "doc_id": hit.doc_id,
                        "title": hit.title,
                        "snippet": hit.snippet,
                        **hit.extra,
                    }
                    for i, hit in enumerate(hits)
                ],
            },
            result_set=ResultSet(hits=hits, query=query, total=payload.get("total")),
            # Shown to the agent only in feedback conditions.
            diagnostics={
                "analyzed_terms": payload.get("analyzed_terms"),
                "terms_with_no_matches": payload.get("unmatched_terms"),
                "documents_before_filters": payload.get("pre_filter_total"),
                "documents_after_filters": payload.get("total"),
                "score_explanation": payload.get("explain"),
            },
        )

    def _autocomplete(self, args: Mapping[str, Any]) -> ToolResponse:
        prefix = str(args.get("prefix") or "")
        if not prefix:
            return ToolResponse(error="`prefix` must be a non-empty string.")
        try:
            payload = self._get(
                "/suggest",
                {"q": prefix, "size": int(args.get("limit") or 10), "index": self.index},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(error=f"autocomplete failed: {exc}")
        return ToolResponse(
            body={"prefix": prefix, "suggestions": payload.get("suggestions", [])}
        )

    # -- transport -------------------------------------------------------

    def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url)
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        # Everything that could change ranking belongs in the manifest -- but
        # never the credential.
        info["base_url"] = self.base_url
        info["index"] = self.index
        info["options"] = {k: v for k, v in self.options.items() if k != "api_key"}
        return info
