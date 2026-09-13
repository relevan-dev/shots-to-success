"""Relevan adapter: the hosted search API at api.relevan.dev as a backend.

Unlike the bundled BM25 adapter, this one talks to a service over HTTP. That
changes what ``build`` has to do -- create an index, deploy a mapping derived
from the dataset, bulk-ingest the corpus, and wait for the asynchronous index
pipeline to catch up -- but nothing about the tool contract.

The tool surface follows what the API actually offers:

- ``search`` (retrieval) -- ``POST /v1/{org}/indexes/{index}/search``. Filters
  and per-term boosts are part of the search contract, so they are part of the
  tool.
- ``describe_index`` (auxiliary) -- the index's own agent-oriented skill doc,
  generated from its live mapping. Free: it returns no documents.
- ``explain_result`` (auxiliary, feedback-only) -- why one already-seen
  document scores the way it does, via a search pinned to that document id.
- ``report_relevant_result`` (auxiliary) -- only when ``send_feedback=true``;
  see the note below.

Two things are worth knowing before running this against a real org.

**Feedback is off by default.** Relevan asks every caller to report what it did
with a search result, because that is what trains its ranking model. But a
benchmark needs the ranking to hold still: "two runs of the same query must
return the same ranking" (docs/ADAPTERS.md). Reporting judgments mid-run makes
episode order affect episode results. So ``send_feedback`` defaults to false,
and when it is enabled every report is sent with ``source="eval"`` -- the value
the API defines for a test suite or benchmark run rather than real traffic. The
tool reports the *agent's own* judgment; the hidden relevance labels are never
visible to an adapter and are never sent.

**Document ids round-trip.** Relevan's ids allow ``[A-Za-z0-9_.:-]`` only, and
grading requires the ids handed back to match the dataset's exactly. Anything
outside that set is base32-encoded on the way in and decoded on the way out;
the original is also stored in the document body as a belt-and-braces copy.

Run it with::

    export RELEVAN_API_KEY=...
    sts run --adapter relevan --dataset trec-tot --condition feedback

See docs/RELEVAN.md for the options and the ingest path.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Sequence

from ..types import Document, Hit, ResultSet, ToolKind, ToolResponse, ToolSpec
from .base import SearchAdapter

DEFAULT_BASE_URL = "https://api.relevan.dev"
#: Documents per bulk request, and results per search -- both API caps.
BULK_MAX = 1000
SIZE_MAX = 100
#: Ids the API accepts verbatim; anything else is encoded (see `_encode_id`).
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
ID_MAX = 512
ENCODED_PREFIX = "b32."
INDEX_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
#: The dataset's own id, carried in the document body so a hit can name it
#: without relying on the encoding above.
ID_FIELD = "sts_doc_id"
#: A string field becomes filterable below this many distinct values, and only
#: if its values are short enough to be labels rather than prose.
MAX_FILTER_CARDINALITY = 5000
MAX_FILTER_VALUE_CHARS = 100
SNIPPET_CHARS = 220
#: Longest a 429's Retry-After will be honored before giving up on the wait.
MAX_RETRY_SLEEP = 120.0
#: The English analyzer drops these, so they never appear among matched terms.
#: Reporting them as "matched nothing" would be a false signal on every search.
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with".split()
)


class RelevanApiError(RuntimeError):
    """A non-2xx response, carrying the API's own message."""

    def __init__(self, status: int, message: str, payload: Any = None) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message
        self.payload = payload if isinstance(payload, dict) else {}


class RelevanAdapter(SearchAdapter):
    """Relevan's hosted search API (api.relevan.dev)."""

    name: ClassVar[str] = "relevan"
    version: ClassVar[str] = "1"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.base_url = str(options.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = options.get("api_key") or os.environ.get("RELEVAN_API_KEY")
        self.org = options.get("org") or options.get("org_id")
        self.index_name = options.get("index")
        self.timeout = float(options.get("timeout", 30.0))
        self.max_retries = int(options.get("max_retries", 3))
        self.batch_size = max(1, min(int(options.get("batch_size", BULK_MAX)), BULK_MAX))
        #: How many documents to read before fixing the mapping. Fields that
        #: appear only after this point are dropped rather than risking a
        #: mapping-validation rejection of a whole batch.
        self.sample_size = max(1, int(options.get("sample_size", 1000)))
        self.language = str(options.get("language", "english"))
        self.reindex = bool(options.get("reindex", False))
        self.send_feedback = bool(options.get("send_feedback", False))
        self.poll_interval = float(options.get("poll_interval", 5.0))
        self.poll_timeout = float(options.get("poll_timeout", 1800.0))

        self.filter_fields: list[str] = []
        self.range_fields: list[str] = []
        self.searchable_fields: list[str] = []
        #: Whether the dataset's own id is filterable, which is what lets
        #: `explain_result` pin a search to one document.
        self._id_filterable = False
        self._field_kinds: dict[str, str] = {}
        self._declared: set[str] = set()
        self._mapping_revision: int | None = None
        self._mapping_hash: str | None = None
        self._skill_doc: str | None = None
        self._documents_indexed: int | None = None
        # Filled by the most recent search, so feedback can be attributed to it.
        self._last_query_id: str | None = None
        self._last_positions: dict[str, int] = {}
        self._built = False

    # -- build -----------------------------------------------------------

    def build(
        self,
        corpus: Iterable[Document],
        *,
        dataset_name: str,
        cache_key: str = "",
    ) -> None:
        """Create the index, deploy a mapping, ingest, and wait for readiness.

        The default index name embeds ``cache_key`` -- a hash of the dataset
        configuration and this adapter's version -- so an index that already
        exists under that name holds this exact corpus. That is the cache
        check: it costs one API call, where counting the corpus would cost a
        full scan of a multi-gigabyte dataset. Pass ``reindex=true`` to ingest
        again anyway (documents upsert by id).
        """

        self._resolve_org()
        if not self.index_name:
            self.index_name = _default_index_name(dataset_name, cache_key)
        if not INDEX_NAME.match(self.index_name) or len(self.index_name) > 64:
            raise ValueError(
                f"{self.name}: index name {self.index_name!r} is not URL-safe "
                "(lowercase alphanumeric and hyphens, max 64 characters)"
            )

        detail = self._get_index()
        if detail is None:
            self._create_index(dataset_name)
        elif not self.reindex and detail.get("documentsAccepted"):
            if detail.get("readyToSearch"):
                self._load_mapping()
                self._documents_indexed = detail.get("documentsIndexed")
                self._built = True
                self._log(
                    f"reusing index {self.index_name!r} "
                    f"({detail.get('documentsIndexed')} documents indexed)"
                )
                return
            self._log(
                f"index {self.index_name!r} holds "
                f"{detail.get('documentsAccepted')} documents but is still "
                "indexing; resuming ingest"
            )

        self._ingest(corpus)
        self._wait_until_ready()
        self._load_mapping()
        self._built = True

    def close(self) -> None:
        self._skill_doc = None

    # -- tool contract ---------------------------------------------------

    def tools(self) -> Sequence[ToolSpec]:
        properties: dict[str, Any] = {
            "query": {
                "type": "string",
                "description": "Free-text keywords to search for.",
            },
            "k": {
                "type": "integer",
                "description": f"How many results to return (max {SIZE_MAX}).",
            },
            "boost": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": (
                    "Per-term weight multipliers, e.g. {\"tetra\": 2}. Adds an "
                    "extra match clause for each term across the searchable "
                    "fields; it does not change how the base query matches."
                ),
            },
        }
        if self.filter_fields or self.range_fields:
            properties["filters"] = self._filter_schema()

        specs = [
            ToolSpec(
                name="search",
                description=(
                    "Full-text search over the collection. Returns a ranked list "
                    "of results with a title and snippet for each."
                    + (
                        " Optional filters narrow the candidate set before "
                        "ranking."
                        if self.filter_fields or self.range_fields
                        else ""
                    )
                ),
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": ["query"],
                },
                kind=ToolKind.RETRIEVAL,
                handler=self._search,
            ),
            ToolSpec(
                name="describe_index",
                description=(
                    "The index's own documentation: which fields exist, which "
                    "are searchable or filterable, and how to query them. Use "
                    "it before spending a search on a collection you have not "
                    "seen."
                ),
                input_schema={"type": "object", "properties": {}},
                kind=ToolKind.AUXILIARY,
                handler=self._describe_index,
            ),
        ]
        if self._id_filterable:
            specs.append(
                ToolSpec(
                    name="explain_result",
                    description=(
                        "Why a specific document scores the way it does for a "
                        "query: which fields and indexed terms contributed, and "
                        "whether it matches the query at all."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "doc_id": {"type": "string"},
                        },
                        "required": ["query", "doc_id"],
                    },
                    kind=ToolKind.AUXILIARY,
                    handler=self._explain,
                    feedback_only=True,
                )
            )
        if self.send_feedback:
            specs.append(
                ToolSpec(
                    name="report_relevant_result",
                    description=(
                        "Report that a result from your last search satisfies "
                        "the information need. This is your own judgment, and "
                        "it does not change the current results."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string"},
                            "position": {
                                "type": "integer",
                                "description": "Its rank in the results, if known.",
                            },
                        },
                        "required": ["doc_id"],
                    },
                    kind=ToolKind.AUXILIARY,
                    handler=self._report_relevant,
                )
            )
        return specs

    def _filter_schema(self) -> dict[str, Any]:
        scalar = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]}
        return {
            "type": "array",
            "description": (
                "Filters, ANDed together. Each is an exact match (`value`), a "
                "set membership (`values`), or a range (`gte`/`lte`, numeric "
                "and date fields only)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": sorted(set(self.filter_fields) | set(self.range_fields)),
                    },
                    "value": scalar,
                    "values": {"type": "array", "items": scalar},
                    "gte": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                    "lte": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                },
                "required": ["field"],
            },
        }

    # -- handlers --------------------------------------------------------

    def _search(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResponse(error="`query` must be a non-empty string.")
        size = max(1, min(int(args.get("k") or 10), SIZE_MAX))

        filters, filter_error = self._build_filters(args.get("filters"))
        if filter_error:
            return ToolResponse(error=filter_error)
        boost, boost_error = _build_boost(args.get("boost"))
        if boost_error:
            return ToolResponse(error=boost_error)

        payload: dict[str, Any] = {"query": query, "size": size}
        if filters:
            payload["filters"] = filters
        if boost:
            payload["boost"] = boost

        try:
            data = self._post(self._index_path("/search"), payload)
        except RelevanApiError as exc:
            return ToolResponse(error=self._search_error(exc))
        except Exception as exc:  # noqa: BLE001 - the agent should see and recover
            return ToolResponse(error=f"search failed: {type(exc).__name__}: {exc}")

        context = dict(data.get("searchContext") or {})
        rows = list(data.get("results") or [])
        hits = [self._to_hit(row) for row in rows]
        total = context.get("totalHits")

        self._last_query_id = context.get("queryId")
        self._last_positions = {hit.doc_id: i for i, hit in enumerate(hits)}

        body = {
            "query": query,
            "returned": len(hits),
            "total_matches": total,
            "results": [
                {
                    "rank": i + 1,
                    "doc_id": hit.doc_id,
                    "title": hit.title,
                    "snippet": hit.snippet,
                    **_display_fields(hit),
                }
                for i, hit in enumerate(hits)
            ],
        }
        return ToolResponse(
            body=body,
            result_set=ResultSet(hits=hits, query=query, total=total),
            diagnostics=self._diagnostics(query, context, rows, filters),
        )

    def _describe_index(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        body: dict[str, Any] = {
            "index": self.index_name,
            "documents": self._documents_indexed,
            "searchable_fields": self.searchable_fields,
            "filterable_fields": self.filter_fields,
            "range_fields": self.range_fields,
        }
        try:
            body["guide"] = self._skill()
        except RelevanApiError as exc:
            body["guide"] = f"(the index guide is unavailable: {exc.message})"
        except Exception as exc:  # noqa: BLE001
            body["guide"] = f"(the index guide is unavailable: {exc})"
        return ToolResponse(body=body)

    def _explain(self, args: Mapping[str, Any]) -> ToolResponse:
        """Score explanation for one document, via a search pinned to its id.

        This spends an API request but not a shot: it returns one document the
        agent has already seen, so it cannot stand in for a search.
        """
        self._require_built()
        query = str(args.get("query") or "").strip()
        doc_id = str(args.get("doc_id") or "")
        if not query or not doc_id:
            return ToolResponse(error="`query` and `doc_id` are both required.")
        try:
            data = self._post(
                self._index_path("/search"),
                {
                    "query": query,
                    "size": 1,
                    # ID_FIELD holds the dataset's own id, not the encoded one
                    # the API knows the document by.
                    "filters": [{"field": ID_FIELD, "value": doc_id}],
                },
            )
        except RelevanApiError as exc:
            return ToolResponse(error=self._search_error(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(error=f"explain failed: {type(exc).__name__}: {exc}")

        rows = list(data.get("results") or [])
        if not rows:
            return ToolResponse(
                body={
                    "doc_id": doc_id,
                    "query": query,
                    "matched": False,
                    "note": (
                        "This document does not match the query at all -- it "
                        "shares no indexed term with it."
                    ),
                }
            )
        row = rows[0]
        context = dict(data.get("searchContext") or {})
        matched_terms = [str(t) for t in row.get("matchedTerms") or []]
        return ToolResponse(
            body={
                "doc_id": doc_id,
                "query": query,
                "matched": True,
                "score": row.get("score"),
                "explanation": row.get("explanation"),
                "matched_fields": row.get("matchedFields") or [],
                "matched_terms": matched_terms,
                "query_terms_with_no_match": _unmatched_terms(query, matched_terms),
                "scoring_summary": context.get("scoringSummary"),
            }
        )

    def _report_relevant(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        doc_id = str(args.get("doc_id") or "")
        if not doc_id:
            return ToolResponse(error="`doc_id` must be a non-empty string.")
        if self._last_query_id is None:
            return ToolResponse(
                error="Run a search before reporting a result from one."
            )
        position = args.get("position")
        if position is None:
            position = self._last_positions.get(doc_id)
        payload: dict[str, Any] = {
            "actionName": "llm_judgment",
            # Never "production": these judgments come from a benchmark run,
            # not from a real user or agent session.
            "source": "eval",
            "queryId": self._last_query_id,
            "objectId": _encode_id(doc_id),
        }
        if position is not None:
            payload["position"] = max(0, int(position))
        try:
            self._post(self._index_path("/feedback", version="v1-beta"), payload)
        except RelevanApiError as exc:
            return ToolResponse(error=f"feedback was not accepted: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(error=f"feedback failed: {type(exc).__name__}: {exc}")
        return ToolResponse(body={"reported": doc_id, "recorded": True})

    # -- result shaping --------------------------------------------------

    def _to_hit(self, row: Mapping[str, Any]) -> Hit:
        content = dict(row.get("document") or {})
        doc_id = content.get(ID_FIELD) or _decode_id(str(row.get("id") or ""))
        highlights = row.get("highlights") or {}
        extra: dict[str, Any] = {
            key: _truncate(value)
            for key, value in content.items()
            if key not in (ID_FIELD, "title", "text")
            and isinstance(value, (str, int, float, bool))
            and value not in ("", None)
        }
        score = row.get("score")
        extra["score"] = round(float(score), 4) if score is not None else None
        # Recorded on the hit, but rendered only through diagnostics -- the
        # explanation is feedback, and feedback is the variable under test.
        extra["explanation"] = row.get("explanation")
        extra["matched_fields"] = list(row.get("matchedFields") or [])
        extra["matched_terms"] = list(row.get("matchedTerms") or [])
        return Hit(
            doc_id=str(doc_id),
            score=float(score) if score is not None else 0.0,
            title=str(content.get("title") or ""),
            snippet=_snippet(highlights, content),
            extra=extra,
        )

    def _diagnostics(
        self,
        query: str,
        context: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        filters: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        matched_terms = [str(t) for t in context.get("matchedTerms") or []]
        unmatched = _unmatched_terms(query, matched_terms)
        total = context.get("totalHits")

        notes: list[str] = []
        if unmatched:
            notes.append(
                "These query terms matched nothing in the collection: "
                + ", ".join(unmatched)
                + ". The collection uses different vocabulary for this concept."
            )
        if total == 0:
            notes.append(
                "No document matched. The filters may be the cause; try the "
                "same query without them."
                if filters
                else "No document matched. Drop the most specific terms, or "
                "use the vocabulary the index guide describes."
            )
        if isinstance(total, int) and total > 1000:
            notes.append(
                f"The query matches {total} documents; it is probably too broad "
                "to rank precisely."
            )
        suggestion = context.get("suggestions")
        if suggestion:
            notes.append(str(suggestion))

        return {
            "query_used": context.get("queryUsed"),
            "total_matches": total,
            "matched_terms": matched_terms,
            "query_terms_with_no_match": unmatched,
            "top_contributing_fields": list(context.get("topContributingFields") or []),
            "scoring_summary": context.get("scoringSummary"),
            "searchable_fields": list(context.get("searchableFields") or []),
            "filters_applied": list(filters) if filters else None,
            "top_result_explanation": rows[0].get("explanation") if rows else None,
            "notes": notes,
        }

    def _search_error(self, exc: RelevanApiError) -> str:
        """Turn an API error into something the agent can act on."""
        if exc.status == 400 and self.filter_fields:
            return (
                f"{exc.message} Filterable fields are: "
                f"{', '.join(sorted(set(self.filter_fields) | set(self.range_fields)))}."
            )
        if exc.status == 429:
            wait = exc.payload.get("retryAfterSeconds")
            return (
                "The search rate limit is exhausted"
                + (f"; it resets in {wait}s." if wait else ".")
            )
        return f"search failed: {exc.message}"

    def _build_filters(
        self, raw: Any
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        if raw in (None, [], {}):
            return None, None
        known = sorted(set(self.filter_fields) | set(self.range_fields))
        if not known:
            return None, "This collection has no filterable fields; search by text only."
        if isinstance(raw, Mapping):  # a lenient reading of {field: value}
            raw = [{"field": k, "value": v} for k, v in raw.items()]
        if not isinstance(raw, (list, tuple)):
            return None, "`filters` must be an array of {field, value} objects."

        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                return None, "each filter must be an object with a `field`."
            field = str(item.get("field") or "")
            if field not in known:
                return None, (
                    f"unknown filter field {field!r}. Filterable fields are: "
                    f"{', '.join(known)}."
                )
            clause: dict[str, Any] = {"field": field}
            if "values" in item and item["values"] not in (None, []):
                values = item["values"]
                if not isinstance(values, (list, tuple)):
                    return None, f"`values` for {field!r} must be an array."
                clause["values"] = list(values)
            elif "value" in item and item["value"] not in (None, ""):
                clause["value"] = item["value"]
            elif item.get("gte") is not None or item.get("lte") is not None:
                if field not in self.range_fields:
                    return None, (
                        f"{field!r} is not a range field; use `value` or "
                        f"`values`. Range fields are: "
                        f"{', '.join(self.range_fields) or 'none'}."
                    )
                for bound in ("gte", "lte"):
                    if item.get(bound) is not None:
                        clause[bound] = item[bound]
            else:
                return None, (
                    f"filter on {field!r} needs one of `value`, `values`, "
                    "`gte`, or `lte`."
                )
            out.append(clause)
        return out, None

    # -- ingest ----------------------------------------------------------

    def _ingest(self, corpus: Iterable[Document]) -> None:
        started = time.perf_counter()
        stream = iter(corpus)
        sample = list(itertools.islice(stream, self.sample_size))
        if not sample:
            raise RuntimeError(f"{self.name}: the corpus is empty, nothing to index")

        mapping = self._infer_mapping(sample)
        self._put_mapping(mapping)

        sent = 0
        dropped_fields: set[str] = set()
        for batch in _batched(itertools.chain(sample, stream), self.batch_size):
            documents = [
                {
                    "id": _encode_id(doc.doc_id),
                    "content": self._content(doc, dropped_fields),
                }
                for doc in batch
            ]
            self._post(
                self._index_path("/documents/_bulk"), {"documents": documents}
            )
            sent += len(documents)
            if sent % 10000 < self.batch_size:
                self._log(f"sent {sent} documents")
        if dropped_fields:
            self._log(
                "these fields were not in the mapping sample and were dropped: "
                + ", ".join(sorted(dropped_fields))
                + " (raise sample_size to include them)"
            )
        self._log(
            f"sent {sent} documents in {time.perf_counter() - started:.1f}s; "
            "waiting for the index to catch up"
        )

    def _infer_mapping(self, sample: Sequence[Document]) -> dict[str, Any]:
        """Derive a Relevan mapping from a sample of the corpus.

        Nothing here is dataset-specific: fields come from ``Document.fields``,
        and a string field earns ``filterable`` only if it looks like a label
        rather than prose.
        """
        fields: dict[str, dict[str, Any]] = {
            ID_FIELD: {"kind": "id", "use": ["filterable"]},
            "title": {
                "kind": "text",
                "use": ["searchable", "highlightable", "autocompletable"],
            },
        }
        if any(doc.text for doc in sample):
            fields["text"] = {"kind": "text", "use": ["searchable", "highlightable"]}

        kinds: dict[str, str] = {}
        values: dict[str, set[str]] = {}
        longest: dict[str, int] = {}
        for doc in sample:
            for key, raw in doc.fields.items():
                if key in fields:
                    continue
                value = _scalarize(raw)
                if value in (None, "") or value == doc.text:
                    continue
                kind = _kind_of(value)
                if kinds.setdefault(key, kind) != kind:
                    kinds[key] = "text"  # mixed types: index it as text
                if isinstance(value, str):
                    seen = values.setdefault(key, set())
                    if len(seen) <= MAX_FILTER_CARDINALITY:
                        seen.add(value)
                    longest[key] = max(longest.get(key, 0), len(value))

        for key, kind in sorted(kinds.items()):
            if kind == "boolean":
                fields[key] = {"kind": "boolean", "use": ["filterable"]}
            elif kind == "number":
                fields[key] = {"kind": "number", "use": ["filterable", "sortable"]}
            elif (
                len(values.get(key, ())) <= MAX_FILTER_CARDINALITY
                and longest.get(key, 0) <= MAX_FILTER_VALUE_CHARS
            ):
                fields[key] = {
                    "kind": "text",
                    "use": ["searchable", "filterable", "aggregatable"],
                }
            else:
                fields[key] = {
                    "kind": "text",
                    "use": ["searchable", "highlightable"],
                }
        return {"fields": fields, "language": self.language}

    def _content(self, doc: Document, dropped: set[str]) -> dict[str, Any]:
        content: dict[str, Any] = {ID_FIELD: doc.doc_id}
        if doc.title:
            content["title"] = doc.title
        if doc.text and "text" in self._declared:
            content["text"] = doc.text
        for key, raw in doc.fields.items():
            if key in (ID_FIELD, "title", "text"):
                continue
            value = _scalarize(raw)
            # A field that just repeats the body would be indexed twice and
            # double-count every term it contains.
            if value in (None, "") or value == doc.text:
                continue
            if key not in self._declared:
                dropped.add(key)
                continue
            value = _coerce(value, self._field_kinds.get(key, "text"))
            if value is None:
                continue
            content[key] = value
        return content

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.poll_timeout
        last_indexed = -1
        while True:
            detail = self._get_index() or {}
            indexed = detail.get("documentsIndexed")
            accepted = detail.get("documentsAccepted")
            if detail.get("status") == "failed":
                raise RuntimeError(
                    f"{self.name}: index {self.index_name!r} reports status 'failed'"
                )
            if detail.get("readyToSearch"):
                self._documents_indexed = indexed
                self._log(f"index ready: {indexed} documents searchable")
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"{self.name}: index {self.index_name!r} still indexing after "
                    f"{self.poll_timeout:.0f}s ({indexed} of {accepted} documents). "
                    "Raise poll_timeout, or rerun once ingest has caught up."
                )
            if indexed != last_indexed:
                self._log(f"indexed {indexed} of {accepted} documents")
                last_indexed = indexed
            time.sleep(self.poll_interval)

    # -- index and mapping plumbing --------------------------------------

    def _resolve_org(self) -> None:
        if self.org:
            return
        orgs = self._get("/v1/organizations")
        if isinstance(orgs, Mapping):
            orgs = orgs.get("organizations") or []
        if not orgs:
            raise RuntimeError(
                f"{self.name}: the API key resolves to no organization"
            )
        if len(orgs) > 1:
            names = ", ".join(str(o.get("slug") or o.get("id")) for o in orgs)
            raise RuntimeError(
                f"{self.name}: the key resolves to several organizations "
                f"({names}); pass --adapter-opt org=<slug>"
            )
        self.org = orgs[0].get("slug") or orgs[0].get("id")

    def _get_index(self) -> dict[str, Any] | None:
        try:
            return self._get(self._index_path(""))
        except RelevanApiError as exc:
            if exc.status == 404:
                return None
            raise

    def _create_index(self, dataset_name: str) -> None:
        self._post(
            f"/v1/{self._org_segment()}/indexes",
            {
                "name": self.index_name,
                "displayName": f"shots-to-success: {dataset_name}",
                "description": (
                    f"Benchmark corpus for the {dataset_name} dataset, ingested "
                    "by the shots-to-success relevan adapter."
                ),
            },
        )
        self._log(f"created index {self.index_name!r}")

    def _put_mapping(self, mapping: Mapping[str, Any]) -> None:
        revision = self._post(self._index_path("/mappings"), dict(mapping))
        revision = dict(revision) if isinstance(revision, Mapping) else {}
        # Fall back to what we sent: ingest depends on knowing which fields are
        # declared, and dropping them all would silently index empty documents.
        revision.setdefault("relevanMapping", dict(mapping))
        self._adopt_mapping(revision)
        self._log(
            f"deployed mapping revision {self._mapping_revision} "
            f"({len(self._declared)} fields)"
        )

    def _load_mapping(self) -> None:
        """Read the index's current mapping, for its filterable fields.

        Also the path taken when ingest was skipped: the adapter then knows
        nothing about the corpus except what the index reports.
        """
        payload = self._get(self._index_path("/mappings"))
        revisions = payload.get("mappings") if isinstance(payload, Mapping) else payload
        if not revisions:
            raise RuntimeError(
                f"{self.name}: index {self.index_name!r} has no mapping; "
                "ingest a corpus before searching it"
            )
        latest = max(revisions, key=lambda r: int(r.get("revision") or 0))
        self._adopt_mapping(latest)

    def _adopt_mapping(self, revision: Mapping[str, Any]) -> None:
        mapping = revision.get("relevanMapping") or {}
        fields = dict(mapping.get("fields") or {})
        self._mapping_revision = revision.get("revision")
        self._mapping_hash = revision.get("relevanMappingHash")
        self._declared = set(fields)
        self._field_kinds = {
            name: str(spec.get("kind") or "text") for name, spec in fields.items()
        }
        filterable = {
            name
            for name, spec in fields.items()
            if "filterable" in (spec.get("use") or [])
        }
        self._id_filterable = ID_FIELD in filterable
        # ID_FIELD is the harness's own plumbing. It stays out of every list
        # the agent sees: filtering by an internal id is not a search skill.
        self.searchable_fields = sorted(
            name
            for name, spec in fields.items()
            if "searchable" in (spec.get("use") or []) and name != ID_FIELD
        )
        self.filter_fields = sorted(
            name
            for name in filterable
            if name != ID_FIELD and self._field_kinds.get(name) not in ("number", "date")
        )
        self.range_fields = sorted(
            name
            for name in filterable
            if name != ID_FIELD and self._field_kinds.get(name) in ("number", "date")
        )

    def _skill(self) -> str:
        if self._skill_doc is None:
            self._skill_doc = str(self._request("GET", self._index_path("/skill")))
        return self._skill_doc

    def _index_path(self, suffix: str, *, version: str = "v1") -> str:
        return (
            f"/{version}/{self._org_segment()}/indexes/"
            f"{urllib.parse.quote(str(self.index_name))}{suffix}"
        )

    def _org_segment(self) -> str:
        return urllib.parse.quote(str(self.org))

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("RelevanAdapter.build() must be called before searching")

    def _log(self, message: str) -> None:
        print(f"[{self.name}] {message}", file=sys.stderr)

    # -- transport -------------------------------------------------------

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, body: Mapping[str, Any]) -> Any:
        return self._request("POST", path, body)

    def _request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> Any:
        if not self.api_key:
            raise RuntimeError(
                f"{self.name}: no API key. Set RELEVAN_API_KEY or pass "
                "--adapter-opt api_key=..."
            )
        url = f"{self.base_url}{path}"
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"x-api-key": self.api_key, "accept": "application/json"}
        if payload is not None:
            headers["content-type"] = "application/json"

        for attempt in range(self.max_retries + 1):
            status, response_headers, raw = self._send(method, url, payload, headers)
            if status == 429 and attempt < self.max_retries:
                time.sleep(_retry_after(response_headers, raw))
                continue
            if 200 <= status < 300:
                return _decode_body(raw, response_headers)
            decoded = _decode_body(raw, response_headers)
            # A 401 answers with `error` and no `message`, so read both.
            message = (
                decoded.get("message") or decoded.get("error")
                if isinstance(decoded, Mapping)
                else str(decoded)[:400]
            )
            message = str(message or f"HTTP {status}")
            if status == 401:
                message += (
                    " -- the API key was rejected. Check RELEVAN_API_KEY, or "
                    "that the key belongs to this organization."
                )
            raise RelevanApiError(status, message, decoded)
        raise RelevanApiError(429, "rate limit exhausted after retries")

    def _send(
        self,
        method: str,
        url: str,
        payload: bytes | None,
        headers: Mapping[str, str],
    ) -> tuple[int, Mapping[str, str], str]:
        """One HTTP round trip. Overridden in tests; the only I/O in the class."""
        request = urllib.request.Request(url, data=payload, method=method)
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return (
                    response.status,
                    dict(response.headers),
                    response.read().decode("utf-8"),
                )
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers or {}), exc.read().decode("utf-8")

    # -- provenance ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["base_url"] = self.base_url
        info["org"] = self.org
        info["index"] = self.index_name
        info["mapping_revision"] = self._mapping_revision
        info["mapping_hash"] = self._mapping_hash
        info["documents_indexed"] = self._documents_indexed
        info["searchable_fields"] = self.searchable_fields
        info["filter_fields"] = self.filter_fields
        info["range_fields"] = self.range_fields
        info["send_feedback"] = self.send_feedback
        # Everything that could change ranking belongs in the manifest -- but
        # never the credential.
        info["options"] = {k: v for k, v in self.options.items() if k != "api_key"}
        return info


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _default_index_name(dataset_name: str, cache_key: str) -> str:
    """``sts-<dataset>-<cache key>``: one index per dataset configuration."""
    slug = re.sub(r"[^a-z0-9]+", "-", dataset_name.lower()).strip("-") or "corpus"
    suffix = re.sub(r"[^a-z0-9]+", "", cache_key.lower())[:12]
    name = f"sts-{slug[:40]}" + (f"-{suffix}" if suffix else "")
    return name.strip("-")[:64]


def _encode_id(doc_id: str) -> str:
    """Make a dataset id acceptable to the API, reversibly.

    Ids already inside the allowed character set pass through unchanged, so
    the common case stays readable in the console. Anything else -- and any id
    that would be mistaken for an encoded one -- is base32-encoded, whose
    alphabet is a subset of what the API allows.
    """
    if SAFE_ID.match(doc_id) and not doc_id.startswith(ENCODED_PREFIX):
        encoded = doc_id
    else:
        encoded = ENCODED_PREFIX + base64.b32encode(
            doc_id.encode("utf-8")
        ).decode("ascii").rstrip("=")
    if len(encoded) > ID_MAX:
        raise ValueError(
            f"document id {doc_id!r} is too long for the API "
            f"({len(encoded)} > {ID_MAX} characters once encoded)"
        )
    return encoded


def _decode_id(remote_id: str) -> str:
    if not remote_id.startswith(ENCODED_PREFIX):
        return remote_id
    payload = remote_id[len(ENCODED_PREFIX) :]
    padding = "=" * (-len(payload) % 8)
    try:
        return base64.b32decode(payload + padding).decode("utf-8")
    except Exception:  # noqa: BLE001 - an id we did not write; hand it back as-is
        return remote_id


def _batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _scalarize(value: Any) -> Any:
    """Flatten a dataset field into something a mapping can declare."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value if isinstance(v, (str, int, float, bool))]
        return ", ".join(parts) if parts else None
    return None


def _kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


def _coerce(value: Any, kind: str) -> Any:
    """Make a value match the kind its field was mapped with."""
    if kind == "text":
        return value if isinstance(value, str) else str(value)
    if kind == "number":
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if kind == "boolean":
        return value if isinstance(value, bool) else None
    return value


def _build_boost(raw: Any) -> tuple[dict[str, float] | None, str | None]:
    if raw in (None, {}):
        return None, None
    if not isinstance(raw, Mapping):
        return None, "`boost` must be an object of term -> weight."
    out: dict[str, float] = {}
    for term, weight in raw.items():
        try:
            out[str(term)] = float(weight)
        except (TypeError, ValueError):
            return None, f"boost for {term!r} must be a number."
    return out, None


def _display_fields(hit: Hit) -> dict[str, Any]:
    """Content fields shown to the agent: not the explanation, which is feedback."""
    return {
        key: value
        for key, value in hit.extra.items()
        if key not in ("explanation", "matched_fields", "matched_terms")
    }


def _snippet(highlights: Mapping[str, Any], content: Mapping[str, Any]) -> str:
    for field in ("text", "description", "title"):
        fragments = highlights.get(field)
        if fragments:
            return _strip_tags(str(fragments[0]))[:SNIPPET_CHARS].strip()
    for fragments in highlights.values():
        if fragments:
            return _strip_tags(str(fragments[0]))[:SNIPPET_CHARS].strip()
    for field in ("text", "description"):
        value = content.get(field)
        if isinstance(value, str) and value:
            return value[:SNIPPET_CHARS].strip()
    return ""


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _truncate(value: Any, limit: int = 160) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit].rstrip() + "..."
    return value


def _unmatched_terms(query: str, matched: Sequence[str]) -> list[str]:
    """Query words that nothing in the index matched.

    ``matched`` holds *indexed* terms, which are analyzed and may be stemmed,
    so a word is treated as matched when any indexed term shares its opening
    characters. The comparison errs towards silence: a term is only reported
    as unmatched when nothing resembling it contributed.
    """
    lowered = [t.lower() for t in matched]
    unmatched: list[str] = []
    for word in dict.fromkeys(re.findall(r"[a-z0-9]+", query.lower())):
        if len(word) < 3 or word in STOPWORDS:
            continue
        stub = word[:4]
        if any(term.startswith(stub) or word.startswith(term[:4]) for term in lowered):
            continue
        unmatched.append(word)
    return unmatched


def _retry_after(headers: Mapping[str, str], raw: str) -> float:
    """How long a 429 asks us to wait, from the header or the body."""
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return min(float(value), MAX_RETRY_SLEEP)
            except (TypeError, ValueError):
                break
    try:
        payload = json.loads(raw)
        return min(float(payload.get("retryAfterSeconds", 1.0)), MAX_RETRY_SLEEP)
    except Exception:  # noqa: BLE001
        return 1.0


def _decode_body(raw: str, headers: Mapping[str, str]) -> Any:
    content_type = ""
    for key, value in headers.items():
        if key.lower() == "content-type":
            content_type = value
            break
    if "json" in content_type or (raw[:1] in ("{", "[")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


__all__ = ["RelevanAdapter", "RelevanApiError"]
