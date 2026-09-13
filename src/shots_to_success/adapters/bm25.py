"""Reference search adapter: in-process BM25 with a full agent tool surface.

This is the benchmark's baseline backend. It runs with no infrastructure, is
deterministic, and exposes the four things an agent-facing search system is
argued to need: retrieval, query assistance (autocomplete), collection
introspection (facets, schema), and diagnostics (why these results, what the
filter did, which terms matched nothing).

It is dataset-agnostic: filterable and facetable fields are discovered from
``Document.fields`` while indexing, so the same adapter works on any dataset
implementing the :class:`~shots_to_success.datasets.base.Dataset` interface.
"""

from __future__ import annotations

import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from ..types import Document, Hit, ResultSet, ToolKind, ToolResponse, ToolSpec
from .base import SearchAdapter
from .bm25_index import DEFAULT_FIELD_BOOSTS, Bm25Index, tokenize

#: A string field becomes filterable/facetable below this many distinct values.
MAX_FILTER_CARDINALITY = 5000
SNIPPET_CHARS = 220
#: Fields kept for display and filtering but never added to the text index --
#: they duplicate `text` and would double-count every term.
DISPLAY_ONLY_FIELDS = frozenset({"description", "features"})


class Bm25Adapter(SearchAdapter):
    """BM25 over a locally built index."""

    name: ClassVar[str] = "bm25"
    version: ClassVar[str] = "2"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.k1 = float(options.get("k1", 1.2))
        self.b = float(options.get("b", 0.75))
        self.field_boosts = dict(options.get("field_boosts") or DEFAULT_FIELD_BOOSTS)
        self.cache_dir = Path(options.get("cache_dir", "data/cache"))
        self.use_cache = bool(options.get("use_cache", True))
        self.max_query_terms = int(options.get("max_query_terms", 50))
        self.index = Bm25Index(
            k1=self.k1,
            b=self.b,
            field_boosts=self.field_boosts,
            max_query_terms=self.max_query_terms,
        )
        self.filter_fields: list[str] = []
        self.range_fields: list[str] = []
        self._built = False

    # -- build -----------------------------------------------------------

    def build(
        self,
        corpus: Iterable[Document],
        *,
        dataset_name: str,
        cache_key: str = "",
    ) -> None:
        cache_path = self.cache_dir / f"{dataset_name}-{self.name}-{cache_key}.pkl"
        if self.use_cache and cache_key and cache_path.exists():
            with cache_path.open("rb") as handle:
                state = pickle.load(handle)
            self.index = state["index"]
            self.filter_fields = state["filter_fields"]
            self.range_fields = state["range_fields"]
            self._built = True
            return

        started = time.perf_counter()
        self.index.start()
        string_values: dict[str, set[str]] = {}
        numeric_fields: set[str] = set()
        count = 0
        for doc in corpus:
            # Index the title, the body, and every other string field the
            # dataset supplies. Nothing here is dataset-specific.
            fields = {"title": doc.title, "text": doc.text}
            for key, value in doc.fields.items():
                if key in DISPLAY_ONLY_FIELDS or key in fields:
                    continue
                if isinstance(value, str) and value:
                    fields[key] = value
            meta = {
                key: value
                for key, value in doc.fields.items()
                if key not in DISPLAY_ONLY_FIELDS
            }
            self.index.add(
                doc_id=doc.doc_id,
                title=doc.title,
                snippet=_snippet_source(doc),
                fields=fields,
                meta=meta,
            )
            for key, value in meta.items():
                if isinstance(value, str) and value:
                    seen = string_values.setdefault(key, set())
                    if len(seen) <= MAX_FILTER_CARDINALITY:
                        seen.add(value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_fields.add(key)
            count += 1
            if count % 10000 == 0:
                print(f"[bm25] indexed {count} docs", file=sys.stderr)
        self.index.finalize()
        self.filter_fields = sorted(
            key
            for key, values in string_values.items()
            if len(values) <= MAX_FILTER_CARDINALITY
        )
        self.range_fields = sorted(numeric_fields)
        self._built = True
        print(
            f"[bm25] indexed {count} docs in {time.perf_counter() - started:.1f}s "
            f"({len(self.index.postings)} terms)",
            file=sys.stderr,
        )
        if self.use_cache and cache_key:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".part")
            with tmp.open("wb") as handle:
                pickle.dump(
                    {
                        "index": self.index,
                        "filter_fields": self.filter_fields,
                        "range_fields": self.range_fields,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp.replace(cache_path)

    # -- tool contract ---------------------------------------------------

    def tools(self) -> Sequence[ToolSpec]:
        # A corpus with no structured fields -- most BEIR datasets -- gets no
        # filter argument and no facet tool. Advertising a capability the
        # backend does not have wastes the agent's turns on errors.
        filterable = bool(self.filter_fields or self.range_fields)
        search_properties: dict[str, Any] = {
            "query": {
                "type": "string",
                "description": "Free-text keywords to search for.",
            },
            "k": {
                "type": "integer",
                "description": "How many results to return (max 50).",
            },
        }
        if filterable:
            search_properties["filters"] = self._filter_schema()
        specs = [
            ToolSpec(
                name="search",
                description=(
                    "Full-text search over the collection. Returns a ranked list "
                    "of results with a title and snippet for each."
                    + (
                        " Optional filters narrow the candidate set before "
                        "ranking."
                        if filterable
                        else ""
                    )
                ),
                input_schema={
                    "type": "object",
                    "properties": search_properties,
                    "required": ["query"],
                },
                kind=ToolKind.RETRIEVAL,
                handler=self._search,
            ),
            ToolSpec(
                name="autocomplete",
                description=(
                    "Query suggestions for a prefix, drawn from the collection's "
                    "titles and categories with the number of items behind each. "
                    "Use it to discover the vocabulary the collection actually "
                    "uses before spending a search."
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
            ToolSpec(
                name="describe_index",
                description=(
                    "The index schema: searchable fields and their weights, "
                    "filterable fields, collection size, and the analyzer."
                ),
                input_schema={"type": "object", "properties": {}},
                kind=ToolKind.AUXILIARY,
                handler=self._describe_index,
            ),
            ToolSpec(
                name="explain_result",
                description=(
                    "Why a specific document scores the way it does for a query: "
                    "per-term score contribution and which terms it is missing."
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
            ),
        ]
        if self.filter_fields:
            specs.insert(
                2,
                ToolSpec(
                    name="facets",
                    description=(
                        "Value counts for a field across the documents matching "
                        "a query. Shows which categories a query is pulling from "
                        "and which filter values exist."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "field": {"type": "string", "enum": self.filter_fields},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query", "field"],
                    },
                    kind=ToolKind.AUXILIARY,
                    handler=self._facets,
                ),
            )
        return specs

    def _filter_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for name in self.filter_fields:
            properties[name] = {
                "type": "string",
                "description": f"Keep only documents whose {name} matches exactly.",
            }
            properties[f"{name}_contains"] = {
                "type": "string",
                "description": f"Keep only documents whose {name} contains this text.",
            }
        for name in self.range_fields:
            properties[f"min_{name}"] = {
                "type": "number",
                "description": f"Keep only documents with {name} at or above this.",
            }
        return {
            "type": "object",
            "description": "Optional filters applied before ranking.",
            "properties": properties,
        }

    # -- handlers --------------------------------------------------------

    def _search(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResponse(error="`query` must be a non-empty string.")
        k = max(1, min(int(args.get("k") or 10), 50))
        raw_filters = args.get("filters") or {}
        if not isinstance(raw_filters, Mapping):
            return ToolResponse(error="`filters` must be an object.")
        if raw_filters and not (self.filter_fields or self.range_fields):
            return ToolResponse(
                error="This collection has no filterable fields; search by text only."
            )

        terms = tokenize(query)
        if not terms:
            return ToolResponse(error="`query` contained no searchable terms.")

        predicate, filter_report, bad_filter = self._build_predicate(raw_filters)
        if bad_filter:
            return ToolResponse(error=bad_filter)

        top, before, after = self.index.search(terms, k, predicate, explain=True)
        hits = [self._to_hit(scored, terms) for scored in top]
        result_set = ResultSet(hits=hits, query=query, total=after)

        body = {
            "query": query,
            "returned": len(hits),
            "total_matches": after,
            "results": [
                {
                    "rank": i + 1,
                    "doc_id": hit.doc_id,
                    "title": hit.title,
                    "snippet": hit.snippet,
                    **{
                        key: value
                        for key, value in hit.extra.items()
                        if key != "score_contributors"
                    },
                }
                for i, hit in enumerate(hits)
            ],
        }
        diagnostics = self._diagnostics(
            terms, top, before, after, filter_report, raw_filters
        )
        return ToolResponse(body=body, result_set=result_set, diagnostics=diagnostics)

    def _autocomplete(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        prefix = str(args.get("prefix") or "")
        limit = max(1, min(int(args.get("limit") or 10), 25))
        suggestions = self.index.suggest(prefix, limit)
        return ToolResponse(
            body={
                "prefix": prefix,
                "suggestions": [
                    {"text": text, "documents": count} for text, count in suggestions
                ]
                or "no suggestions for this prefix",
            }
        )

    def _facets(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        field = str(args.get("field") or "")
        if field not in self.filter_fields:
            return ToolResponse(
                error=f"`field` must be one of: {', '.join(self.filter_fields)}"
            )
        query = str(args.get("query") or "").strip()
        limit = max(1, min(int(args.get("limit") or 10), 50))
        terms = tokenize(query)
        scored = self.index.score(terms)
        counter: Counter[str] = Counter()
        for doc_index in scored:
            value = self.index.meta[doc_index].get(field)
            if isinstance(value, str) and value:
                counter[value] += 1
        return ToolResponse(
            body={
                "query": query,
                "field": field,
                "matching_documents": len(scored),
                "values": [
                    {"value": value, "documents": count}
                    for value, count in counter.most_common(limit)
                ],
            }
        )

    def _describe_index(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        info = self.index.describe()
        info["searchable_fields"] = {
            name: f"weight {self.index.boost(name)}"
            for name in self.index._field_order
        }
        info["filterable_fields"] = self.filter_fields
        info["numeric_range_fields"] = self.range_fields
        return ToolResponse(body=info)

    def _explain(self, args: Mapping[str, Any]) -> ToolResponse:
        self._require_built()
        query = str(args.get("query") or "")
        doc_id = str(args.get("doc_id") or "")
        doc_index = self.index.doc_index.get(doc_id)
        if doc_index is None:
            return ToolResponse(error=f"unknown doc_id {doc_id!r}")
        terms = tokenize(query)
        contributions = self.index.explain_doc(terms, doc_index)
        return ToolResponse(
            body={
                "doc_id": doc_id,
                "title": self.index.titles[doc_index],
                "total_score": round(sum(contributions.values()), 4),
                "term_contributions": {
                    term: round(value, 4) for term, value in contributions.items()
                },
                "terms_not_in_document": [
                    term for term, value in contributions.items() if value == 0.0
                ],
            }
        )

    # -- helpers ---------------------------------------------------------

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Bm25Adapter.build() must be called before searching")

    def _build_predicate(self, filters: Mapping[str, Any]):
        """Compile filters into a predicate plus a per-clause removal report."""
        clauses: list[tuple[str, Any]] = []
        for key, value in filters.items():
            if value in (None, ""):
                continue
            if key in self.filter_fields:
                clauses.append(("eq:" + key, str(value).strip().lower()))
            elif key.endswith("_contains") and key[: -len("_contains")] in self.filter_fields:
                clauses.append(("contains:" + key[: -len("_contains")], str(value).strip().lower()))
            elif key.startswith("min_") and key[len("min_") :] in self.range_fields:
                try:
                    clauses.append(("min:" + key[len("min_") :], float(value)))
                except (TypeError, ValueError):
                    return None, {}, f"filter {key!r} must be a number"
            else:
                known = (
                    self.filter_fields
                    + [f"{f}_contains" for f in self.filter_fields]
                    + [f"min_{f}" for f in self.range_fields]
                )
                return None, {}, (
                    f"unknown filter {key!r}. Available filters: {', '.join(known)}"
                )
        if not clauses:
            return None, {}, None

        report: Counter[str] = Counter()

        def predicate(doc_index: int) -> bool:
            meta = self.index.meta[doc_index]
            passed = True
            for clause, expected in clauses:
                kind, _, field = clause.partition(":")
                value = meta.get(field)
                if kind == "eq":
                    ok = isinstance(value, str) and value.strip().lower() == expected
                elif kind == "contains":
                    ok = isinstance(value, str) and expected in value.lower()
                else:
                    ok = isinstance(value, (int, float)) and float(value) >= expected
                if not ok:
                    report[clause] += 1
                    passed = False
            return passed

        return predicate, report, None

    def _to_hit(self, scored, terms: Sequence[str]) -> Hit:
        index = scored.index
        meta = self.index.meta[index]
        extra = {
            key: value
            for key, value in meta.items()
            if isinstance(value, (str, int, float)) and value not in ("", None)
        }
        extra["score"] = round(scored.score, 4)
        extra["score_contributors"] = {
            term: round(value, 4)
            for term, value in sorted(
                scored.contributions.items(), key=lambda kv: -kv[1]
            )
        }
        return Hit(
            doc_id=self.index.doc_ids[index],
            score=round(scored.score, 4),
            title=self.index.titles[index],
            snippet=_make_snippet(self.index.snippets[index], terms),
            extra=extra,
        )

    def _diagnostics(
        self,
        terms: Sequence[str],
        top,
        before: int,
        after: int,
        filter_report: Mapping[str, int],
        raw_filters: Mapping[str, Any],
    ) -> dict[str, Any]:
        stats = self.index.term_stats(list(dict.fromkeys(terms)))
        matched = [s for s in stats if s.df > 0]
        missing = [s for s in stats if s.df == 0]
        all_terms = self.index.docs_matching_all(terms)
        total_docs = len(self.index)

        notes: list[str] = []
        if missing:
            notes.append(
                "These query terms appear in no document in the collection: "
                + ", ".join(sorted(s.term for s in missing))
                + ". The collection uses different vocabulary for this concept."
            )
        if all_terms == 0 and len(set(terms)) > 1:
            notes.append(
                "No single document contains all query terms; results match only "
                "a subset. Consider dropping the least essential term."
            )
        broad = [s for s in matched if s.df > total_docs * 0.2]
        if broad:
            notes.append(
                "These terms are too common to discriminate (present in over 20% "
                "of the collection): " + ", ".join(s.term for s in broad) + "."
            )
        if raw_filters and after == 0 and before > 0:
            notes.append(
                f"Filters removed all {before} text matches. Loosen or drop them."
            )
        elif raw_filters and before > after:
            notes.append(
                f"Filters removed {before - after} of {before} text matches."
            )
        if after > total_docs * 0.1:
            notes.append(
                f"The query matches {after} documents, a large share of the "
                "collection; it is probably too broad to rank precisely."
            )

        result_classes: Counter[str] = Counter()
        facet_field = self.filter_fields[0] if self.filter_fields else None
        if facet_field:
            for scored in top:
                value = self.index.meta[scored.index].get(facet_field)
                if isinstance(value, str) and value:
                    result_classes[value] += 1

        return {
            "collection_size": total_docs,
            "matched_terms": [
                {
                    "term": s.term,
                    "documents": s.df,
                    "idf": round(s.idf, 3),
                    "fields": self.index.fields_for_term(s.term),
                }
                for s in matched
            ],
            "missing_terms": [s.term for s in missing],
            "documents_matching_all_terms": all_terms,
            "documents_matching_any_term": before,
            "documents_after_filters": after,
            "filters_applied": dict(raw_filters) or None,
            "documents_removed_by_filter": (
                {clause: count for clause, count in filter_report.items()} or None
            ),
            "top_result_score_contributors": (
                {
                    term: round(value, 4)
                    for term, value in sorted(
                        top[0].contributions.items(), key=lambda kv: -kv[1]
                    )
                }
                if top
                else None
            ),
            "results_by_" + (facet_field or "category"): [
                {"value": value, "results": count}
                for value, count in result_classes.most_common(5)
            ],
            "notes": notes,
        }

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["index"] = self.index.describe() if self._built else None
        info["filter_fields"] = self.filter_fields
        info["range_fields"] = self.range_fields
        return info


def _snippet_source(doc: Document) -> str:
    description = str(doc.fields.get("description") or "")
    features = str(doc.fields.get("features") or "")
    return (description or doc.text or features)[:2000]


def _make_snippet(source: str, terms: Sequence[str]) -> str:
    """A window around the first matching term, like a real result snippet."""
    if not source:
        return ""
    lowered = source.lower()
    position = -1
    for term in terms:
        position = lowered.find(term)
        if position >= 0:
            break
    if position < 0:
        return source[:SNIPPET_CHARS].strip()
    start = max(0, position - SNIPPET_CHARS // 3)
    text = source[start : start + SNIPPET_CHARS].strip()
    return ("..." if start > 0 else "") + text + ("..." if start + SNIPPET_CHARS < len(source) else "")


__all__ = ["Bm25Adapter"]
