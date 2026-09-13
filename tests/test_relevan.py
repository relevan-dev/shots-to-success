"""The Relevan adapter, against a scripted stand-in for api.relevan.dev.

Every test here runs the real adapter code -- ingest, mapping inference, id
encoding, result normalization -- with only the single HTTP round trip
replaced. No network and no API key.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

import pytest

from shots_to_success.adapters.relevan import (
    ID_FIELD,
    RelevanAdapter,
    _decode_id,
    _encode_id,
    _unmatched_terms,
)
from shots_to_success.types import Document, ToolKind


STAMP = "2026-01-15T10:30:00.000Z"


class FakeApi:
    """A minimal Relevan: it stores documents and matches on shared words.

    Its responses are shaped by the published OpenAPI document, so a test that
    passes here is testing against the API that exists.
    """

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.mappings: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self.searches: list[dict[str, Any]] = []
        self.ready = True
        self.queue: list[tuple[int, dict[str, str], str]] = []

    def handle(
        self, method: str, url: str, payload: bytes | None, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, str], str]:
        path = url.split("api.relevan.dev", 1)[-1]
        self.calls.append((method, path))
        if self.queue:
            return self.queue.pop(0)
        body = json.loads(payload) if payload else None
        status, response = self._route(method, path, body)
        content_type = "text/markdown" if isinstance(response, str) else "application/json"
        raw = response if isinstance(response, str) else json.dumps(response)
        return status, {"content-type": content_type}, raw

    def _route(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        if path == "/v1/organizations":
            return 200, [{"id": "org-1", "slug": "acme", "name": "Acme"}]
        if path == "/v1/acme/indexes" and method == "POST":
            self.indexes[body["name"]] = {"indexName": body["name"]}
            return 201, {
                "id": "ix-1",
                "organizationId": "org-1",
                "indexName": body["name"],
                "displayName": body.get("displayName"),
                "description": body.get("description"),
                "status": "ready",
                "createdAt": STAMP,
                "updatedAt": STAMP,
            }

        match = re.match(r"^/(v1|v1-beta)/acme/indexes/([^/]+)(.*)$", path)
        if not match:
            return 404, {"message": "no route"}
        _, index, suffix = match.groups()
        if index not in self.indexes:
            return 404, {"message": "Index not found"}

        if suffix == "" and method == "GET":
            return 200, {
                "id": "ix-1",
                "organizationId": "org-1",
                "indexName": index,
                "displayName": None,
                "description": None,
                "status": "ready",
                "createdAt": STAMP,
                "updatedAt": STAMP,
                "documentsAccepted": len(self.documents),
                "documentsIndexed": len(self.documents) if self.ready else 0,
                "documentsAcceptedChangedAt": STAMP if self.documents else None,
                "readyToSearch": self.ready and bool(self.documents),
            }
        if suffix == "/mappings" and method == "POST":
            self.mappings.append(
                {
                    "id": f"mapping-{len(self.mappings) + 1}",
                    "revision": len(self.mappings) + 1,
                    "relevanMapping": body,
                    "relevanMappingYaml": "fields: {}\n",
                    "relevanMappingHash": "hash-1",
                    "createdBy": None,
                    "createdAt": STAMP,
                }
            )
            return 201, self.mappings[-1]
        if suffix == "/mappings" and method == "GET":
            return 200, self.mappings  # MappingsList is a bare array
        if suffix == "/documents/_bulk":
            for item in body["documents"]:
                self.documents[item["id"]] = item["content"]
            return 202, {
                "accepted": len(body["documents"]),
                "documentIds": [d["id"] for d in body["documents"]],
            }
        if suffix == "/skill":
            return 200, "# products\n\nSearchable fields: title, text.\n"
        if suffix == "/feedback":
            self.feedback.append(body)
            return 202, {"message": "accepted"}
        if suffix == "/search":
            return self._search(body)
        return 404, {"message": "no route"}

    def _search(self, body: Mapping[str, Any]) -> tuple[int, Any]:
        self.searches.append(dict(body))
        terms = re.findall(r"[a-z0-9]+", str(body["query"]).lower())
        results = []
        for doc_id, content in sorted(self.documents.items()):
            if not _passes(content, body.get("filters") or []):
                continue
            haystack = " ".join(
                str(v) for v in content.values() if isinstance(v, str)
            ).lower()
            matched = [t for t in terms if t in haystack]
            if not matched:
                continue
            results.append(
                {
                    "id": doc_id,
                    "score": float(len(matched)),
                    "document": content,
                    "highlights": {"text": [f"<em>{matched[0]}</em> in context"]},
                    "explanation": f"matched {', '.join(matched)}",
                    "matchedFields": ["title"],
                    "matchedTerms": matched,
                }
            )
        results.sort(key=lambda r: (-r["score"], r["id"]))
        results = results[: int(body.get("size") or 10)]
        matched_terms = sorted({t for r in results for t in r["matchedTerms"]})
        return 200, {
            "took": 3,
            "results": results,
            "searchContext": {
                "queryUsed": body["query"],
                "totalHits": len(results),
                "matchedTerms": matched_terms,
                "topContributingFields": ["title"],
                "scoringSummary": "bm25",
                "suggestions": "try fewer terms",
                "searchableFields": ["title", "text"],
                "sessionId": "session-1",
                "queryId": "query-1",
            },
        }


def _passes(content: Mapping[str, Any], filters: Any) -> bool:
    for clause in filters:
        value = content.get(clause["field"])
        if "value" in clause and value != clause["value"]:
            return False
        if "values" in clause and value not in clause["values"]:
            return False
        if clause.get("gte") is not None and not (
            isinstance(value, (int, float)) and value >= clause["gte"]
        ):
            return False
    return True


class FakeRelevanAdapter(RelevanAdapter):
    """The adapter with its one I/O method pointed at the fake."""

    def __init__(self, api: FakeApi, **options: Any) -> None:
        options.setdefault("api_key", "test-key")
        super().__init__(**options)
        self.api = api

    def _send(self, method, url, payload, headers):  # type: ignore[override]
        return self.api.handle(method, url, payload, headers)


CORPUS = [
    Document(
        doc_id="1",
        title="red velvet accent chair",
        text="a plush red chair for the study",
        fields={"product_class": "Accent Chairs", "rating": 4.5},
    ),
    Document(
        doc_id="2",
        title="blue linen accent chair",
        text="a calm blue chair",
        fields={"product_class": "Accent Chairs", "rating": 3.0},
    ),
    Document(
        doc_id="doc/3 with spaces",
        title="oak dining table",
        text="solid oak table seats six",
        fields={"product_class": "Dining Tables", "rating": 4.0},
    ),
]


@pytest.fixture
def api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def adapter(api: FakeApi) -> FakeRelevanAdapter:
    built = FakeRelevanAdapter(api, poll_interval=0.0)
    built.build(iter(CORPUS), dataset_name="tiny", cache_key="abc123")
    return built


def tools(adapter):
    return {spec.name: spec for spec in adapter.tools()}


# -- ids -------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc_id",
    ["1", "sku-12345", "doc/3 with spaces", "b32.notreallyencoded", "wands:42", "é"],
)
def test_document_ids_round_trip(doc_id):
    encoded = _encode_id(doc_id)
    assert re.match(r"^[A-Za-z0-9_.:-]+$", encoded)
    assert _decode_id(encoded) == doc_id


def test_safe_ids_are_left_readable():
    assert _encode_id("sku-12345") == "sku-12345"


def test_absurdly_long_ids_are_rejected_loudly():
    with pytest.raises(ValueError, match="too long"):
        _encode_id("x" * 600)


# -- build -----------------------------------------------------------------


def test_build_creates_the_index_maps_it_and_ingests(api, adapter):
    assert list(api.indexes) == ["sts-tiny-abc123"]
    assert len(api.documents) == 3
    fields = api.mappings[-1]["relevanMapping"]["fields"]
    assert fields["title"]["use"] == ["searchable", "highlightable", "autocompletable"]
    assert "searchable" in fields["text"]["use"]
    # A short, low-cardinality string field earns a filter; a number earns a range.
    assert "filterable" in fields["product_class"]["use"]
    assert fields["rating"]["kind"] == "number"
    # The harness's own id field is filterable in the mapping but never shown
    # to the agent as one.
    assert adapter.filter_fields == ["product_class"]
    assert adapter.range_fields == ["rating"]


def test_ingested_documents_carry_the_original_id(api, adapter):
    stored = [c[ID_FIELD] for c in api.documents.values()]
    assert sorted(stored) == ["1", "2", "doc/3 with spaces"]


def test_build_is_batched(api):
    corpus = [Document(doc_id=str(i), title=f"item {i}") for i in range(25)]
    built = FakeRelevanAdapter(api, poll_interval=0.0, batch_size=10, sample_size=5)
    built.build(iter(corpus), dataset_name="tiny", cache_key="k")
    bulk_calls = [c for c in api.calls if c[1].endswith("/documents/_bulk")]
    assert len(bulk_calls) == 3
    assert len(api.documents) == 25


def test_an_existing_ready_index_is_not_reingested(api, adapter):
    api.calls.clear()
    again = FakeRelevanAdapter(api, poll_interval=0.0)
    again.build(iter(CORPUS), dataset_name="tiny", cache_key="abc123")
    assert not [c for c in api.calls if c[1].endswith("/documents/_bulk")]
    # It still knows the schema: it read the deployed mapping.
    assert again.filter_fields == adapter.filter_fields


def test_reindex_option_ingests_again(api, adapter):
    api.calls.clear()
    again = FakeRelevanAdapter(api, poll_interval=0.0, reindex=True)
    again.build(iter(CORPUS), dataset_name="tiny", cache_key="abc123")
    assert [c for c in api.calls if c[1].endswith("/documents/_bulk")]


def test_build_waits_for_the_index_to_catch_up(api):
    api.ready = False
    built = FakeRelevanAdapter(api, poll_interval=0.0, poll_timeout=0.0)
    with pytest.raises(RuntimeError, match="still indexing"):
        built.build(iter(CORPUS), dataset_name="tiny", cache_key="k")


def test_fields_missing_from_the_mapping_sample_are_dropped_not_rejected(api):
    corpus = [
        Document(doc_id="1", title="first"),
        Document(doc_id="2", title="second", fields={"late_field": "surprise"}),
    ]
    built = FakeRelevanAdapter(api, poll_interval=0.0, sample_size=1)
    built.build(iter(corpus), dataset_name="tiny", cache_key="k")
    assert "late_field" not in api.mappings[-1]["relevanMapping"]["fields"]
    assert "late_field" not in api.documents["2"]


def test_a_field_that_repeats_the_body_is_not_indexed_twice(api):
    corpus = [Document(doc_id="1", title="t", text="body", fields={"description": "body"})]
    built = FakeRelevanAdapter(api, poll_interval=0.0)
    built.build(iter(corpus), dataset_name="tiny", cache_key="k")
    assert "description" not in api.mappings[-1]["relevanMapping"]["fields"]


def test_an_empty_corpus_fails_the_build(api):
    built = FakeRelevanAdapter(api, poll_interval=0.0)
    with pytest.raises(RuntimeError, match="corpus is empty"):
        built.build(iter([]), dataset_name="tiny", cache_key="k")


def test_several_organizations_ask_for_one(api):
    api.queue.append(
        (
            200,
            {"content-type": "application/json"},
            json.dumps([{"id": "a", "slug": "one"}, {"id": "b", "slug": "two"}]),
        )
    )
    built = FakeRelevanAdapter(api, poll_interval=0.0)
    with pytest.raises(RuntimeError, match="org=<slug>"):
        built.build(iter(CORPUS), dataset_name="tiny", cache_key="k")


def test_searching_before_build_is_the_adapters_bug(api):
    built = FakeRelevanAdapter(api)
    with pytest.raises(RuntimeError, match="build"):
        tools(built)["search"].handler({"query": "chair"})


# -- the tool contract -----------------------------------------------------


def test_only_search_spends_a_shot(adapter):
    kinds = {spec.name: spec.kind for spec in adapter.tools()}
    assert kinds["search"] is ToolKind.RETRIEVAL
    assert set(kinds.values()) - {ToolKind.RETRIEVAL} == {ToolKind.AUXILIARY}
    assert adapter.primary_tool().name == "search"


def test_explain_is_feedback_only(adapter):
    assert tools(adapter)["explain_result"].feedback_only


def test_the_feedback_tool_is_absent_unless_asked_for(api, adapter):
    assert "report_relevant_result" not in tools(adapter)
    opted_in = FakeRelevanAdapter(api, poll_interval=0.0, send_feedback=True)
    opted_in.build(iter(CORPUS), dataset_name="tiny", cache_key="abc123")
    assert "report_relevant_result" in tools(opted_in)


def test_filter_schema_offers_only_real_fields(adapter):
    schema = tools(adapter)["search"].input_schema["properties"]["filters"]
    assert schema["items"]["properties"]["field"]["enum"] == [
        "product_class",
        "rating",
    ]


def test_a_collection_without_filterable_fields_offers_no_filters(api):
    corpus = [Document(doc_id="1", title="only a title")]
    built = FakeRelevanAdapter(api, poll_interval=0.0)
    built.build(iter(corpus), dataset_name="tiny", cache_key="k")
    # Advertising a capability the backend does not have would only spend the
    # agent's turns on errors.
    assert "filters" not in tools(built)["search"].input_schema["properties"]
    # Pinning a search to one document still works: that is the id field.
    assert "explain_result" in tools(built)


# -- search ----------------------------------------------------------------


def test_search_returns_dataset_ids_not_encoded_ones(adapter):
    response = tools(adapter)["search"].handler({"query": "oak table", "k": 5})
    assert response.result_set.doc_ids == ["doc/3 with spaces"]
    assert response.body["results"][0]["doc_id"] == "doc/3 with spaces"


def test_search_ranks_and_renders_results(adapter):
    response = tools(adapter)["search"].handler({"query": "red chair", "k": 5})
    assert response.result_set.doc_ids[0] == "1"
    top = response.body["results"][0]
    assert top["title"] == "red velvet accent chair"
    assert "<em>" not in top["snippet"] and top["snippet"]
    assert top["product_class"] == "Accent Chairs"


def test_the_explanation_is_diagnostics_not_body(adapter):
    response = tools(adapter)["search"].handler({"query": "red chair"})
    rendered = json.dumps(response.body)
    assert "matched red" not in rendered
    assert response.diagnostics["top_result_explanation"] == "matched red, chair"
    # It is still recorded on the hit, so a finished run can be inspected.
    assert response.result_set.hits[0].extra["explanation"] == "matched red, chair"


def test_diagnostics_name_the_terms_that_matched_nothing(adapter):
    response = tools(adapter)["search"].handler({"query": "red flurblewotsit"})
    assert response.diagnostics["query_terms_with_no_match"] == ["flurblewotsit"]
    assert any("matched nothing" in note for note in response.diagnostics["notes"])


def test_filters_reach_the_api_and_narrow_the_results(adapter, api):
    response = tools(adapter)["search"].handler(
        {
            "query": "chair",
            "filters": [{"field": "product_class", "value": "Accent Chairs"}],
        }
    )
    assert set(response.result_set.doc_ids) == {"1", "2"}
    assert api.searches[-1]["filters"] == [
        {"field": "product_class", "value": "Accent Chairs"}
    ]


def test_range_filters_are_passed_through(adapter, api):
    response = tools(adapter)["search"].handler(
        {"query": "chair", "filters": [{"field": "rating", "gte": 4.0}]}
    )
    assert response.result_set.doc_ids == ["1"]


def test_a_range_bound_on_a_text_field_is_refused_with_the_alternative(adapter):
    response = tools(adapter)["search"].handler(
        {"query": "chair", "filters": [{"field": "product_class", "gte": 3}]}
    )
    assert "not a range field" in response.error


def test_unknown_filter_field_lists_the_valid_ones(adapter):
    response = tools(adapter)["search"].handler(
        {"query": "chair", "filters": [{"field": "colour", "value": "red"}]}
    )
    assert "colour" in response.error and "product_class" in response.error


def test_boost_is_forwarded(adapter, api):
    tools(adapter)["search"].handler({"query": "chair", "boost": {"chair": 2}})
    assert api.searches[-1]["boost"] == {"chair": 2.0}


def test_a_non_numeric_boost_is_refused(adapter):
    response = tools(adapter)["search"].handler(
        {"query": "chair", "boost": {"chair": "lots"}}
    )
    assert "must be a number" in response.error


def test_empty_query_is_rejected(adapter):
    assert tools(adapter)["search"].handler({"query": "  "}).error


def test_search_is_deterministic(adapter):
    first = tools(adapter)["search"].handler({"query": "chair", "k": 5})
    second = tools(adapter)["search"].handler({"query": "chair", "k": 5})
    assert first.result_set.doc_ids == second.result_set.doc_ids


def test_k_is_capped_at_the_api_maximum(adapter, api):
    tools(adapter)["search"].handler({"query": "chair", "k": 500})
    assert api.searches[-1]["size"] == 100


def test_an_api_error_reaches_the_agent_instead_of_raising(adapter, api):
    api.queue.append(
        (400, {"content-type": "application/json"}, json.dumps({"message": "Invalid filter."}))
    )
    response = tools(adapter)["search"].handler({"query": "chair"})
    assert response.error and "Invalid filter." in response.error
    assert "product_class" in response.error


def test_a_rate_limit_is_waited_out_then_retried(adapter, api, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(
        "shots_to_success.adapters.relevan.time.sleep", lambda s: slept.append(s)
    )
    api.queue.append(
        (
            429,
            {"content-type": "application/json", "retry-after": "2"},
            json.dumps({"message": "rate limited", "retryAfterSeconds": 2}),
        )
    )
    response = tools(adapter)["search"].handler({"query": "red chair"})
    assert slept == [2.0]
    assert response.error is None
    assert response.result_set.doc_ids[0] == "1"


def test_an_exhausted_rate_limit_is_reported_as_such(adapter, api):
    for _ in range(adapter.max_retries + 1):
        api.queue.append(
            (
                429,
                {"content-type": "application/json"},
                json.dumps({"message": "rate limited", "retryAfterSeconds": 0}),
            )
        )
    response = tools(adapter)["search"].handler({"query": "chair"})
    assert "rate limit" in response.error


def test_the_static_baseline_runs_the_unmodified_query(adapter):
    result_set = adapter.baseline("oak table", 10)
    assert result_set.doc_ids == ["doc/3 with spaces"]


# -- auxiliary tools -------------------------------------------------------


def test_describe_index_returns_the_indexes_own_guide(adapter):
    body = tools(adapter)["describe_index"].handler({}).body
    assert "Searchable fields" in body["guide"]
    assert body["filterable_fields"] == ["product_class"]


def test_describe_index_survives_the_guide_being_unavailable(adapter, api):
    api.queue.append((500, {"content-type": "application/json"}, json.dumps({"message": "boom"})))
    body = tools(adapter)["describe_index"].handler({}).body
    assert "unavailable" in body["guide"]


def test_explain_result_reports_why_a_document_scored(adapter, api):
    body = tools(adapter)["explain_result"].handler(
        {"query": "red chair", "doc_id": "1"}
    ).body
    assert body["matched"] is True
    assert body["matched_terms"] == ["red", "chair"]
    assert api.searches[-1]["filters"] == [{"field": ID_FIELD, "value": "1"}]


def test_explain_result_finds_a_document_whose_id_had_to_be_encoded(adapter, api):
    body = tools(adapter)["explain_result"].handler(
        {"query": "oak table", "doc_id": "doc/3 with spaces"}
    ).body
    assert body["matched"] is True
    # The filter names the dataset's id, which is what the document body holds;
    # the encoded form is only ever the API's own handle on the document.
    assert api.searches[-1]["filters"] == [
        {"field": ID_FIELD, "value": "doc/3 with spaces"}
    ]


def test_explain_result_says_so_when_a_document_does_not_match(adapter):
    body = tools(adapter)["explain_result"].handler(
        {"query": "oak table", "doc_id": "1"}
    ).body
    assert body["matched"] is False


def test_feedback_is_sent_as_an_eval_judgment_against_the_last_query(api):
    adapter = FakeRelevanAdapter(api, poll_interval=0.0, send_feedback=True)
    adapter.build(iter(CORPUS), dataset_name="tiny", cache_key="abc123")
    tools(adapter)["search"].handler({"query": "red chair"})
    response = tools(adapter)["report_relevant_result"].handler({"doc_id": "1"})
    assert response.error is None
    sent = api.feedback[-1]
    assert sent["actionName"] == "llm_judgment"
    assert sent["source"] == "eval"  # never "production": this is a benchmark
    assert sent["queryId"] == "query-1"
    assert sent["objectId"] == "1"
    assert sent["position"] == 0


def test_feedback_before_a_search_is_refused(api):
    adapter = FakeRelevanAdapter(api, poll_interval=0.0, send_feedback=True)
    adapter.build(iter(CORPUS), dataset_name="tiny", cache_key="abc123")
    assert tools(adapter)["report_relevant_result"].handler({"doc_id": "1"}).error


# -- provenance ------------------------------------------------------------


def test_describe_records_the_setup_and_not_the_key(api):
    adapter = FakeRelevanAdapter(api, poll_interval=0.0, api_key="secret")
    adapter.build(iter(CORPUS), dataset_name="tiny", cache_key="abc123")
    info = adapter.describe()
    assert info["name"] == "relevan"
    assert info["index"] == "sts-tiny-abc123"
    assert info["org"] == "acme"
    assert info["mapping_revision"] == 1
    assert "secret" not in json.dumps(info)


# -- term comparison -------------------------------------------------------


def test_unmatched_terms_tolerate_stemming():
    assert _unmatched_terms("red chairs", ["red", "chair"]) == []
    assert _unmatched_terms("red sofa", ["red"]) == ["sofa"]
    # Short words carry no signal either way.
    assert _unmatched_terms("a of the", []) == []
