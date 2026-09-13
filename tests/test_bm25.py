"""The reference adapter: ranking, filters, diagnostics, and the tool surface."""

from __future__ import annotations

from shots_to_success.adapters.bm25_index import light_stem, tokenize
from shots_to_success.types import ToolKind


def tools(adapter):
    return {spec.name: spec for spec in adapter.tools()}


def test_light_stemmer_folds_plurals():
    assert light_stem("chairs") == "chair"
    assert light_stem("boxes") == "box"
    assert light_stem("ladies") == "lady"
    # Short words and double-s words are left alone.
    assert light_stem("gas") == "gas"
    assert light_stem("glass") == "glass"
    assert tokenize("Red Chairs!") == ["red", "chair"]


def test_search_ranks_the_matching_document_first(adapter):
    response = tools(adapter)["search"].handler({"query": "red chair", "k": 3})
    assert response.result_set.doc_ids[0] == "1"
    assert response.body["results"][0]["title"] == "red velvet accent chair"


def test_search_is_deterministic(adapter):
    first = tools(adapter)["search"].handler({"query": "chair", "k": 5})
    second = tools(adapter)["search"].handler({"query": "chair", "k": 5})
    assert first.result_set.doc_ids == second.result_set.doc_ids


def test_missing_terms_are_reported(adapter):
    response = tools(adapter)["search"].handler({"query": "red flurblewotsit", "k": 3})
    assert response.diagnostics["missing_terms"] == ["flurblewotsit"]
    assert any("appear in no document" in note for note in response.diagnostics["notes"])


def test_filters_narrow_results_and_are_accounted_for(adapter):
    response = tools(adapter)["search"].handler(
        {"query": "chair", "k": 5, "filters": {"product_class": "Accent Chairs"}}
    )
    assert set(response.result_set.doc_ids) == {"1", "2"}
    diagnostics = response.diagnostics
    assert diagnostics["documents_after_filters"] < diagnostics["documents_matching_any_term"]
    assert any("Filters removed" in note for note in diagnostics["notes"])


def test_filter_that_matches_nothing_says_so(adapter):
    response = tools(adapter)["search"].handler(
        {"query": "chair", "filters": {"product_class": "Nope"}}
    )
    assert response.result_set.hits == []
    assert any("removed all" in note for note in response.diagnostics["notes"])


def test_unknown_filter_is_an_error_that_lists_the_valid_ones(adapter):
    response = tools(adapter)["search"].handler(
        {"query": "chair", "filters": {"colour": "red"}}
    )
    assert response.error and "colour" in response.error
    assert "product_class" in response.error


def test_empty_query_is_rejected(adapter):
    assert tools(adapter)["search"].handler({"query": "  "}).error


def test_autocomplete_suggests_from_the_catalog(adapter):
    response = tools(adapter)["autocomplete"].handler({"prefix": "red", "limit": 5})
    assert any(s["text"].startswith("red") for s in response.body["suggestions"])


def test_facets_count_values_across_matches(adapter):
    response = tools(adapter)["facets"].handler(
        {"query": "chair", "field": "product_class"}
    )
    values = {v["value"]: v["documents"] for v in response.body["values"]}
    assert values["Accent Chairs"] == 2


def test_explain_shows_why_a_document_scores(adapter):
    response = tools(adapter)["explain_result"].handler(
        {"query": "red chair", "doc_id": "3"}
    )
    assert response.body["total_score"] == 0.0
    assert set(response.body["terms_not_in_document"]) == {"red", "chair"}


def test_tool_kinds_are_declared(adapter):
    kinds = {name: spec.kind for name, spec in tools(adapter).items()}
    assert kinds["search"] is ToolKind.RETRIEVAL
    assert kinds["autocomplete"] is ToolKind.AUXILIARY
    assert all(
        kind is ToolKind.AUXILIARY
        for name, kind in kinds.items()
        if name != "search"
    )


# -- corpora with no structured fields (most BEIR datasets) ----------------


def _unstructured_adapter():
    from shots_to_success.adapters.bm25 import Bm25Adapter
    from shots_to_success.types import Document

    built = Bm25Adapter(use_cache=False)
    built.build(
        [
            Document(doc_id="1", title="a study of red chairs", text="chairs, red"),
            Document(doc_id="2", title="blue sofa report", text="sofas"),
        ],
        dataset_name="plain",
    )
    return built


def test_a_corpus_without_fields_advertises_no_filters_or_facets():
    adapter = _unstructured_adapter()
    names = set(tools(adapter))
    assert "facets" not in names
    assert "filters" not in tools(adapter)["search"].input_schema["properties"]


def test_filters_on_an_unfilterable_corpus_are_rejected_clearly():
    adapter = _unstructured_adapter()
    response = tools(adapter)["search"].handler(
        {"query": "red", "filters": {"anything": "x"}}
    )
    assert "no filterable fields" in (response.error or "")


def test_search_still_works_without_fields():
    adapter = _unstructured_adapter()
    assert tools(adapter)["search"].handler({"query": "red chairs"}).result_set.doc_ids[0] == "1"


# -- long queries ----------------------------------------------------------


def test_a_repeated_query_term_is_scored_once(adapter):
    """BM25 scores each distinct query term once. Counting a repeat twice
    silently over-weights it -- 'home sweet home sign' is a real query."""
    once = tools(adapter)["search"].handler({"query": "red chair", "k": 5})
    twice = tools(adapter)["search"].handler({"query": "red red chair", "k": 5})
    assert once.result_set.hits[0].score == twice.result_set.hits[0].score


def test_long_queries_keep_only_the_most_discriminative_terms(adapter):
    index = adapter.index
    index.max_query_terms = 2
    # "chair" is common in this corpus, "velvet" is rare, so velvet survives.
    selected = index.select_terms(["chair", "velvet", "red"])
    assert "velvet" in selected and len(selected) == 2


def test_term_selection_is_deterministic_and_leaves_short_queries_alone(adapter):
    index = adapter.index
    index.max_query_terms = 50
    assert index.select_terms(["red", "chair"]) == ["red", "chair"]
    index.max_query_terms = 1
    assert index.select_terms(["red", "chair"]) == index.select_terms(["chair", "red"])


def test_score_contributors_are_reported_for_returned_hits(adapter):
    response = tools(adapter)["search"].handler({"query": "red chair", "k": 1})
    contributors = response.result_set.hits[0].extra["score_contributors"]
    assert set(contributors) == {"red", "chair"}
    assert abs(sum(contributors.values()) - response.result_set.hits[0].score) < 1e-3
