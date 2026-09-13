"""Dataset loaders: parsing, splits, and the label-leakage boundary."""

from __future__ import annotations

import json
import zipfile

import pytest

from shots_to_success.datasets import load_dataset
from shots_to_success.datasets.beir import BeirDataset
from shots_to_success.datasets.trec_tot import TrecTotDataset, _clean_markup

# -- BEIR ------------------------------------------------------------------


@pytest.fixture
def beir_dir(tmp_path):
    root = tmp_path / "beir" / "fiqa"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"_id": "d1", "title": "Roth IRA basics", "text": "contribution limits"},
                {"_id": "d2", "title": "Mortgage rates", "text": "fixed vs variable"},
            ]
        )
    )
    (root / "queries.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"_id": "q1", "text": "how much can I put in a Roth IRA"},
                {"_id": "q2", "text": "unjudged query"},
            ]
        )
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n"
    )
    return tmp_path


def test_beir_parses_the_standard_three_file_layout(beir_dir):
    dataset = BeirDataset(data_dir=beir_dir, subset="fiqa")
    docs = list(dataset.corpus())
    assert [d.doc_id for d in docs] == ["d1", "d2"]
    assert docs[0].title == "Roth IRA basics"
    assert dataset.judgments().relevant_docs("q1") == {"d1"}


def test_beir_keeps_only_queries_with_judgments_in_the_split(beir_dir):
    dataset = BeirDataset(data_dir=beir_dir, subset="fiqa")
    # q2 has no qrel, so it cannot be scored and is not an episode.
    assert [q.query_id for q in dataset.queries()] == ["q1"]


def test_beir_reports_the_subset_so_each_gets_its_own_index_cache(beir_dir):
    a = BeirDataset(data_dir=beir_dir, subset="fiqa").describe()
    b = BeirDataset(data_dir=beir_dir, subset="nfcorpus").describe()
    assert a != b and a["subset"] == "fiqa"


def test_beir_rejects_nested_subsets_clearly(beir_dir):
    with pytest.raises(ValueError, match="cqadupstack"):
        BeirDataset(data_dir=beir_dir, subset="cqadupstack/android")


def test_beir_missing_split_names_what_is_available(beir_dir):
    dataset = BeirDataset(data_dir=beir_dir, subset="fiqa", split="train")
    with pytest.raises(FileNotFoundError, match="test"):
        dataset.judgments()


# -- TREC ToT --------------------------------------------------------------


@pytest.fixture
def tot_dir(tmp_path):
    root = tmp_path / "trec-tot"
    root.mkdir(parents=True)
    corpus = [
        {
            "doc_id": "330",
            "page_title": "Actrius",
            "text": "A 1997 Catalan drama film about three actresses.",
            "wikidata_classes": [["Q11424", "film"]],
            "infoboxes": [
                {
                    "name": "film",
                    "params": {
                        "director": "[[Ventura Pons]]",
                        "country": "Spain",
                        "released": "{{Film date|1997|01|24}}",
                        "starring": "ubl|[[Nuria Espert]]|[[Rosa Sarda]]",
                    },
                }
            ],
        }
    ]
    with zipfile.ZipFile(root / "TREC-ToT.zip", "w") as zf:
        zf.writestr(
            "TREC-TOT/corpus.jsonl", "\n".join(json.dumps(r) for r in corpus)
        )
        zf.writestr(
            "TREC-TOT/dev/queries.jsonl",
            json.dumps(
                {
                    "id": "152",
                    "text": "that film about three actresses, Spanish I think",
                    "title": "forum post title",
                    "wikipedia_id": "330",
                    "wikipedia_url": "https://en.wikipedia.org/wiki/Actrius",
                    "imdb_url": "https://imdb.com/title/tt0118577",
                }
            ),
        )
        zf.writestr("TREC-TOT/dev/qrel.txt", "152 0 330 1\n")
    return tmp_path


def test_tot_reads_documents_and_structured_infobox_fields(tot_dir):
    dataset = TrecTotDataset(data_dir=tot_dir, split="dev")
    doc = next(iter(dataset.corpus()))
    assert doc.doc_id == "330" and doc.title == "Actrius"
    assert doc.fields["type"] == "film"
    assert doc.fields["director"] == "Ventura Pons"
    assert doc.fields["year"] == "1997"
    assert doc.fields["decade"] == "1990s"


def test_tot_has_exactly_one_correct_answer_per_query(tot_dir):
    judgments = TrecTotDataset(data_dir=tot_dir, split="dev").judgments()
    assert judgments.relevant_docs("152") == {"330"}


def test_tot_never_leaks_the_answer_into_the_episode(tot_dir):
    """The query records carry the answer's Wikipedia and IMDb ids. If any of
    them reached the agent the benchmark would measure nothing."""
    query = TrecTotDataset(data_dir=tot_dir, split="dev").queries()[0]
    blob = (query.text + json.dumps(query.metadata)).lower()
    for leak in ("wikipedia_id", "wikipedia.org", "imdb", "actrius", "330"):
        assert leak not in blob


def test_tot_truncation_is_configurable_and_recorded(tot_dir):
    dataset = TrecTotDataset(data_dir=tot_dir, split="dev", max_text_chars=10)
    assert len(next(iter(dataset.corpus())).text) == 10
    assert dataset.describe()["max_text_chars"] == 10


def test_tot_can_combine_splits(tot_dir):
    dataset = TrecTotDataset(data_dir=tot_dir, split="dev")
    assert dataset._splits() == ["dev"]
    assert TrecTotDataset(data_dir=tot_dir, split="train+dev")._splits() == [
        "train",
        "dev",
    ]


def test_tot_missing_archive_says_how_to_get_it(tmp_path):
    dataset = TrecTotDataset(data_dir=tmp_path, split="dev")
    with pytest.raises(FileNotFoundError, match="sts prepare"):
        dataset.queries()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[[Ventura Pons]]", "Ventura Pons"),
        ("ubl|[[A]]|[[B]]", "A, B"),
        ("[[Catalan language|Catalan]]", "Catalan"),
        ("", ""),
        (None, ""),
    ],
)
def test_wiki_markup_is_stripped(raw, expected):
    assert _clean_markup(raw) == expected


# -- registry --------------------------------------------------------------


def test_all_builtin_datasets_resolve_by_name():
    for name in ("wands", "beir", "trec-tot"):
        assert load_dataset(name, data_dir="/nonexistent").name == name
