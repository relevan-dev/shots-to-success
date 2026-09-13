"""A small, dependency-free BM25F-lite index.

Written in plain Python on purpose: a benchmark's reference backend should be
readable and reproducible by anyone who wants to check what it did, without
pulling in a search engine. It supports the things the benchmark needs to give
an agent honest feedback -- per-term document frequencies, per-field matches,
score decomposition, and filter accounting.
"""

from __future__ import annotations

import bisect
import heapq
import math
import re
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Field boosts applied at term-frequency level (BM25F-lite): a term occurring
#: in the title counts as if it occurred `boost` times in a single bag of
#: words. Part of the index definition, so it is reported in the manifest.
#:
#: Fields are discovered from the documents themselves; anything not named
#: here is indexed at DEFAULT_BOOST. Short categorical fields are worth more
#: than body text per occurrence, which is what the named entries encode.
DEFAULT_FIELD_BOOSTS: dict[str, float] = {
    "title": 3.0,
    "product_class": 2.0,
    "category": 1.0,
    "text": 1.0,
}
DEFAULT_BOOST = 1.0

#: Long verbose queries (a tip-of-the-tongue description runs to ~120 terms)
#: are dominated by common words whose posting lists are enormous and whose
#: idf is near zero. Keeping only the most discriminative terms is what makes
#: such datasets tractable, and it barely moves the ranking. Short queries are
#: unaffected.
DEFAULT_MAX_QUERY_TERMS = 50


def light_stem(token: str) -> str:
    """Deterministic suffix-stripping stemmer.

    Product search lives or dies on plural matching ("chairs" vs "chair"), but
    a full Porter stemmer would make the "missing terms" diagnostic hard to
    interpret. These three rules are the useful 90%.
    """
    if len(token) > 3 and not token.isdigit():
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 4 and token[-3] in "sxzh":
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [light_stem(t) for t in TOKEN_RE.findall(text.lower())]


@dataclass(slots=True)
class TermStat:
    term: str
    df: int
    idf: float


@dataclass(slots=True)
class ScoredDoc:
    index: int
    score: float
    #: Per-term score contribution, for explain and score-contributor feedback.
    contributions: dict[str, float] = field(default_factory=dict)


class Bm25Index:
    """Inverted index with BM25 scoring over boosted fields."""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        field_boosts: dict[str, float] | None = None,
        max_query_terms: int = DEFAULT_MAX_QUERY_TERMS,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.max_query_terms = max_query_terms
        self.field_boosts = dict(field_boosts or DEFAULT_FIELD_BOOSTS)
        self.doc_ids: list[str] = []
        self.doc_index: dict[str, int] = {}
        self.titles: list[str] = []
        self.snippets: list[str] = []
        self.meta: list[dict[str, Any]] = []
        #: term -> (doc indices, boosted term frequencies)
        self.postings: dict[str, tuple[array, array]] = {}
        #: term -> set of fields it occurs in, per doc, kept only as a
        #: term -> field-name bitmask to keep memory flat.
        self.term_fields: dict[str, int] = {}
        self.doc_len: array = array("f")
        self.avgdl: float = 0.0
        self.suggest_phrases: list[str] = []
        self.suggest_counts: list[int] = []
        self._building: dict[str, tuple[list[int], list[float]]] | None = None
        # Field order is discovered as documents arrive, so the index works on
        # any dataset without being told its schema up front. Positions are
        # append-only because they index into the per-term field bitmask.
        self._field_order: list[str] = []

    def _field_position(self, name: str) -> int:
        try:
            return self._field_order.index(name)
        except ValueError:
            if len(self._field_order) >= 63:
                raise ValueError(
                    "Bm25Index supports at most 63 indexed fields"
                ) from None
            self._field_order.append(name)
            return len(self._field_order) - 1

    def boost(self, name: str) -> float:
        return self.field_boosts.get(name, DEFAULT_BOOST)

    # -- build -----------------------------------------------------------

    def start(self) -> None:
        self._building = defaultdict(lambda: ([], []))
        self._phrase_counter: Counter[str] = Counter()

    def add(self, doc_id: str, title: str, snippet: str, fields: dict[str, str], meta: dict[str, Any]) -> None:
        assert self._building is not None, "call start() before add()"
        index = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_index[doc_id] = index
        self.titles.append(title)
        self.snippets.append(snippet)
        self.meta.append(meta)

        weighted: dict[str, float] = defaultdict(float)
        field_mask: dict[str, int] = defaultdict(int)
        length = 0.0
        for name, value in fields.items():
            if not value:
                continue
            position = self._field_position(name)
            boost = self.boost(name)
            tokens = tokenize(value)
            length += boost * len(tokens)
            for term, count in Counter(tokens).items():
                weighted[term] += boost * count
                field_mask[term] |= 1 << position
        for term, tf in weighted.items():
            docs, freqs = self._building[term]
            docs.append(index)
            freqs.append(tf)
            self.term_fields[term] = self.term_fields.get(term, 0) | field_mask[term]
        self.doc_len.append(length)

        # Autocomplete surface: the product class plus leading n-grams of the
        # title, which is what a catalog-backed suggester would offer.
        # Short categorical values make good suggestions; long body text does
        # not. No field name is assumed, so this works on any dataset.
        for value in meta.values():
            if isinstance(value, str) and 0 < len(value.strip()) <= 60:
                self._phrase_counter[value.strip().lower()] += 1
        title_tokens = TOKEN_RE.findall(title.lower())[:5]
        for n in range(1, min(4, len(title_tokens)) + 1):
            self._phrase_counter[" ".join(title_tokens[:n])] += 1

    def finalize(self, min_suggest_count: int = 2) -> None:
        assert self._building is not None, "call start() before finalize()"
        for term, (docs, freqs) in self._building.items():
            self.postings[term] = (array("i", docs), array("f", freqs))
        self._building = None
        n = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        phrases = sorted(
            (p, c)
            for p, c in self._phrase_counter.items()
            if c >= min_suggest_count and len(p) >= 3
        )
        self.suggest_phrases = [p for p, _ in phrases]
        self.suggest_counts = [c for _, c in phrases]
        del self._phrase_counter

    # -- statistics ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.doc_ids)

    def df(self, term: str) -> int:
        posting = self.postings.get(term)
        return len(posting[0]) if posting else 0

    def idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.df(term)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def term_stats(self, terms: Sequence[str]) -> list[TermStat]:
        return [TermStat(term=t, df=self.df(t), idf=self.idf(t)) for t in terms]

    def fields_for_term(self, term: str) -> list[str]:
        mask = self.term_fields.get(term, 0)
        return [name for i, name in enumerate(self._field_order) if mask & (1 << i)]

    def docs_matching_all(self, terms: Sequence[str]) -> int:
        """How many documents contain every term. Distinguishes a query that is
        too narrow (0) from one that is too broad (thousands)."""
        present = [t for t in set(terms) if self.df(t) > 0]
        if not present or len(present) != len(set(terms)):
            return 0
        present.sort(key=self.df)
        acc: set[int] | None = None
        for term in present:
            docs = set(self.postings[term][0])
            acc = docs if acc is None else (acc & docs)
            if not acc:
                return 0
        return len(acc or ())

    # -- search ----------------------------------------------------------

    def select_terms(self, terms: Sequence[str]) -> list[str]:
        """Keep the most discriminative distinct terms, highest idf first.

        Ties break on the term itself, so selection is deterministic.
        """
        distinct = list(dict.fromkeys(terms))
        if len(distinct) <= self.max_query_terms:
            return distinct
        ranked = sorted(distinct, key=lambda t: (-self.idf(t), t))
        return ranked[: self.max_query_terms]

    def score(self, terms: Sequence[str]) -> dict[int, float]:
        """BM25 over the union of posting lists.

        Accumulates into a flat doc -> score map rather than per-document
        objects; on a large index with a long query that difference dominates
        runtime.
        """
        scored: dict[int, float] = {}
        k1, b, avgdl, doc_len = self.k1, self.b, self.avgdl, self.doc_len
        get = scored.get
        for term in self.select_terms(terms):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            docs, freqs = posting
            for doc_index, tf in zip(docs, freqs):
                norm = 1.0 - b + b * (doc_len[doc_index] / avgdl)
                contribution = idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
                scored[doc_index] = get(doc_index, 0.0) + contribution
        return scored

    def search(
        self,
        terms: Sequence[str],
        k: int,
        predicate: Callable[[int], bool] | None = None,
        explain: bool = False,
    ) -> tuple[list[ScoredDoc], int, int]:
        """Return ``(top_k, matched_before_filter, matched_after_filter)``."""
        scored = self.score(terms)
        before = len(scored)
        items: Any = scored.items()
        if predicate is not None:
            items = [pair for pair in items if predicate(pair[0])]
        after = len(items)
        # Stable ordering: score desc, then doc index asc, so reruns of the
        # same query are byte-identical.
        top = heapq.nsmallest(k, items, key=lambda kv: (-kv[1], kv[0]))
        out = [ScoredDoc(index=index, score=score) for index, score in top]
        if explain:
            # Decompose only the documents actually returned. Doing it for
            # every candidate is what made long queries slow.
            selected = self.select_terms(terms)
            for entry in out:
                entry.contributions = {
                    term: value
                    for term, value in self.explain_doc(selected, entry.index).items()
                    if value
                }
        return out, before, after

    def explain_doc(self, terms: Sequence[str], doc_index: int) -> dict[str, float]:
        contributions: dict[str, float] = {}
        for term in terms:
            posting = self.postings.get(term)
            if not posting:
                contributions[term] = 0.0
                continue
            docs, freqs = posting
            position = _index_of(docs, doc_index)
            if position < 0:
                contributions[term] = 0.0
                continue
            tf = freqs[position]
            norm = 1.0 - self.b + self.b * (self.doc_len[doc_index] / self.avgdl)
            contributions[term] = (
                self.idf(term) * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
            )
        return contributions

    # -- suggest ---------------------------------------------------------

    def suggest(self, prefix: str, limit: int = 10) -> list[tuple[str, int]]:
        prefix = " ".join(TOKEN_RE.findall(prefix.lower()))
        if not prefix:
            return []
        start = bisect.bisect_left(self.suggest_phrases, prefix)
        out: list[tuple[str, int]] = []
        for i in range(start, len(self.suggest_phrases)):
            phrase = self.suggest_phrases[i]
            if not phrase.startswith(prefix):
                break
            out.append((phrase, self.suggest_counts[i]))
        out.sort(key=lambda pair: (-pair[1], pair[0]))
        return out[:limit]

    def describe(self) -> dict[str, Any]:
        return {
            "docs": len(self.doc_ids),
            "terms": len(self.postings),
            "avgdl": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
            "max_query_terms": self.max_query_terms,
            "indexed_fields": {
                name: self.boost(name) for name in self._field_order
            },
            "analyzer": "lowercase + [a-z0-9]+ + light suffix stemmer",
        }


def _index_of(sorted_ints: array, value: int) -> int:
    """Posting lists are appended in ascending doc order, so bisect works."""
    position = bisect.bisect_left(sorted_ints, value)
    if position < len(sorted_ints) and sorted_ints[position] == value:
        return position
    return -1


__all__ = [
    "DEFAULT_FIELD_BOOSTS",
    "Bm25Index",
    "ScoredDoc",
    "TermStat",
    "light_stem",
    "tokenize",
]
