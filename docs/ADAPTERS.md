# Writing a search adapter

An adapter is how a search system enters the benchmark. It wraps your backend
and exposes it to the agent as a set of tools. The harness makes no assumptions
about how retrieval works — it only needs each tool to declare whether calling
it spends a shot, and for retrieval tools to return a ranked list of document
ids it can grade.

You do not need to modify this repository. Point `--adapter` at an import path:

```bash
sts run --adapter mypkg.adapters:MyAdapter --adapter-opt base_url=https://search.internal
```

A complete worked example against a hosted HTTP service is in
[`examples/http_adapter.py`](../examples/http_adapter.py).

## The interface

```python
from shots_to_success.adapters import SearchAdapter
from shots_to_success.types import Hit, ResultSet, ToolKind, ToolResponse, ToolSpec


class MyAdapter(SearchAdapter):
    name = "my-engine"
    version = "1"

    def build(self, corpus, *, dataset_name, cache_key=""):
        """Ingest the corpus. A no-op is fine if your backend already has it."""

    def tools(self):
        """Every tool your backend exposes. The harness filters by condition."""
        return [ToolSpec(...)]

    def describe(self):
        """Provenance for the run manifest. Never include credentials."""
        return super().describe()

    def close(self):
        """Release connections."""
```

`__init__` receives whatever `--adapter-opt key=value` flags were passed, as
`self.options`. Values are JSON-decoded when possible, so `--adapter-opt k1=1.5`
arrives as a float and `--adapter-opt fields='["title"]'` as a list.

## Tool kinds: what counts as a shot

This is the single most important decision in an adapter.

| Kind | Spends a shot | For |
|---|---|---|
| `ToolKind.RETRIEVAL` | Yes | Anything returning ranked documents that could be submitted as the answer |
| `ToolKind.AUXILIARY` | No | Autocomplete, facets, spell check, schema inspection, score explanation |

The benchmark's primary metric counts retrieval attempts. If a provider offers
an autocomplete endpoint that helps the agent build a better query before
searching, that help should be free — otherwise the metric punishes systems for
offering it. Conversely, marking a tool auxiliary when it actually returns the
answer would let a system search for free and make its score meaningless.

The rule: if the agent could submit its output as the final result set, it is
retrieval.

## Returning results

A retrieval handler returns a `ToolResponse` with `result_set` set:

```python
def _search(self, args):
    rows = my_backend.query(args["query"], size=args.get("k", 10))
    hits = [
        Hit(
            doc_id=str(row.id),          # must match the dataset's doc ids
            score=row.score,
            title=row.title,
            snippet=row.snippet,
            extra={"category": row.category},
        )
        for row in rows
    ]
    return ToolResponse(
        body={"results": [...]},          # what the agent reads
        result_set=ResultSet(hits=hits, query=args["query"], total=rows.total),
        diagnostics={...},                # feedback conditions only
    )
```

`doc_id` must match the ids the dataset yields from `corpus()`. If they do not
line up, every episode scores zero and nothing else will tell you why — this is
the most common adapter bug. Check it before a large run:

```bash
sts run --condition static --limit 20   # should not be 0% on an easy dataset
```

`body` is what the agent sees. `result_set` is what the grader scores. Keeping
them separate means you can render results however reads best without changing
what is measured.

## Diagnostics: the Feedback Track's lever

`ToolResponse.diagnostics` is rendered to the agent **only** in conditions with
feedback enabled. Everything else about the response stays identical, which is
what makes a `results_only` vs `feedback` comparison attributable to feedback
alone.

Put anything your engine can honestly say about why it returned these results:

- which query terms matched, and their document frequencies
- terms that matched nothing at all — the strongest single signal, because it
  distinguishes "the collection doesn't have this" from "you asked wrong"
- how many documents survived each filter
- per-term score contributions for the top result
- the analyzer's output, if it differs from the raw query

Diagnostics must be derived from index statistics only. Anything traceable to
relevance judgments is leakage and invalidates the run.

## Errors

Return `ToolResponse(error="...")` rather than raising. A failed search is
something the agent should be able to read and recover from, and recovery is
what the benchmark measures. Make the message actionable — list the valid
filter names rather than saying "invalid filter".

Exceptions that escape a handler are caught, reported to the agent as a tool
error, and recorded; they will not kill the run, but they are your bug.

## Determinism

Two runs of the same query must return the same ranking. Non-determinism shows
up as noise in `bad_retry_rate` and makes conditions incomparable. Break score
ties on a stable key such as document id.

## Registering it

There is nothing to register. `--adapter mypkg.module:MyAdapter` imports the
class directly. The only requirement is that it subclasses `SearchAdapter` and
is importable from the working directory or an installed package.

## Checklist before a real run

- [ ] `doc_id`s match the dataset's ids exactly (string type included)
- [ ] Exactly one retrieval tool, unless you deliberately offer several
- [ ] Auxiliary tools genuinely cannot return the answer
- [ ] `describe()` includes every ranking-affecting setting, and no credentials
- [ ] Same query twice gives the same ranking
- [ ] `sts run --condition static --limit 20` gives a plausible non-zero score
