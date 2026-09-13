# The Relevan adapter

Runs the benchmark against [Relevan](https://relevan.dev)'s hosted search API
(`api.relevan.dev`) instead of the in-process BM25 baseline. It is a built-in,
so there is nothing to import:

```bash
export RELEVAN_API_KEY=...
sts prepare --dataset trec-tot
sts sweep --dataset trec-tot --adapter relevan \
    --conditions static,results_only,feedback --limit 25
```

The first run ingests the corpus, which is the slow part. Every later run
against the same dataset reuses the index (see *Index reuse* below).

## Options

Pass them as `--adapter-opt key=value`; values are JSON-decoded where possible.

| Option | Default | What it does |
|---|---|---|
| `api_key` | `$RELEVAN_API_KEY` | Org-scoped key, sent as `x-api-key`. Never recorded in the manifest. |
| `org` | resolved from the key | Organization id or slug. Only needed if the key resolves to more than one. |
| `index` | `sts-<dataset>-<cache key>` | Index to search. Give it a name to point at an index you already ingested. |
| `base_url` | `https://api.relevan.dev` | For a staging deployment. |
| `reindex` | `false` | Ingest again even if the index already holds documents (upserts by id). |
| `send_feedback` | `false` | Exposes the feedback tool. Read *Feedback* below before turning it on. |
| `sample_size` | `1000` | Documents read before the mapping is fixed. |
| `batch_size` | `1000` | Documents per bulk request (the API's own cap). |
| `language` | `english` | Analyzer language for text fields. |
| `poll_interval` / `poll_timeout` | `5` / `1800` | How long to wait for ingest to become searchable. |
| `timeout` / `max_retries` | `30` / `3` | Per-request timeout, and retries on a 429 or a transient network error. |

## What `build` does

1. **Resolves the organization** from the key (`GET /v1/organizations`) unless
   `org` was given.
2. **Names the index** `sts-<dataset>-<cache key>`, where the cache key is the
   harness's hash of the dataset configuration and the adapter version.
3. **Creates it** if it does not exist.
4. **Infers a mapping** from the first `sample_size` documents and deploys it
   as a mapping revision. Field uses are derived, not hard-coded:

   | Source | Mapping |
   |---|---|
   | `Document.title` | `searchable`, `highlightable`, `autocompletable` |
   | `Document.text` | `searchable`, `highlightable` |
   | short, low-cardinality string field | `searchable`, `filterable`, `aggregatable` |
   | long or high-cardinality string field | `searchable`, `highlightable` |
   | number | `filterable`, `sortable` |
   | boolean | `filterable` |

   A field whose value merely repeats `Document.text` is dropped, so its terms
   are not counted twice. Fields that first appear after the sample are dropped
   too — the whole batch would otherwise be rejected for a field the mapping
   does not declare. Raise `sample_size` if a dataset has sparse fields you
   care about; the build logs any field it dropped.
5. **Bulk-ingests** in batches of 1000 (`POST /documents/_bulk`).
6. **Waits** for `readyToSearch`, reporting `documentsIndexed` as it climbs.

### Index reuse

Ingest is skipped when the index already exists, holds documents, and reports
`readyToSearch`. Because the default index name embeds the dataset's cache key,
an index under that name *is* this corpus — which makes the check one API call
rather than a full scan of a multi-gigabyte dataset. Pass `reindex=true` to
ingest anyway.

If you point `index` at something you ingested yourself, the adapter reads the
deployed mapping and works from that. Two things are then on you: the document
ids must be the dataset's own, and the corpus must be the one the judgments
were made against. Mismatched ids score every episode zero and nothing else
will tell you why.

### Document ids

Relevan accepts `[A-Za-z0-9_.:-]` in an id; the benchmark needs the ids handed
back to be the dataset's, exactly. Ids inside that set pass through unchanged.
Anything else is base32-encoded on the way in and decoded on the way out, and
the original is also stored in the document body as `sts_doc_id`, which is what
a hit is read from first.

## The tool surface

| Tool | Kind | Costs a shot | API |
|---|---|---|---|
| `search` | retrieval | yes | `POST /indexes/{index}/search` |
| `describe_index` | auxiliary | no | `GET /indexes/{index}/skill` |
| `explain_result` | auxiliary, feedback-only | no | a search pinned to one document id |
| `report_relevant_result` | auxiliary, opt-in | no | `POST /v1-beta/.../feedback` |

`search` exposes the query, `k`, Relevan's filters (exact, set membership, and
range) and its per-term `boost`. Filter fields come from the deployed mapping,
so a collection with no filterable fields gets no `filters` argument at all
rather than an argument that only produces errors.

`describe_index` returns the index's own skill doc — the Markdown guide Relevan
generates from the live mapping. It returns no documents, so it is free.

`explain_result` runs a search pinned to one document id to report why that
document scored what it did. It costs an API request but not a shot: it can
only speak about a document the agent has already seen, so it cannot stand in
for a search.

### What lands in diagnostics

Everything from `searchContext`, which is what the Feedback Track exists to
measure: `queryUsed`, `totalHits`, `matchedTerms`, `topContributingFields`,
`scoringSummary`, `searchableFields`, Relevan's own reformulation
`suggestions`, and the top result's `explanation`. The adapter adds the query
words that no indexed term resembles — stopwords excluded, and compared on
stems, because a term reported as missing when it merely stemmed differently
is worse than saying nothing.

None of it comes from relevance judgments. An adapter never sees them.

## Feedback

Relevan asks every caller to report what it did with a result, because that is
what trains its ranking. A benchmark needs the opposite: rankings that hold
still, so a difference between two runs is attributable to the condition rather
than to what an earlier episode taught the index.

So `send_feedback` defaults to false. Turn it on and the agent gets a
`report_relevant_result` tool for the results it judged relevant; every report
is sent with `source: "eval"` — the value the API defines for a benchmark run
rather than real traffic — and `actionName: "llm_judgment"`, since it is the
agent's own judgment with no human confirming it. The hidden labels are never
sent, because the adapter cannot see them.

Runs made with feedback on are not comparable to runs made with it off, and are
not reproducible against each other. The manifest records the setting.

## Before a real run

The checklist in [ADAPTERS.md](ADAPTERS.md) applies. Two entries deserve extra
attention here:

```bash
# ids line up and the index is really searchable
sts run --dataset trec-tot --adapter relevan --condition static --limit 20
```

A static baseline of exactly 0% is almost always an id mismatch rather than a
hard dataset. And on the free tier, searches are limited to 60 per minute per
organization: the adapter waits out a 429 and retries (`max_retries`), but a
large sweep will spend real time in that wait.

## Known limits

- **One session per request.** Relevan groups requests into sessions, and the
  natural unit here is the episode, but the adapter interface has no episode
  boundary to hang that on. Each search starts its own session.
- **The mapping is inferred, not tuned.** It is a defensible default derived
  from the dataset's own fields, not a configuration a Relevan engineer would
  write for that corpus. The System Track is where tuning it belongs — and if
  you do tune it, the mapping revision and hash are in the manifest, so the two
  runs stay distinguishable.

## Validating against the API

Request and response shapes here were checked against the live OpenAPI document
at `https://api.relevan.dev/openapi`: every route the adapter calls, every
request body it sends, and every response shape the test fake replays. The
fake in [`tests/test_relevan.py`](../tests/test_relevan.py) answers with
spec-complete payloads for that reason — a test that passes against it is
testing against the API that exists. The narrative version of the same surface
is at `https://api.relevan.dev/llms.txt`.
