# Choosing a dataset

Not every IR dataset works as a multi-shot search benchmark. Two properties
decide it, and they pull in opposite directions on most collections.

## 1. Headroom

If a single unmodified query already succeeds, there is nothing for a
multi-shot agent to demonstrate and the metric measures noise. Measure it
before committing to a dataset — it costs no tokens:

```bash
sts run --dataset <name> --condition static
```

## 2. Judgment density, or a single correct answer

This one is easy to miss and it matters more.

A multi-shot agent reformulates and surfaces documents the original pooling
never saw. Those documents are unjudged, and unjudged is scored as
non-relevant. So on a shallowly pooled dataset, an agent that successfully
explores is **systematically punished for it** — the benchmark penalizes the
exact behavior it exists to reward.

There are two ways to be safe:

- **Dense judgments.** Enough of the corpus is judged per query that a good
  new result is probably labelled. WANDS is the strong case here: a median of
  215 judged documents per query.
- **Exactly one correct answer, by construction.** In known-item retrieval
  there is one right document and it is labelled, so "unjudged means
  irrelevant" is not an assumption — it is true. TREC ToT and ArguAna work
  this way.

What does *not* work is a topical dataset with a shallow pool: many relevant
documents exist, only a couple were ever judged. `judged@k` is reported on
every run so this stays visible.

## Measured

All numbers from this harness's own BM25 adapter, single unmodified query,
`success@10`. Our nDCG@10 matches published BM25 baselines closely on every
BEIR subset tested (SCIDOCS 0.158 vs ~0.158, FiQA 0.242 vs ~0.236, SciFact
0.673 vs ~0.665, NFCorpus 0.319 vs ~0.325), which is the check that the
baseline is faithful rather than accidentally weak.

| Dataset | Queries | Corpus | Success@10 | nDCG@10 | judged@10 | Median judged/query | Verdict |
|---|---|---|---|---|---|---|---|
| `wands` | 379 | 43K | **86%** | 0.713 | 87% | 215 | Dense, but saturated |
| `beir` scifact | 300 | 5K | 81% | 0.673 | 9% | 1 | Saturated, shallow pool |
| `beir` arguana | 1,406 | 8.7K | 72% | 0.341 | 7% | 1 (by construction) | Saturated |
| `beir` nfcorpus | 323 | 3.6K | 68% | 0.319 | 29% | 16 | Shallow pool |
| `beir` scidocs | 1,000 | 25K | 50% | 0.158 | 8% | 30 | Shallow pool |
| `beir` fiqa | 648 | 57K | 47% | 0.242 | 7% | 2 | Shallow pool |
| `trec-tot` dev | 150 | 232K | **8%** | 0.065 | 1% | 1 (by construction) | **Both properties hold** |

## Recommendation: TREC Tip-of-the-Tongue

Someone has seen a film and cannot remember its name, so they describe it from
memory — at length, out of order, and partly wrong:

> Movie from the early 2000s I believe about three people living in an
> apartment but never running into each other. [...] It is a Korean or Chinese
> film I think. Art house flick… I think it won a few awards from film
> festivals like Cannes.

Why it fits:

- **Headroom.** This harness's BM25 reaches Success@10 of 8% on the 2023 dev
  set, against a published Anserini BM25 baseline of 9.3% — the gap is
  analysis and text truncation, and it confirms the baseline is faithful
  rather than accidentally weak. The best single-shot baseline in the track's
  own repo (GPT-4 query rewriting) gets 28.7%. Nearly all of the range is
  unclaimed.
- **No pooling bias.** Exactly one page is the answer. Everything else really
  is wrong, so the grader is honest at any depth.
- **Recoverable, not impossible.** GPT-4 tripling BM25 shows the gap is closed
  by better queries — which is the benchmark's whole thesis. Compare BRIGHT,
  where BM25 scores ~8.5 nDCG@10 but the gap needs domain reasoning rather
  than reformulation.
- **Reformulation is literally the task.** Real users on these forums iterate:
  try the plot, then the ending, then the actor. That is a multi-shot search
  episode occurring naturally.
- **Structured fields for the Tooling Track.** Infoboxes yield country,
  language, year, decade, director — exactly the half-remembered details the
  queries lean on ("Korean or Chinese", "early 2000s"). Filters and facets can
  be right or wrong in interesting ways.

Costs, honestly: a 950 MB download, ~80s to index 231,852 pages, and 3.4 GB of
RAM in the reference backend. Queries average 135 terms, so the index keeps
only the 50 highest-idf terms per query (`--adapter-opt max_query_terms=N`);
without that, a single search takes 15 seconds instead of 0.18. And there are
only 150 dev queries (300 with `--opt split=train+dev`), so confidence
intervals are wide — report them.

```bash
sts prepare --dataset trec-tot
sts sweep --dataset trec-tot --opt split=dev \
    --conditions static,results_only,feedback --max-shots 3
```

## When to use each

- **`trec-tot`** — the headline benchmark. Hard, unbiased, and reformulation is
  the task.
- **`wands`** — dense judgments and rich product structure make it the best
  choice for the Tooling Track, where filters and facets matter. Use
  `--difficulty hard` to get past the 86% ceiling.
- **`beir` fiqa / scidocs** — useful as secondary evidence that a result is not
  an artifact of one collection. Treat their absolute numbers with suspicion
  because of the shallow pools; trust the *differences between conditions*,
  which are affected equally.
- **`beir` scifact / arguana / nfcorpus** — too saturated to discriminate.

## Adding your own

Implement `Dataset` and pass `--dataset mypkg.module:MyDataset`. Then run the
two checks above before trusting any number from it:

```bash
sts run --dataset mypkg.module:MyDataset --condition static
```

Look at `success@10` for headroom and `judged@10` for pooling. If success is
above ~70%, use `--difficulty hard`. If `judged@10` is low *and* your dataset
has many relevant documents per query, be careful: your agent will be punished
for finding good unjudged documents, and no amount of prompting fixes that.
