# **Shots to Success Benchmark**

Large language models often struggle to use search interfaces reliably. Query syntax can be difficult to generate, filters may be applied incorrectly, and agents often cannot tell why a search failed. Did the document not exist? Did the query use the wrong terms? Was the search too broad, too narrow, or filtered incorrectly?

The Shots to Success Benchmark measures how many search attempts an LLM agent needs to retrieve relevant documents. Existing benchmarks often measure whether an agent eventually answers correctly. This benchmark measures whether a search system helps an agent find relevant results faster, recover from failed searches, and stop once good results have been found.


## **Quick start**

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

sts prepare --dataset trec-tot                          # ~950 MB
sts sweep --dataset trec-tot --limit 25                 # all conditions, then compare
```

`sts sweep` builds the index once, pins one episode set, runs every condition
against it, and prints a comparison table. See **Using the harness** below.

Three datasets ship: `trec-tot` (recommended), `wands`, and `beir` (15+
subsets). Which to use, and why it matters more than it looks, is in
[docs/DATASETS.md](docs/DATASETS.md).

## 

## **Success Criteria**

Each episode has two forms of success criteria: one visible to the agent and one hidden from the agent.

The agent is given an information need and a fixed search budget. After each search attempt, it must decide whether to stop or retry. The agent should stop when it believes the current top results contain documents that directly satisfy the information need.

The evaluator uses hidden relevance judgments from the underlying IR dataset. An episode reaches **Success@10** when at least one judged-relevant document appears in the top 10 results. These judgments are never shown to the agent and are used only for scoring.

Existing IR datasets can be adapted into this format. The original query becomes the information need, the corpus becomes the searchable collection, and the existing relevance judgments become the hidden evaluator labels.

## 

## **Testing**

Traditional information retrieval (IR) benchmarks measure a static query against a ranked result set. That is useful, but it does not match how LLM agents interact with search. Agents can inspect results, reason about failure modes, rewrite queries, apply filters, and retry. Relevan’s core claim is that search engines should expose feedback that helps agents make better retrieval decisions.

We will adapt traditional IR datasets into **feedback-guided retrieval episodes**.

Each episode follows this loop:

1. The agent receives an information need adapted from a benchmark query.  
2. The agent executes a search using a fixed search tool contract.  
3. The search engine returns results and, depending on the test condition, retrieval feedback.  
4. The agent returns a decision: \`stop\` or \`retry\`.  
5. If the agent retries, it submits a modified query or search action.  
6. If the agent stops, the current ranked result set becomes the final submitted result set.  
7. If the agent exhausts the search budget, the final attempt is submitted automatically.  
8. The evaluator scores both the final submitted result set and the full episode trajectory against hidden relevance judgments.

The agent will have a fixed search budget, such as three total attempts. The benchmark does not automatically stop when a relevant document first appears, because the agent does not know the hidden relevance judgments. Instead, the agent must decide when to stop based only on the information need, returned results, and any feedback available in the test condition.

## 

## **Metrics**

The benchmark is designed around one primary question:

How many search attempts does an LLM agent need to retrieve relevant documents?

The primary metric is:

* **Shots to Success@10:** the number of search attempts required before a judged-relevant document first appears in the top 10 results.

We will also track four guardrail metrics:

* **Success@10:** the percentage of episodes where the agent retrieves at least one judged-relevant document in the top 10 within the search budget.  
* **Recovery rate:** among episodes that fail on the first attempt, the percentage that succeed on a later attempt.  
* **Bad retry rate:** the percentage of retries that make retrieval quality worse than the previous attempt.  
* **Oversearch rate:** the percentage of successful episodes where the agent continues searching after the result set first satisfies Success@10.

The benchmark harness may also record traditional IR metrics such as nDCG@10, Recall@10, and MRR@10 for deeper analysis. These are useful diagnostics, but they are not the primary benchmark outputs.

We will initially evaluate the Feedback Track using three retrieval conditions: 

1. **Static baseline:** the original benchmark query is executed once with no retries.  
2. **Results-only agent:** the agent sees normal search results and snippets, then may stop or retry.  
3. **Feedback-guided agent:** the agent sees normal search results plus retrieval assistance and diagnostics such as matched terms, missing terms, filter behavior, and score contributors.

Future tracks may evaluate agent-facing tooling and full system improvements, where search tools, index enrichment, ranking configuration, or retrieval architecture are allowed to vary.

## 

## **Controlled Variables**

The benchmark is designed to isolate the effect of retrieval feedback. To make comparisons fair, each test condition must use the same model, agent prompt, information need, searchable corpus, relevance judgments, search budget, and search tool contract.

The only intended difference between conditions is the feedback available to the agent after each search attempt.

The following variables must remain constant across comparable runs:

* **Model:** each condition must use the same LLM and model version.  
* **Agent prompt:** each condition must use the same task instructions, stopping rule, output format, and search budget.  
* **Information need:** each condition must receive the same benchmark episode.  
* **Corpus and index:** each condition must search the same document collection with the same indexed fields and ranking configuration, unless ranking configuration is the subject of the experiment.  
* **Hidden relevance judgments:** each condition must be scored against the same evaluator labels.  
* **Search budget:** each condition must receive the same maximum number of search attempts.  
* **Search tool contract:** each condition must have the same available search actions. Feedback-guided conditions may receive additional diagnostic information, but should not receive additional hidden labels or extra search capabilities.  
* **Result rendering:** each condition should see the same result fields, snippets, ordering format, and metadata, except for the feedback explicitly being tested.  
* **Sampling configuration:** each condition should use the same temperature, decoding parameters, and retry policy.

This ensures that differences in Shots to Success@10, Recovery Rate, Bad Retry Rate, and Oversearch Rate can be attributed to the presence or absence of retrieval feedback rather than unrelated changes in the agent, dataset, prompt, or evaluator.

## 

## **Controlled Variables and Benchmark Tracks**

The benchmark is designed to compare retrieval systems fairly by isolating what changes between test conditions. Each benchmark run must declare which variables are fixed and which variable is being tested.

Across all comparable runs, the following must remain constant:

* **Information need:** each condition must receive the same benchmark episode.  
* **Hidden relevance judgments:** each condition must be scored against the same evaluator labels.  
* **Model:** each condition must use the same LLM and model version.  
* **Agent prompt:** each condition must use the same task instructions, stopping rule, output format, and search budget.  
* **Search budget:** each condition must receive the same maximum number of search attempts.  
* **Sampling configuration:** each condition should use the same temperature, decoding parameters, and retry policy.  
* **Result rendering:** each condition should see the same result fields, snippets, ordering format, and metadata, except for any feedback explicitly being tested.

Other variables may either be fixed or intentionally varied depending on the benchmark track.

### **Feedback Track**

The Feedback Track isolates the effect of retrieval feedback. In this track, the corpus, index, ranking configuration, search tool contract, and result rendering remain fixed. The only intended difference is whether the agent receives retrieval diagnostics after each search attempt.

This track answers:

Given the same search system and the same agent, does feedback help the agent find relevant documents in fewer attempts?

### **Tooling Track**

The Tooling Track measures whether better agent-facing search tools reduce shots to success. In this track, systems may expose different search actions or tool contracts, such as keyword search, structured filters, semantic search, faceting, query expansion, explainability, or diagnostic endpoints.

This track answers:

Given the same model and information need, do better search primitives help the agent retrieve relevant documents in fewer attempts?

### **System Track**

The System Track measures end-to-end retrieval performance for agentic search. In this track, systems may vary indexing strategy, enrichment, ranking configuration, retrieval tools, feedback, and diagnostics. This allows systems to use techniques such as entity extraction, synonym expansion, generated metadata, embeddings, taxonomy normalization, reranking, or other relevance improvements.

This track answers:

Which retrieval system helps an agent reach relevant documents fastest under the same benchmark episodes, model, prompt, search budget, and relevance judgments?



---

# **Using the harness**

## **Install**

```bash
pip install -e .              # or: uv pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

The only runtime dependency is the `anthropic` SDK. The reference search
backend is pure Python — no search engine to stand up.

## **Commands**

| Command | What it does |
|---|---|
| `sts list` | Built-in datasets, adapters, and conditions |
| `sts prepare --dataset trec-tot` | Download source files and report corpus size |
| `sts episodes --difficulty hard -o set.json` | Materialize an episode set for pinning |
| `sts run --condition feedback` | Run one condition |
| `sts sweep --conditions static,results_only,feedback` | Run several against one shared episode set, then compare |
| `sts report runs/a runs/b` | Comparison table, with comparability warnings |
| `sts inspect runs/a --failures` | Per-episode trajectories, for reading what actually happened |
| `sts regrade runs/a --k 3` | Re-score a finished run — no model calls |

## **A run**

```bash
sts sweep \
    --dataset wands --adapter bm25 \
    --conditions static,results_only,feedback \
    --difficulty hard --max-shots 3 --k 10 \
    --model claude-opus-5 --effort high
```

Each condition writes `runs/<run_id>/`:

```
manifest.json      every controlled variable, the exact system prompt and its
                   hash, the episode set hash, dataset and adapter provenance
episodes.jsonl     one graded episode per line, written as it finishes
episode_set.json   the query ids scored
summary.json       aggregate metrics
```

Runs are resumable: rerunning the same `--run-id` skips episodes already
recorded. Episodes are graded from the recorded result sets, so `sts regrade`
can rescore an entire run at a different `k` or relevance threshold without
spending a single token.

## **How it fits together**

```
Dataset  ──queries──────────────────────────┐
   │                                        │
   ├──corpus────▶  SearchAdapter  ──tools──▶ Agent loop ──▶ Episode
   │                                                          │
   └──judgments (hidden) ──────────────────▶  Grader  ◀───────┘
```

The agent loop is never handed judgments and cannot import them — a test
asserts this against the module's imports, because it is the property that
makes the hidden labels actually hidden. Everything the model saw is recorded,
so grading is a pure function of the transcript.

## **Adapters are the extension point**

A search system enters the benchmark by exposing tools. Each tool declares
whether calling it spends a shot:

- **`RETRIEVAL`** — returns a ranked, gradable result set. Costs one shot.
- **`AUXILIARY`** — autocomplete, facets, schema, score explanation. Free.

That split is deliberate. If a provider exposes an autocomplete endpoint that
helps an agent build a better query, charging a shot for it would penalize the
system for offering it. So auxiliary calls are recorded but do not count.

```bash
sts run --adapter mypkg.adapters:MyAdapter --adapter-opt base_url=https://...
```

No registration, no changes to this repository. Full guide in
[docs/ADAPTERS.md](docs/ADAPTERS.md); a hosted-service template with an
autocomplete endpoint is in [examples/http_adapter.py](examples/http_adapter.py).

The bundled `bm25` adapter is the reference implementation and exposes the full
surface: `search` (with filters), `autocomplete`, `facets`, `describe_index`,
and `explain_result`.

A second adapter, `relevan`, runs the benchmark against the hosted search API at
`api.relevan.dev` — it creates an index, derives a mapping from the dataset's
own fields, ingests the corpus, and exposes `search` (filters and per-term
boosts), the index's generated skill doc, and per-document score explanation:

```bash
export RELEVAN_API_KEY=...
sts run --dataset trec-tot --adapter relevan --condition feedback
```

Ingest happens once per dataset; later runs reuse the index. Details, options,
and why result feedback is off by default are in [docs/RELEVAN.md](docs/RELEVAN.md).

## **Conditions**

| Condition | Model in loop | Diagnostics | Auxiliary tools | Track |
|---|---|---|---|---|
| `static` | no | — | — | baseline |
| `results_only` | yes | no | no | Feedback |
| `feedback` | yes | **yes** | no | Feedback |
| `tooling` | yes | yes | **yes** | Tooling |

`results_only` and `feedback` are byte-identical in prompt and tool contract —
verified by a test — so the difference between them is attributable to
diagnostics alone. `tooling` widens the tool contract, which is a different
track and a different question.

## **Metrics**

Primary:

- **`shots_to_success`** — mean attempts until a judged-relevant document first
  appeared in the top *k*, over episodes that ever succeeded. Always read it
  next to `success@k`: a system that only solves easy queries gets a
  flattering mean.
- **`shots_to_success_censored`** — the same, with failures charged
  `max_shots + 1`. One scalar that orders systems without hiding failures.

Guardrails: `success@k` (on the **submitted** set), `success_any@k` (any
attempt), `recovery_rate`, `bad_retry_rate` (retries that lowered nDCG@k),
`oversearch_rate` (successful episodes that kept searching after the need was
already met).

Diagnostics: `ndcg@k`, `recall@k`, `mrr@k`, `stop_precision` (of the documents
the agent named as satisfying the need, how many were), `confidence_accuracy`
(was the agent right that it had found something), and `judged@k`.

`judged@k` deserves attention. Unjudged documents are scored as non-relevant,
the standard pooled-judgment assumption; `judged@k` is how visible the cost of
that assumption stays. On WANDS it runs around 87%, so roughly one result in
eight is scored as a miss only because nobody labelled it.

## **Comparability**

The manifest records every controlled variable, the exact system prompt and its
hash, and a hash of the episode set. `sts report` compares those across runs and
prints warnings when they drift:

```
Comparability warnings:
  ! effort differs: "high", "low"
  ! runs scored different episode sets: bf0dd8619169, 91ba22c40e17
```

Two notes on reproducibility. Sampling is not a lever on current models —
`temperature` and `top_p` are rejected by Opus 5 — so runs are pinned by model,
`effort`, prompt version, budget, and episode set instead, and repeated runs
will still vary. And `--difficulty hard` is computed from *the adapter's own*
baseline, so two different backends produce different episode sets; pin one with
`--episode-set` when comparing across backends.

## **Choosing a dataset (this matters more than it looks)**

Two properties decide whether a dataset can measure multi-shot search at all,
and most IR datasets fail one of them. Full analysis and the measured table are
in [docs/DATASETS.md](docs/DATASETS.md); the short version:

**Headroom.** On WANDS, a single unmodified BM25 query already reaches
**Success@10 of 86%**. There is almost nothing for a multi-shot agent to
demonstrate. Always measure this first — it costs no tokens:

```bash
sts run --dataset <name> --condition static
```

**Judgment density, or a single correct answer.** A multi-shot agent surfaces
documents the original pooling never saw, and unjudged is scored as
non-relevant — so on a shallowly pooled dataset the benchmark *punishes the
exact behavior it exists to reward*. WANDS is safe because it is densely
judged (median 215 judged documents per query). Most BEIR datasets are not:
FiQA and SciFact judge a median of 1-2 documents per query. `judged@k` is
reported on every run so this stays visible.

Known-item retrieval escapes the problem entirely: if exactly one document is
correct and it is labelled, "unjudged means irrelevant" is not an assumption,
it is true.

| Dataset | Success@10 (single shot) | Median judged/query | Verdict |
|---|---|---|---|
| `wands` | 86% | 215 | Dense, but saturated |
| `beir` fiqa | 47% | 2 | Shallow pool |
| `beir` scidocs | 50% | 30 | Shallow pool |
| **`trec-tot`** | **8%** | 1 (by construction) | **Both properties hold** |

**TREC Tip-of-the-Tongue** is the recommended default: someone describes a film
they cannot name, vaguely and partly wrongly, and exactly one of 232k Wikipedia
pages is the answer. BM25 gets 8%; GPT-4 query rewriting gets 28.7% in the
track's own baselines — hard, but demonstrably recoverable by better queries,
which is precisely this benchmark's thesis.

For datasets that are saturated but worth keeping (WANDS has product structure
that the Tooling Track needs), build a hard subset — the queries the adapter's
own single-shot baseline fails:

```bash
sts episodes --dataset wands --difficulty hard -o data/wands-hard.json
# 53 of 379 queries fail the single-shot baseline
```

On that subset the static baseline scores 0% by construction, so everything a
multi-shot agent achieves is attributable to multi-shot search. Reporting on a
hard subset is a magnifying glass, not a substitute for the full set — report
both, and always name which set a number came from.

## **Adding a dataset**

Implement `Dataset` (queries, corpus, hidden judgments) and pass
`--dataset mypkg.module:MyDataset`. Then run the two checks above before
trusting a number from it. One method is worth attention:
`relevance_rubric()` states what "satisfies the information need" means for
your collection. It is shown to the agent verbatim and is identical across
conditions — without it the agent optimizes a different bar than the grader
scores.

## **Development**

```bash
uv pip install -e ".[dev]"
python -m pytest          # 84 tests, no network or API key needed
```

The suite runs the whole loop against a scripted stand-in for the API, so
budget enforcement, stopping behavior, and condition wiring are all covered
offline.
