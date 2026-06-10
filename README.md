# **Shots to Success Benchmark**

Large language models often struggle to use search interfaces reliably. Query syntax can be difficult to generate, filters may be applied incorrectly, and agents often cannot tell why a search failed. Did the document not exist? Did the query use the wrong terms? Was the search too broad, too narrow, or filtered incorrectly?

The Shots to Success Benchmark measures how many search attempts an LLM agent needs to retrieve relevant documents. Existing benchmarks often measure whether an agent eventually answers correctly. This benchmark measures whether a search system helps an agent find relevant results faster, recover from failed searches, and stop once good results have been found.

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

