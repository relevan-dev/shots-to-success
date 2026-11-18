# Entire checkpoint ranking eval

This directory replays the search calls in Entire's public `entireio/cli` checkpoints. It scores four rankers on the files the agent read next:

- `ripgrep`: Entire's baseline backend, raw `rg` order.
- `pgr`: Entire's rule-based ranker, built from `entireio/pgr`.
- `relevan`: Relevan's production mapping and query pipeline, hybrid retrieval, no feedback.
- `relevan_ltr`: the same retrieval, then the shipped `rerank:ltr` LambdaMART model trained on clicks from the training split.

A click is a `Read` of a file within six tool calls after a `Grep` or a shell `rg`/`grep`, before the next search.

## Test sets

- `d1_time_split`: 1,252 search-then-read cases from the 106 latest sessions. Training uses the earlier 422 sessions. The cutoff is the session start 2026-03-24T01:32Z.
- `d2_first_search` and `d2_pre_read`: Entire's own offline benchmark cases (`entireio_cli_offline_ir_public60`), with Entire's labels and Entire's relevance rule. The `strict` variant trains only on sessions that started before the task checkpoint, and never on a benchmark session. The `all_history` variant also trains on later sessions.

## Run

The scripts need clones of `entireio/pgr` and `entireio/cli` in one directory, `$E`, and a `relevan-dev/monorepo` checkout.

```bash
git -C $E/cli worktree add ../cli-0415 0890284d33ff10c8ededd3938797618f5e7f8248
(cd $E/pgr && cargo build --release)
mkdir -p $E/eval && cp *.py $E/eval/
python3 -I $E/eval/extract_pairs.py $E/pgr/public_release/data/entireio_cli_checkpoints_2026_04_15/checkpoint_transcripts.jsonl.gz $E/cli-0415 $E/eval/pairs.jsonl
python3 -I $E/eval/build_dataset.py $E
uv run $E/eval/embed.py $E/eval/data/entire-cli $E/eval/data/vec    # bge-small-en-v1.5, then zero-pad to 1,024 columns as vec1024-*.npy
docker run -d -p 127.0.0.1:9201:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true opensearchproject/opensearch:3.5.0
(cd monorepo/packages/scripts && pnpm ltr:collect --dataset entire-cli --dir $E/eval/data/entire-cli --vectors $E/eval/data/vec1024 --out $E/eval/data/collected.jsonl --fill)
uv run $E/eval/evaluate.py $E monorepo results.json
```

`label_collected.py` writes the training-split click labels into a copy of `collected.jsonl`. `train_ltr.py --labels clicks` on that copy gives `entire-cli-ltr.json`, a `rerank:ltr` config for `PUT /features/rerank:ltr`.

## Results

`results.json` holds every number. Each cell is the mean over cases; the top 10 files count.

| Test set | n | Metric | ripgrep | pgr | Relevan | Relevan + clicks |
|---|---:|---|---:|---:|---:|---:|
| d1_time_split | 1,252 | MRR | 0.305 | 0.387 | 0.321 | 0.397 |
| | | Hit@1 | 22.8% | 32.7% | 25.6% | 33.2% |
| | | Hit@3 | 35.2% | 43.0% | 36.3% | 45.0% |
| d2_first_search_strict | 50 | MRR | 0.275 | 0.393 | 0.412 | 0.460 |
| | | Hit@1 | 18.0% | 32.0% | 32.0% | 34.0% |
| | | Hit@3 | 32.0% | 42.0% | 46.0% | 54.0% |
| d2_pre_read_strict | 132 | MRR | 0.172 | 0.253 | 0.321 | 0.366 |
| | | Hit@1 | 11.4% | 20.5% | 25.8% | 27.3% |
| | | Hit@3 | 19.7% | 28.0% | 35.6% | 39.4% |

The 95% intervals come from a bootstrap over sessions (d1) or tasks (d2).

- Clicks improve Relevan on d1: MRR +0.076 [+0.061, +0.091].
- On d1, Relevan + clicks ties `pgr`: MRR +0.010 [-0.008, +0.028]. Relevan without feedback loses to `pgr`: -0.066 [-0.089, -0.042].
- On d2_pre_read_strict, Relevan + clicks beats `pgr`: MRR +0.113 [+0.037, +0.199], Hit@3 +11.4 points [+2.9, +20.7].
- On d2_first_search_strict, Relevan + clicks leads `pgr` by MRR +0.067, but the interval [-0.024, +0.161] includes zero.

## Limits

- A read is a weak label. The agent may read a file it then finds useless.
- d1 queries are `Grep` regex patterns. Relevan gets them through a small adapter that turns regex syntax into words, so exact regex structure is lost.
- `rerank:ltr` reorders only the top 30 hybrid hits. Its features have no file identity and no test-or-source flag.
- The lab runs Relevan's production query code on a local OpenSearch with `bge-small-en-v1.5` vectors. Production uses a different embedding model.
- `rg` order is not deterministic. Our `ripgrep` numbers on d2 differ from Entire's published ones (MRR 0.275 against 0.318). Our `pgr` numbers are close (0.393 against 0.405).

## End-to-end pilot on Qwen

`agent_qwen.py` runs Entire's public60 agent loop (`pgr/eval/v2` tools and system prompt, 12 turns) on an OpenAI-compatible endpoint. The pilot used Qwen3-Coder-30B-A3B-Instruct-FP8 on vLLM. Both arms use `pgr` for `read_code`, `find_files`, and `list_dir`. Only `search_code` differs:

- `pgr`: Entire's `pgr`.
- `relevan`: the candidates are `rg`'s matches plus Relevan's top 30. Relevan's order comes first. Each file shows at most three match lines. `entire-search-server.ts` serves the lab search with one click-trained `rerank:ltr` model per task. `train_task_models.py` trains each model only on sessions that started before its task.

The suite has no answer key. "Edit target reached" means the agent opened a file that the original checkpoint edited. 57 of the 60 tasks have one. `score_pilot.py` scores a run directory.

Two repeats, 120 paired runs for each arm (`pilot/`):

| Metric | pgr | Relevan + clicks | Difference [95% CI] |
|---|---:|---:|---|
| Edit target reached | 39.2% | 45.8% | +6.7 pts [+0.8, +13.3] |
| Opened a file Entire's Claude run opened | 70.0% | 75.8% | +5.8 pts [-0.8, +12.5] |
| Answered within 12 turns | 35.0% | 40.8% | +5.8 pts [-1.7, +13.3] |
| Tool calls per run | 10.40 | 10.37 | -0.03 [-0.44, +0.41] |
| search_code calls per run | 2.77 | 2.92 | +0.15 [-0.18, +0.48] |
| Tokens per run | 45,573 | 47,328 | +1,756 [-2,078, +6,361] |
| Wall clock per run | 14.9 s | 14.3 s | -0.6 s [-2.1, +0.8] |

Two earlier runs of the Relevan arm had harness defects, and the table leaves them out. The first printed rg context blocks, about seven times the size of `pgr`'s output, and echoed malformed tool arguments back to vLLM. The second used `rg` for the other tools. Both reached the edit target more often than `pgr` (25 and 26 of 60, against 22).
