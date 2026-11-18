# /// script
# requires-python = ">=3.12"
# dependencies = ["lightgbm==4.6.0", "numpy>=2"]
# ///
"""Score ripgrep, pgr, Relevan, and Relevan + rerank:ltr trained on checkpoint clicks.

    uv run evaluate.py <entire dir> <monorepo dir> <out.json>
"""
import importlib.util, json, math, random, sys, types
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT, MONO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DATA = ROOT / 'eval/data'
REPO = ROOT / 'cli-0415'
K = 10

spec = importlib.util.spec_from_file_location('train_ltr', MONO / 'packages/scripts/src/ltr/train_ltr.py')
train_ltr = importlib.util.module_from_spec(spec); spec.loader.exec_module(train_ltr)
LTR_ARGS = types.SimpleNamespace(trees=100, leaves=15, learning_rate=0.05)

sys.path.insert(0, str(ROOT / 'pgr/eval/v2'))
from backends import baseline as rg_backend, pgr_backend
sys.path.insert(0, str(ROOT / 'pgr/public_release/benchmarks/entireio_cli_offline_ir_public60'))
import run_offline_ir as entire_ir  # Entire's own path parser and relevance rule

lines = (DATA / 'collected.jsonl').read_text().splitlines()
header = json.loads(lines[0])
cands = {q['qid']: q for q in map(json.loads, lines[1:])}
train = [json.loads(l) for l in (DATA / 'train.jsonl').read_text().splitlines() if l]
d1 = [json.loads(l) for l in (DATA / 'd1_test.jsonl').read_text().splitlines() if l]
d2 = json.loads((DATA / 'd2_cases.json').read_text())
bench_sessions = {c['session_id'] for s in d2.values() for c in s}

def first_rank(paths, relevant):
    for i, p in enumerate(paths[:K], 1):
        if entire_ir._is_relevant(p, tuple(relevant)):
            return i
    return None

def metrics(rank):
    return {'mrr': 0.0 if rank is None else 1 / rank, 'hit1': rank == 1,
            'hit3': rank is not None and rank <= 3, 'hit5': rank is not None and rank <= 5,
            'hit10': rank is not None}

def ltr_model(instances):
    """Train rerank:ltr on click labels aggregated per query: 2 = clicked in half or more of its searches, 1 = clicked, 0 = shown, not clicked."""
    seen, clicked = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for r in instances:
        seen[r['qid']] += 1
        for c in r['clicks']:
            clicked[r['qid']][c] += 1
    queries = []
    for qid, n in seen.items():
        q = cands.get(qid)
        if not q or not q['candidates']:
            continue
        cs = [dict(c, label=(2 if clicked[qid][c['id']] * 2 >= n else 1) if clicked[qid][c['id']] else 0) for c in q['candidates']]
        if any(c['label'] for c in cs):
            queries.append({'candidates': cs})
    if len(queries) < 20:
        return None, len(queries)
    return train_ltr.train(queries, LTR_ARGS), len(queries)

def relevan_order(qid, model=None):
    q = cands.get(qid)
    if not q or not q['candidates']:
        return []
    ids = [c['id'] for c in q['candidates']]
    if model is None:
        return ids[:K]
    scores = train_ltr.predict(model, q)
    return [ids[i] for i in np.argsort(-scores, kind='stable')][:K]

def grep_orders(query, glob):
    out = {}
    for name, backend in (('ripgrep', rg_backend), ('pgr', pgr_backend)):
        text = backend.search_code(repo_root=str(REPO), query=query, path_glob=glob or '', max_files=K)
        out[name] = entire_ir._extract_paths(text)
    return out

def score_cases(cases, relevant_of, models_for):
    rows = []
    for c in cases:
        rel = relevant_of(c)
        q = c.get('tool_input', {}).get('query', c.get('query'))
        glob = c.get('tool_input', {}).get('path_glob', c.get('glob', ''))
        orders = grep_orders(q, glob)
        orders['relevan'] = relevan_order(c['qid'])
        orders['relevan_ltr'] = relevan_order(c['qid'], models_for(c))
        rows.append({'case': c.get('case_id') or c['qid'], 'group': c.get('task_id') or c['session_id'],
                     **{s: metrics(first_rank(o, rel)) for s, o in orders.items()}})
    return rows

SYSTEMS = ['ripgrep', 'pgr', 'relevan', 'relevan_ltr']
METRICS = ['mrr', 'hit1', 'hit3', 'hit5', 'hit10']

def summarize(rows):
    s = {sys_: {m: float(np.mean([r[sys_][m] for r in rows])) for m in METRICS} for sys_ in SYSTEMS}
    groups = defaultdict(list)
    for r in rows:
        groups[r['group']].append(r)
    keys = list(groups)
    rng = random.Random(42)
    paired = {}
    for a, b in [('relevan_ltr', 'pgr'), ('relevan_ltr', 'relevan'), ('relevan', 'pgr'), ('pgr', 'ripgrep')]:
        for m in ('mrr', 'hit1', 'hit3'):
            diffs = []
            for _ in range(2000):
                sample = [r for k in (rng.choice(keys) for _ in keys) for r in groups[k]]
                diffs.append(np.mean([float(r[a][m]) - float(r[b][m]) for r in sample]))
            point = float(np.mean([float(r[a][m]) - float(r[b][m]) for r in rows]))
            paired[f'{a}-{b}:{m}'] = [point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    return {'n': len(rows), 'groups': len(keys), 'overall': s, 'paired_cluster_bootstrap': paired}

results = {}

# D1: time split. Train on sessions before the cutoff, test on sessions after it.
model_d1, n_train_q = ltr_model(train)
rows = score_cases(d1, lambda c: c['clicks'], lambda c: model_d1)
results['d1_time_split'] = {**summarize(rows), 'train_instances': len(train), 'train_queries': n_train_q}

# D2: Entire's public offline benchmark cases. Train only on earlier sessions, never the benchmark sessions.
cache = {}
def strict_model(c):
    key = c['created_at']
    if key not in cache:
        cache[key] = ltr_model([r for r in train + d1 if r['session_id'] not in bench_sessions and r['created_at'] < key])[0]
    return cache[key]
all_nonbench = ltr_model([r for r in train + d1 if r['session_id'] not in bench_sessions])[0]
for subset, cases in d2.items():
    rel = lambda c: c['relevant_files']
    results[f'd2_{subset}_strict'] = summarize(score_cases(cases, rel, strict_model))
    results[f'd2_{subset}_all_history'] = summarize(score_cases(cases, rel, lambda c: all_nonbench))

pgr_backend.cleanup() if hasattr(pgr_backend, 'cleanup') else None
OUT.write_text(json.dumps(results, indent=2))
for name, r in results.items():
    print(f"\n## {name}  n={r['n']} groups={r['groups']}")
    for s in SYSTEMS:
        o = r['overall'][s]
        print(f"{s:12s} MRR {o['mrr']:.3f}  H@1 {100*o['hit1']:.1f}  H@3 {100*o['hit3']:.1f}  H@5 {100*o['hit5']:.1f}  H@10 {100*o['hit10']:.1f}")
    for k, v in r['paired_cluster_bootstrap'].items():
        print(f"  {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
