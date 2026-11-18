# /// script
# requires-python = ">=3.12"
# dependencies = ["lightgbm==4.6.0", "numpy>=2"]
# ///
"""Write one rerank:ltr config per pilot task, trained only on non-benchmark sessions that started before the task."""
import importlib.util, json, sys, types, datetime
from pathlib import Path

ROOT, MONO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('train_ltr', MONO / 'packages/scripts/src/ltr/train_ltr.py')
train_ltr = importlib.util.module_from_spec(spec); spec.loader.exec_module(train_ltr)
from collections import defaultdict

DATA = ROOT / 'eval/data'
lines = (DATA / 'collected.jsonl').read_text().splitlines()
header = json.loads(lines[0])
cands = {q['qid']: q for q in map(json.loads, lines[1:])}
rows = [json.loads(l) for f in ('train.jsonl', 'd1_test.jsonl') for l in (DATA / f).read_text().splitlines() if l]
tasks = json.loads((ROOT / 'pgr/public_release/benchmarks/entireio_cli_ranking_public60/tasks_full60.json').read_text())
bench = {t['selection_meta']['session_id'] for t in tasks}
OUT.mkdir(parents=True, exist_ok=True)
args = types.SimpleNamespace(trees=100, leaves=15, learning_rate=0.05)
summary = {}
for t in tasks:
    cutoff = t['selection_meta']['checkpoint_created_at']
    inst = [r for r in rows if r['session_id'] not in bench and r['created_at'] < cutoff]
    seen, clicked = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for r in inst:
        seen[r['qid']] += 1
        for c in r['clicks']:
            clicked[r['qid']][c] += 1
    queries = []
    for qid, n in seen.items():
        q = cands.get(qid)
        if not q or not q['candidates']:
            continue
        cs = [dict(c, label=(2 if 2 * clicked[qid][c['id']] >= n else 1) if clicked[qid][c['id']] else 0) for c in q['candidates']]
        if any(c['label'] for c in cs):
            queries.append({'candidates': cs})
    summary[t['id']] = len(queries)
    if len(queries) < 20:
        continue
    model = train_ltr.train(queries, args)
    config = {'fields': header['roles'], 'topK': header['topK'],
              'model': {'kind': 'gbdt', 'featureNames': header['features'],
                        'trees': [train_ltr.flat_tree(x['tree_structure']) for x in model.dump_model()['tree_info']]},
              'training': {'labelSource': 'clicks', 'queries': len(queries), 'ndcgBefore': 0, 'ndcgAfter': 0,
                           'vector': True, 'trainedAt': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')}}
    (OUT / f"{t['id']}.json").write_text(json.dumps(config))
print(json.dumps({'models': sum(v >= 20 for v in summary.values()), 'tasks': len(summary),
                  'min_queries': min(summary.values()), 'median_queries': sorted(summary.values())[30]}))
