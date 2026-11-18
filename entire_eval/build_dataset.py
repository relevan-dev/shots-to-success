"""Build the BEIR-style lab dataset and the train/test case files."""
import hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])          # ~/.capy/work/entire
REPO = ROOT / 'cli-0415'
OUT = ROOT / 'eval' / 'data'
BENCH = ROOT / 'pgr/public_release/benchmarks'
TEST_SHARE = 0.2
MAX_CHARS = 60000
SKIP_DIRS = ('.git/', 'vendor/', 'node_modules/')
TEXT_EXT = {'.go', '.md', '.json', '.yaml', '.yml', '.toml', '.sh', '.txt', '.mod', '.sum', '.ts', '.js', '.py', '.tmpl', '.jsonl', '.html', '.css', ''}

SYMBOL = re.compile(r'^(?:func(?:\s*\([^)]*\))?\s+([A-Za-z_]\w*)|type\s+([A-Za-z_]\w*)|(?:const|var)\s+([A-Za-z_]\w*))', re.M)

def qid(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]

def normalize_query(q):
    q = re.sub(r'\\[bBsSdDwW]', ' ', q)
    q = re.sub(r'\\(.)', r'\1', q)
    q = re.sub(r'[\^$.*+?()\[\]{}|]', ' ', q)
    return re.sub(r'\s+', ' ', q).strip()

OUT.mkdir(parents=True, exist_ok=True)
docs = []
for p in sorted(REPO.rglob('*')):
    rel = p.relative_to(REPO).as_posix()
    if not p.is_file() or any(s in rel + '/' for s in SKIP_DIRS) or p.suffix not in TEXT_EXT:
        continue
    try:
        text = p.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        continue
    syms = [next(g for g in m.groups() if g) for m in SYMBOL.finditer(text)] if p.suffix == '.go' else []
    title = rel + ' ' + ' '.join(re.split(r'[/._-]', rel)) + ' ' + ' '.join(dict.fromkeys(syms))
    docs.append({'_id': rel, 'title': title[:4000], 'text': text[:MAX_CHARS]})
doc_ids = {d['_id'] for d in docs}

pairs = [json.loads(l) for l in (ROOT / 'eval/pairs.jsonl').read_text().splitlines() if l.strip()]
pairs = [p for p in pairs if p['clicks']]
for p in pairs:
    p['clicks'] = [c for c in p['clicks'] if c in doc_ids]
    p['nq'] = normalize_query(p['query'])
pairs = [p for p in pairs if p['clicks'] and len(p['nq']) >= 2]

tasks = json.loads((BENCH / 'entireio_cli_ranking_public60/tasks_full60.json').read_text())
bench_sessions = {t['selection_meta']['session_id'] for t in tasks}
task_meta = {t['id']: t['selection_meta'] for t in tasks}

session_start = {}
for p in pairs:
    session_start[p['session_id']] = min(session_start.get(p['session_id'], p['created_at']), p['created_at'])
ordered = sorted(session_start.values())
cutoff = ordered[int(len(ordered) * (1 - TEST_SHARE))]

train, d1_test = [], []
seen_test = set()
for p in pairs:
    rec = {'qid': qid(p['nq']), 'query': p['query'], 'nq': p['nq'], 'glob': p['glob'], 'clicks': p['clicks'],
           'session_id': p['session_id'], 'created_at': p['created_at'], 'kind': p['kind']}
    if session_start[p['session_id']] >= cutoff:
        key = (p['session_id'], p['nq'])
        if key not in seen_test:
            seen_test.add(key)
            d1_test.append(rec)
    else:
        train.append(rec)

def bench_cases(subset):
    obj = json.loads((BENCH / f'entireio_cli_offline_ir_public60/{subset}/results.json').read_text())
    out = []
    for c in obj['cases']:
        nq = normalize_query(c['tool_input']['query'])
        meta = task_meta.get(c['task_id'], {})
        out.append({'case_id': c['case_id'], 'task_id': c['task_id'], 'qid': qid(nq), 'nq': nq,
                    'tool_input': c['tool_input'], 'relevant_files': c['relevant_files'],
                    'session_id': meta.get('session_id'), 'created_at': meta.get('checkpoint_created_at')})
    return out

d2 = {s: bench_cases(s) for s in ('first_search', 'pre_read')}

queries = {}
qrels = defaultdict(set)
for r in train + d1_test:
    queries[r['qid']] = r['nq']
    qrels[r['qid']].update(r['clicks'])
for s in d2.values():
    for c in s:
        queries[c['qid']] = c['nq']
        qrels[c['qid']].update(f for f in c['relevant_files'] if f in doc_ids)
        if not qrels[c['qid']]:
            qrels[c['qid']].add(docs[0]['_id'])  # keep the query in the lab; its labels come from the case file

lab = OUT / 'entire-cli'
(lab / 'qrels').mkdir(parents=True, exist_ok=True)
(lab / 'corpus.jsonl').write_text('\n'.join(json.dumps(d) for d in docs) + '\n')
(lab / 'queries.jsonl').write_text('\n'.join(json.dumps({'_id': k, 'text': v}) for k, v in queries.items()) + '\n')
(lab / 'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\n' + ''.join(f'{k}\t{d}\t1\n' for k, ds in qrels.items() for d in sorted(ds)))
(OUT / 'train.jsonl').write_text('\n'.join(json.dumps(r) for r in train) + '\n')
(OUT / 'd1_test.jsonl').write_text('\n'.join(json.dumps(r) for r in d1_test) + '\n')
(OUT / 'd2_cases.json').write_text(json.dumps(d2))
print(json.dumps({'docs': len(docs), 'pairs': len(pairs), 'cutoff': cutoff, 'train': len(train),
                  'train_sessions': len({r['session_id'] for r in train}), 'd1_test': len(d1_test),
                  'd1_test_sessions': len({r['session_id'] for r in d1_test}), 'unique_queries': len(queries),
                  'd2_first': len(d2['first_search']), 'd2_pre_read': len(d2['pre_read']),
                  'bench_sessions_in_train': sum(r['session_id'] in bench_sessions for r in train)}))
