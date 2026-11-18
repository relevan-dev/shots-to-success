"""Score the Qwen pilot runs: target files reached, steps, tokens, wall clock, and tool-call reliability."""
import gzip, json, random, statistics, sys
from pathlib import Path

ROOT, RUNS = Path(sys.argv[1]), Path(sys.argv[2])
REPO = ROOT / 'cli-0415'
B = ROOT / 'pgr/public_release/benchmarks'
tasks = json.loads((B / 'entireio_cli_ranking_public60/tasks_full60.json').read_text())
ids = {t['id'] for t in tasks}

files = [p.relative_to(REPO).as_posix() for p in REPO.rglob('*') if p.is_file() and '/.git/' not in '/' + p.relative_to(REPO).as_posix()]
def resolve(path):
    path = (path or '').strip().lstrip('./')
    parts = path.split('/')
    for i in range(len(parts)):
        suf = '/'.join(parts[i:])
        hits = [f for f in files if f == suf or f.endswith('/' + suf)]
        if len(hits) == 1 and (len(parts) - i >= 2 or i == len(parts) - 1):
            return hits[0]
    return None

edits = {}
with gzip.open(ROOT / 'pgr/public_release/data/entireio_cli_checkpoints_2026_04_15/checkpoint_transcripts.jsonl.gz', 'rt') as fh:
    for line in fh:
        ck = json.loads(line)
        if ck['checkpoint_id'] not in ids:
            continue
        s = set()
        for k in ck.get('transcript_stripped') or {}:
            for l in ck['transcript_stripped'][k].splitlines():
                try:
                    ev = json.loads(l)
                except Exception:
                    continue
                content = (ev.get('message') or {}).get('content') if ev.get('type') == 'assistant' else None
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get('type') == 'tool_use' and b.get('name') in ('Edit', 'Write', 'MultiEdit', 'NotebookEdit'):
                        f = resolve((b.get('input') or {}).get('file_path', ''))
                        if f:
                            s.add(f)
        edits[ck['checkpoint_id']] = s

claude_reads = {}
for f in (B / 'entireio_cli_ranking_public60/full60/results/baseline').glob('*.json'):
    o = json.loads(f.read_text())
    claude_reads[o['task_id']] = {r for r in (resolve(tc['tool_input'].get('path', '')) for tc in o['tool_calls'] if tc['tool_name'] == 'read_code') if r}

def reached(run, targets):
    for i, tc in enumerate(run['tool_calls'], 1):
        if tc['tool_name'] == 'read_code' and resolve(tc['tool_input'].get('path', '')) in targets:
            return i
    return None

rows = {}
for cond in ('pgr', 'relevan'):
    for t in tasks:
        r = json.loads((RUNS / cond / f"{t['id']}.json").read_text())
        e, c = edits.get(t['id'], set()), claude_reads.get(t['id'], set())
        re_, rc = reached(r, e) if e else None, reached(r, c) if c else None
        calls = r['tool_calls']
        rows.setdefault(t['id'], {})[cond] = {
            'has_edit_target': bool(e), 'edit_hit': re_ is not None, 'claude_hit': rc is not None,
            'calls_to_edit_target': re_, 'calls_to_claude_target': rc,
            'tool_calls': len(calls), 'search_calls': sum(x['tool_name'] == 'search_code' for x in calls),
            'read_calls': sum(x['tool_name'] == 'read_code' for x in calls), 'turns': r['turns'],
            'answered': bool(r['answer'].strip()), 'tokens': r['input_tokens'] + r['output_tokens'],
            'input_tokens': r['input_tokens'], 'output_tokens': r['output_tokens'], 'wall_s': r['wall_clock_ms'] / 1000,
            'malformed': r['malformed_tool_calls'], 'unknown': r['unknown_tools'], 'text_tool': r['text_only_tool_attempts'], 'error': bool(r['error'])}

def mean(cond, key, only=None):
    v = [float(x[cond][key]) for x in rows.values() if (only is None or x[cond][only])]
    return statistics.mean(v) if v else float('nan')

def boot(key, only=None):
    keys = [k for k, x in rows.items() if only is None or x['pgr'][only]]
    d = [float(rows[k]['relevan'][key]) - float(rows[k]['pgr'][key]) for k in keys]
    rng = random.Random(42)
    sims = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(4000))
    return statistics.mean(d), sims[100], sims[3899]

n_edit = sum(x['pgr']['has_edit_target'] for x in rows.values())
summary = {'tasks': len(rows), 'tasks_with_edit_target': n_edit, 'arms': {}}
for cond in ('pgr', 'relevan'):
    tot_calls = sum(x[cond]['tool_calls'] for x in rows.values())
    summary['arms'][cond] = {
        'edit_target_reached': sum(x[cond]['edit_hit'] for x in rows.values()),
        'claude_read_reached': sum(x[cond]['claude_hit'] for x in rows.values()),
        'answered': sum(x[cond]['answered'] for x in rows.values()),
        'mean_tool_calls': mean(cond, 'tool_calls'), 'mean_search_calls': mean(cond, 'search_calls'),
        'mean_read_calls': mean(cond, 'read_calls'), 'mean_turns': mean(cond, 'turns'),
        'mean_tokens': mean(cond, 'tokens'), 'mean_input_tokens': mean(cond, 'input_tokens'), 'mean_output_tokens': mean(cond, 'output_tokens'),
        'mean_wall_s': mean(cond, 'wall_s'),
        'median_calls_to_edit_target': statistics.median([x[cond]['calls_to_edit_target'] for x in rows.values() if x[cond]['edit_hit']] or [float('nan')]),
        'tool_calls_total': tot_calls, 'malformed': sum(x[cond]['malformed'] for x in rows.values()),
        'unknown_tool': sum(x[cond]['unknown'] for x in rows.values()), 'text_only_tool_attempts': sum(x[cond]['text_tool'] for x in rows.values()),
        'errors': sum(x[cond]['error'] for x in rows.values())}
summary['paired_relevan_minus_pgr'] = {k: boot(k) for k in ('edit_hit', 'claude_hit', 'tool_calls', 'search_calls', 'tokens', 'wall_s', 'answered')}
print(json.dumps(summary, indent=1))
(RUNS / 'summary.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1))
