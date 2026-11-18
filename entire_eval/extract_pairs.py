"""Extract (search query -> files read next) click pairs from Entire's public checkpoint export."""
import gzip, json, re, shlex, sys
from pathlib import Path

DATA = Path(sys.argv[1])
REPO = Path(sys.argv[2])
OUT = Path(sys.argv[3])
WINDOW = 6  # tool calls after a search that may count as clicks

repo_files = [p.relative_to(REPO).as_posix() for p in REPO.rglob('*') if p.is_file() and '.git/' not in p.as_posix()]
by_suffix = {}
for f in repo_files:
    parts = f.split('/')
    for i in range(len(parts)):
        by_suffix.setdefault('/'.join(parts[i:]), set()).add(f)

def resolve(path):
    p = path.strip().strip('"\'')
    if not p:
        return None
    parts = p.split('/')
    for i in range(len(parts)):
        cand = by_suffix.get('/'.join(parts[i:]))
        if cand and len(cand) == 1 and len(parts) - i >= 2 or (cand and len(cand) == 1 and i == len(parts) - 1 and '.' in p):
            return next(iter(cand))
    return None

def tool_uses(chunks):
    for k in sorted(chunks, key=lambda x: int(x)):
        for line in chunks[k].splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('type') != 'assistant':
                continue
            content = (ev.get('message') or {}).get('content')
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_use':
                    yield ev.get('timestamp'), block.get('name'), block.get('input') or {}

def bash_search_query(cmd):
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return None
    for i, t in enumerate(toks):
        if t in ('rg', 'grep') or t.endswith('/rg'):
            rest = [x for x in toks[i+1:] if x not in ('|', '&&', ';')]
            pat = None
            j = 0
            while j < len(rest):
                x = rest[j]
                if x in ('-e', '--regexp'):
                    return rest[j+1] if j + 1 < len(rest) else None
                if x in ('-g', '--glob', '-t', '--type', '-A', '-B', '-C', '-m', '--max-count', '--include', '--exclude'):
                    j += 2; continue
                if x.startswith('-'):
                    j += 1; continue
                return x
            return pat
    return None

n_ckpt = 0
rows = []
with gzip.open(DATA, 'rt') as fh:
    for line in fh:
        ck = json.loads(line)
        n_ckpt += 1
        calls = list(tool_uses(ck.get('transcript_stripped') or {}))
        for i, (ts, name, inp) in enumerate(calls):
            query, kind, glob = None, None, ''
            if name == 'Grep':
                query, kind, glob = inp.get('pattern'), 'grep', inp.get('glob') or ''
            elif name == 'Bash':
                q = bash_search_query(inp.get('command', ''))
                if q:
                    query, kind = q, 'bash'
            if not query or len(query) < 2:
                continue
            clicks = []
            for ts2, n2, i2 in calls[i+1:i+1+WINDOW]:
                if n2 in ('Grep', 'Glob') or (n2 == 'Bash' and bash_search_query(i2.get('command', ''))):
                    break
                if n2 == 'Read':
                    f = resolve(i2.get('file_path', ''))
                    if f and f not in clicks:
                        clicks.append(f)
            rows.append({'checkpoint_id': ck['checkpoint_id'], 'session_id': ck['session_id'],
                         'created_at': ck['checkpoint_created_at'], 'ts': ts, 'kind': kind,
                         'query': query, 'glob': glob, 'clicks': clicks, 'prompt': (ck.get('prompt') or '')[:300]})

OUT.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
withc = [r for r in rows if r['clicks']]
print('checkpoints', n_ckpt, 'searches', len(rows), 'with_clicks', len(withc),
      'grep', sum(r['kind']=='grep' for r in withc), 'bash', sum(r['kind']=='bash' for r in withc))
