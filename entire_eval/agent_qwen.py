# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.50", "fastembed==0.7.3", "numpy>=2", "requests"]
# ///
"""Entire's public60 agent loop on an OpenAI-compatible endpoint, with pgr or Relevan as search_code.

    uv run agent_qwen.py <entire dir> <out dir> --conditions pgr relevan [--limit N] [--parallel 4]
"""
import argparse, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, requests
from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument('root', type=Path); p.add_argument('out', type=Path)
p.add_argument('--conditions', nargs='+', default=['pgr', 'relevan'])
p.add_argument('--limit', type=int, default=0); p.add_argument('--parallel', type=int, default=4)
p.add_argument('--max-turns', type=int, default=12)
p.add_argument('--model', default='Qwen3-Coder-30B-A3B-Instruct-FP8')
p.add_argument('--base-url', default='http://127.0.0.1:18000/v1')
a = p.parse_args()

sys.path.insert(0, str(a.root / 'pgr/eval/v2'))
from tools import TOOL_DEFINITIONS, SYSTEM_PROMPT
from backends import baseline as rg, pgr_backend
REPO = str(a.root / 'cli-0415')
TOOLS = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'], 'parameters': t['input_schema']}} for t in TOOL_DEFINITIONS]
client = OpenAI(base_url=a.base_url, api_key=os.environ['QWEN_VLLM_API_KEY'], timeout=600)

from fastembed import TextEmbedding
_embed = TextEmbedding('BAAI/bge-small-en-v1.5'); _lock = threading.Lock()
def vector(q):
    with _lock:
        v = list(_embed.query_embed([q]))[0]
    out = np.zeros(1024, dtype=np.float32); out[:384] = v
    return out.tolist()

def words(q):
    q = re.sub(r'\\[bBsSdDwW]', ' ', q); q = re.sub(r'\\(.)', r'\1', q)
    return re.sub(r'\s+', ' ', re.sub(r'[\^$.*+?()\[\]{}|"\']', ' ', q)).strip()

def relevan_search(task_id, ti):
    """Candidates are rg's matches plus Relevan's top 30. Relevan's click-trained order goes first."""
    q, mf = ti['query'], ti.get('max_files', 10) or 10
    per_file = min(ti.get('max_matches_per_file', 3) or 3, 3)
    kw = dict(path_glob=ti.get('path_glob', ''), file_type=ti.get('file_type', ''), max_matches_per_file=per_file,
              context_before=0, context_after=0)
    rg_out = rg.search_code(repo_root=REPO, query=q, max_files=200, **kw)
    blocks, cur = {}, None
    for line in rg_out.splitlines():
        if line and not line.startswith(' ') and not line.startswith('('):
            cur = line.strip().removeprefix('./'); blocks[cur] = []
        elif cur and not line.startswith('('):
            blocks[cur].append(line)
    nq = words(q)
    ids = []
    if nq:
        r = requests.post('http://127.0.0.1:18100', json={'query': nq, 'vector': vector(nq), 'model': task_id, 'size': 30}, timeout=60)
        ids = r.json()['ids'] if r.ok else []
    glob = kw['path_glob']
    if glob:
        import fnmatch
        ids = [i for i in ids if fnmatch.fnmatch(i, glob) or fnmatch.fnmatch(i.rsplit('/', 1)[-1], glob)]
    order = [i for i in ids if i in blocks] + [f for f in blocks if f not in ids] + [i for i in ids if i not in blocks]
    out = []
    for f in order[:mf]:
        out.append(f)
        if blocks.get(f):
            out.extend([l[:160] for l in blocks[f] if '|' in l][:per_file])
        else:
            out.append('  (no literal match; ranked by meaning)')
    return '\n'.join(out) if out else 'No matches found.'

def dispatch(cond, task_id, name, ti):
    if name == 'search_code':
        if cond == 'relevan':
            return relevan_search(task_id, ti)
        return pgr_backend.search_code(repo_root=REPO, query=ti['query'], path_glob=ti.get('path_glob', ''), file_type=ti.get('file_type', ''),
                                       max_files=ti.get('max_files', 10), max_matches_per_file=ti.get('max_matches_per_file', 3))
    b = pgr_backend
    if name == 'read_code':
        return b.read_code(repo_root=REPO, path=ti['path'], start_line=ti.get('start_line', 1), end_line=ti.get('end_line', 0), max_lines=ti.get('max_lines', 80))
    if name == 'find_files':
        return b.find_files(repo_root=REPO, pattern=ti.get('pattern', ''), glob=ti.get('glob', ''), file_type=ti.get('file_type', ''), max_results=ti.get('max_results', 50))
    if name == 'list_dir':
        return b.list_dir(repo_root=REPO, path=ti.get('path', '.'), recursive=ti.get('recursive', False), max_results=ti.get('max_results', 100))
    return f'Unknown tool: {name}'

def run(task, cond):
    path = a.out / cond / f"{task['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    res = {'task_id': task['id'], 'condition': cond, 'task_type': task['type'], 'prompt': task['prompt'], 'answer': '', 'tool_calls': [],
           'turns': 0, 'input_tokens': 0, 'output_tokens': 0, 'malformed_tool_calls': 0, 'unknown_tools': 0, 'text_only_tool_attempts': 0, 'error': ''}
    msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': task['prompt']}]
    t0 = time.time()
    try:
        while res['turns'] < a.max_turns:
            res['turns'] += 1
            r = client.chat.completions.create(model=a.model, messages=msgs, tools=TOOLS, max_tokens=4096)
            res['input_tokens'] += r.usage.prompt_tokens; res['output_tokens'] += r.usage.completion_tokens
            m = r.choices[0].message
            calls = m.tool_calls or []
            if not calls:
                res['answer'] = m.content or ''
                if re.search(r'<tool_call>|<function=', res['answer']):
                    res['text_only_tool_attempts'] += 1
                break
            msgs.append({'role': 'assistant', 'content': m.content or '', 'tool_calls': [c.model_dump() for c in calls]})
            for c in calls:
                try:
                    ti = json.loads(c.function.arguments or '{}')
                    if not isinstance(ti, dict):
                        raise ValueError('arguments are not an object')
                except Exception as e:
                    res['malformed_tool_calls'] += 1
                    out = f'Invalid tool arguments: {e}'
                    ti = {'_raw': c.function.arguments}
                    c.function.arguments = '{}'
                    msgs[-1]['tool_calls'] = [x.model_dump() for x in calls]
                else:
                    if c.function.name not in {t['name'] for t in TOOL_DEFINITIONS}:
                        res['unknown_tools'] += 1
                    ts = time.time()
                    try:
                        out = dispatch(cond, task['id'], c.function.name, ti)
                    except KeyError as e:
                        res['malformed_tool_calls'] += 1
                        out = f'Missing required argument: {e}'
                    res['tool_calls'].append({'tool_name': c.function.name, 'tool_input': ti, 'output': out[:500], 'output_chars': len(out),
                                              'duration_ms': (time.time() - ts) * 1000, 'turn': res['turns']})
                msgs.append({'role': 'tool', 'tool_call_id': c.id, 'content': out})
    except Exception as e:
        res['error'] = repr(e)[:500]
    res['wall_clock_ms'] = (time.time() - t0) * 1000
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    return res

tasks = json.loads((a.root / 'pgr/public_release/benchmarks/entireio_cli_ranking_public60/tasks_full60.json').read_text())
if a.limit:
    tasks = tasks[:a.limit]
jobs = [(t, c) for t in tasks for c in a.conditions]
with ThreadPoolExecutor(a.parallel) as ex:
    for r in ex.map(lambda j: run(*j), jobs):
        print(r['task_id'], r['condition'], 'turns', r['turns'], 'calls', len(r['tool_calls']), 'bad', r['malformed_tool_calls'], 'err', r['error'][:80], flush=True)
