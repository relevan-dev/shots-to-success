"""Write a collected.jsonl copy with click labels from the training split, for train_ltr.py."""
import json, sys
from collections import defaultdict
collected, train, out = sys.argv[1:4]
seen, clicked = defaultdict(int), defaultdict(lambda: defaultdict(int))
for l in open(train):
    r = json.loads(l); seen[r['qid']] += 1
    for c in r['clicks']:
        clicked[r['qid']][c] += 1
lines = open(collected).read().splitlines()
rows = [lines[0]]
for l in lines[1:]:
    q = json.loads(l)
    if q['qid'] not in seen:
        continue
    n = seen[q['qid']]
    for c in q['candidates']:
        k = clicked[q['qid']][c['id']]
        c['label'] = (2 if 2 * k >= n else 1) if k else 0
        c['gain'] = c['label']
    q['idealDcg'] = sum((2 ** g - 1) / __import__('math').log2(i + 2) for i, g in enumerate(sorted((v and (2 if 2 * v >= n else 1) for v in clicked[q['qid']].values()), reverse=True)[:10]))
    rows.append(json.dumps(q))
open(out, 'w').write('\n'.join(rows) + '\n')
print(len(rows) - 1)
