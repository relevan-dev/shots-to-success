# /// script
# requires-python = ">=3.11"
# dependencies = ["fastembed==0.7.3", "numpy>=2"]
# ///
"""Cache lab vectors for the BEIR-style dataset, in dataset order."""
import json, sys
import numpy as np
from fastembed import TextEmbedding

lab, prefix = sys.argv[1], sys.argv[2]
qrel_ids = {l.split('\t')[0] for l in open(f'{lab}/qrels/test.tsv').read().splitlines()[1:]}
docs = [json.loads(l) for l in open(f'{lab}/corpus.jsonl')]
queries = [q for q in (json.loads(l) for l in open(f'{lab}/queries.jsonl')) if q['_id'] in qrel_ids]
model = TextEmbedding('BAAI/bge-small-en-v1.5')
d = np.array(list(model.embed([(x['title'] + '\n' + x['text'])[:2000] for x in docs])), dtype='<f4')
q = np.array(list(model.query_embed([x['text'] for x in queries])), dtype='<f4')
np.save(f'{prefix}-d.npy', d); np.save(f'{prefix}-q.npy', q)
print(d.shape, q.shape)
