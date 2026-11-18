// Lab search server for the pilot. Copy into relevan-dev/monorepo packages/scripts/src/ and run with tsx.
import type { Reranker } from "@relevan/search/query/rerank/reranker";

import { createQueryEmbeddingTool } from "@relevan/search/query/enrich/query-embedding-tool";
import { executeQuery } from "@relevan/search/query/query-pipeline";
import { createLtrReranker, rerankLtrOptionsSchema } from "@relevan/search/query/rerank/ltr-reranker";
import { defaultQueryStrategy } from "@relevan/search/query/retrieve/query-strategy";
import { readFileSync, readdirSync } from "node:fs";
import { createServer } from "node:http";

import { LAB_EMBEDDING_MODEL } from "./ltr/lab-index";
import { openLab } from "./ltr/lab-run";

const [dir, vectors, modelsDir] = process.argv.slice(2) as [string, string, string];
const lab = await openLab({ datasetName: "entire-cli", dir, vectors, fill: false, topK: 30, semanticWeight: 0.6 });
const rerankers = new Map<string, Reranker>();
for (const file of readdirSync(modelsDir)) {
  const config = rerankLtrOptionsSchema.parse(JSON.parse(readFileSync(`${modelsDir}/${file}`, "utf-8")));
  rerankers.set(file.replace(/\.json$/, ""), createLtrReranker({ ...config, topK: 30 }, 5000));
}
const strategy = defaultQueryStrategy({ fieldBoosts: lab.setup.fieldBoosts }, {}, { fields: {} }, { semanticWeight: 0.6 });

createServer((req, res) => {
  let body = "";
  req.on("data", (chunk: Buffer) => (body += chunk.toString()));
  req.on("end", () => {
    const { query, vector, model, size } = JSON.parse(body) as { query: string; vector: number[]; model?: string; size?: number };
    const embeddingTool = createQueryEmbeddingTool({ embedQuery: async () => Promise.resolve(vector) }, LAB_EMBEDDING_MODEL, { timeoutMs: 5000 });
    const reranker = model ? rerankers.get(model) : undefined;
    executeQuery(
      {
        requestId: crypto.randomUUID(),
        organizationId: "eval",
        apiKeyId: "eval", // pragma: allowlist secret
        indexInstance: lab.instance,
        queryPartition: { status: "off" },
        rawQuery: query,
        size: size ?? 10,
        strategy,
        enrichmentTools: [embeddingTool],
        ...(reranker ? { reranker } : {}),
      },
      { searchClient: lab.searchClient },
    )
      .then((result) => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ ids: result.result.results.map((hit) => hit.id), reranked: Boolean(reranker) }));
      })
      .catch((error: unknown) => {
        res.writeHead(500);
        res.end(String(error));
      });
  });
}).listen(18100, "127.0.0.1");
console.log(`entire search server on 18100, ${String(rerankers.size)} models`);
