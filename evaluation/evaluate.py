import json
import sys
from collections import defaultdict
from pathlib import Path

TOP_K = int(next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--top-k"), 3))
sys.argv = ["evaluate.py", "--retriever-only", "--top-k", str(TOP_K)]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core import Settings
from llama_index.core.llms.mock import MockLLM
from llama_index.embeddings.ollama import OllamaEmbedding

from lib.config import DATASET_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, PERSIST_DIR
from lib.indexer import load_or_build_index
from lib.retriever import build_hybrid_query_engine

#setup

Settings.embed_model = OllamaEmbedding(
    model_name=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)
Settings.llm = MockLLM(max_tokens=0)

index = load_or_build_index(DATASET_DIR, PERSIST_DIR)
query_engine = build_hybrid_query_engine(index, top_k=TOP_K)
retriever = query_engine._retriever

eval_path = Path(__file__).parent / "eval_queries.json"
with open(eval_path) as f:
    queries = json.load(f)

#evaluation loop

hits = 0
total = len(queries)
reciprocal_ranks = []
results = []

for q in queries:
    nodes = retriever.retrieve(q["question"])
    retrieved_ids = [str(node.node.metadata["id"]) for node in nodes]
    expected_id = str(q["expected_guide_id"])

    hit = expected_id in retrieved_ids
    if hit:
        hits += 1
        rank = retrieved_ids.index(expected_id) + 1
        reciprocal_ranks.append(1 / rank)
    else:
        rank = None
        reciprocal_ranks.append(0.0)

    results.append({
        "question": q["question"],
        "category": q["category"],
        "expected_id": expected_id,
        "retrieved_ids": retrieved_ids,
        "hit": hit,
        "rank": rank,
        "rr": reciprocal_ranks[-1],
    })

    print(f"{'HIT ' if hit else 'MISS'} | expected {expected_id} | got {retrieved_ids} | {q['question'][:60]}")

#metrics

hit_rate = hits / total
mrr = sum(reciprocal_ranks) / total

category_hits = defaultdict(int)
category_totals = defaultdict(int)
for r in results:
    cat = r["category"]
    category_totals[cat] += 1
    if r["hit"]:
        category_hits[cat] += 1

print(f"\nResults @ k={TOP_K}")
print(f"  Hit rate: {hits}/{total} = {hit_rate:.1%}")
print(f"  MRR:      {mrr:.3f}")
print(f"\nPer category:")
for cat in sorted(category_totals):
    h = category_hits[cat]
    t = category_totals[cat]
    print(f"  {cat:<25} {h}/{t}  ({h/t:.0%})")

#save results
output = {
    "k": TOP_K,
    "hit_rate": round(hit_rate, 4),
    "mrr": round(mrr, 4),
    "per_category": {
        cat: {
            "hits": category_hits[cat],
            "total": category_totals[cat],
            "hit_rate": round(category_hits[cat] / category_totals[cat], 4),
        }
        for cat in sorted(category_totals)
    },
    "results": results,
}

out_path = Path(__file__).parent / f"results_k{TOP_K}.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {out_path}")