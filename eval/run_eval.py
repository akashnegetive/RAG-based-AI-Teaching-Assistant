"""Evaluate the RAG pipeline with RAGAS.

Usage:
    python -m eval.run_eval --dataset eval/dataset.jsonl [--scope-by-lecture]

Requires OPENAI_API_KEY and an indexed corpus. Produces eval/results/<timestamp>.json and prints
a metrics table. Run it before and after any retrieval/prompt change — that's the whole point.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag_ta.config import settings
from rag_ta.llm.answer import generate_answer
from rag_ta.retrieval.retriever import HybridRetriever


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(dataset: list[dict], scope_by_lecture: bool) -> dict:
    retriever = HybridRetriever(settings)
    rows = []
    t0 = time.time()
    for ex in dataset:
        title = ex.get("lecture") if scope_by_lecture else None
        res = retriever.retrieve(ex["question"], title=title)
        ans = generate_answer(ex["question"], res.chunks, res.answerable, settings)
        rows.append(
            {
                "user_input": ex["question"],
                "response": ans.text,
                "retrieved_contexts": [c.text for c in res.chunks],
                "reference": ex["ground_truth"],
                "answerable": res.answerable,
            }
        )
    latency = (time.time() - t0) / max(len(rows), 1)

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    ds = Dataset.from_list(rows)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    summary = {k: float(v) for k, v in scores.to_pandas().mean(numeric_only=True).items()}
    summary["avg_latency_s"] = round(latency, 2)
    summary["n"] = len(rows)
    return {
        "config": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "reranker": settings.reranker,
            "chunk_seconds": settings.chunk_target_seconds,
            "final_top_k": settings.final_top_k,
        },
        "metrics": summary,
        "rows": rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    ap.add_argument("--scope-by-lecture", action="store_true")
    args = ap.parse_args()

    result = run(load_dataset(Path(args.dataset)), args.scope_by_lecture)
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nConfig:", json.dumps(result["config"]))
    print("\n{:<22} {:>8}".format("metric", "score"))
    for k, v in result["metrics"].items():
        print(f"{k:<22} {v:>8.3f}" if isinstance(v, float) else f"{k:<22} {v:>8}")
    print(f"\nSaved → {out}")
