"""End-to-end retrieval over a real (temporary) Chroma collection, with fake embeddings and
a fake reranker so no API key is required."""

import chromadb
import pytest

from rag_ta.config import Settings
from rag_ta.retrieval.retriever import HybridRetriever
from rag_ta.retrieval.sparse import SparseIndex

DOCS = {
    "c1": ("ML", "Ridge regression adds an L2 penalty that shrinks coefficients toward zero.", 0.0),
    "c2": ("ML", "Lasso regression uses an L1 penalty which can set coefficients exactly to zero.", 45.0),
    "c3": ("ML", "Bias variance tradeoff: high bias underfits, high variance overfits.", 90.0),
    "c4": ("DSA", "Kadane's algorithm finds the maximum subarray sum in linear time.", 0.0),
}

# toy 3-d "embeddings": [regularisation-ness, bias/variance-ness, dsa-ness]
VECS = {"c1": [1, 0, 0], "c2": [0.9, 0.1, 0], "c3": [0, 1, 0], "c4": [0, 0, 1]}
QUERY_VECS = {"L1 penalty": [1, 0, 0], "overfitting": [0, 1, 0], "maximum subarray": [0, 0, 1]}


class FakeReranker:
    def rerank(self, query, chunks):
        for c in chunks:
            c.score = 1.0 if any(w in c.text.lower() for w in query.lower().split()) else 0.05
        return sorted(chunks, key=lambda c: c.score, reverse=True)


@pytest.fixture
def retriever(tmp_path):
    cfg = Settings(data_dir=tmp_path, reranker="none", final_top_k=2, min_relevance=0.5)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection("test_col", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=list(DOCS),
        documents=[d[1] for d in DOCS.values()],
        embeddings=[VECS[i] for i in DOCS],
        metadatas=[{"title": d[0], "index": n, "start": d[2], "end": d[2] + 45} for n, d in enumerate(DOCS.values())],
    )
    return HybridRetriever(
        cfg,
        collection=col,
        sparse=SparseIndex.from_collection(col),
        reranker=FakeReranker(),
        embed_fn=lambda q: QUERY_VECS.get(q, [0.3, 0.3, 0.3]),
    )


def test_hybrid_finds_exact_keyword_even_when_dense_is_weak(retriever):
    # dense vector for this query points at "regularisation"; BM25 still surfaces the exact token "Kadane"
    retriever._embed = lambda q: [1, 0, 0]
    res = retriever.retrieve("Kadane")
    assert res.chunks[0].id == "c4"
    assert "sparse" in res.chunks[0].sources


def test_title_scope_restricts_results(retriever):
    res = retriever.retrieve("L1 penalty", title="DSA")
    assert all(c.title == "DSA" for c in res.chunks)


def test_relevance_threshold_flags_unanswerable(retriever):
    retriever._embed = lambda q: [0.3, 0.3, 0.3]
    res = retriever.retrieve("quantum chromodynamics")
    assert res.answerable is False


def test_returns_final_top_k_with_timestamps(retriever):
    res = retriever.retrieve("L1 penalty")
    assert len(res.chunks) == 2
    assert res.chunks[0].id == "c2"
    assert res.chunks[0].timestamp == "0:45–1:30"
    assert res.answerable


def test_empty_collection(tmp_path):
    cfg = Settings(data_dir=tmp_path, reranker="none")
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection("empty")
    r = HybridRetriever(
        cfg,
        collection=col,
        sparse=SparseIndex.from_collection(col),
        reranker=FakeReranker(),
        embed_fn=lambda q: [0, 0, 0],
    )
    res = r.retrieve("anything")
    assert res.chunks == [] and not res.answerable
