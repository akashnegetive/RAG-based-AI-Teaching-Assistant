"""Hybrid retriever: dense (Chroma) + sparse (BM25) -> RRF -> rerank -> threshold."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.models import RetrievedChunk
from rag_ta.retrieval.fusion import rrf
from rag_ta.retrieval.rerank import Reranker, build_reranker
from rag_ta.retrieval.sparse import SparseIndex
from rag_ta.store import get_collection

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    candidates_considered: int
    answerable: bool  # False when even the best chunk is below min_relevance


class HybridRetriever:
    def __init__(
        self,
        cfg: Settings = default_settings,
        collection=None,
        sparse: SparseIndex | None = None,
        reranker: Reranker | None = None,
        embed_fn=None,
    ):
        self.cfg = cfg
        self.collection = collection if collection is not None else get_collection(cfg)
        self.sparse = sparse if sparse is not None else SparseIndex.from_collection(self.collection)
        self.reranker = reranker if reranker is not None else build_reranker(cfg)
        if embed_fn is None:
            from rag_ta.retrieval.embeddings import embed_query

            embed_fn = lambda q: embed_query(q, cfg)  # noqa: E731
        self._embed = embed_fn

    def refresh_sparse(self) -> None:
        """Call after ingesting/deleting so BM25 sees the new documents."""
        self.sparse = SparseIndex.from_collection(self.collection)

    # ---- individual stages -------------------------------------------------

    def _dense(self, query: str, title: str | None) -> list[str]:
        if self.collection.count() == 0:
            return []
        kwargs = {
            "query_embeddings": [self._embed(query)],
            "n_results": min(self.cfg.dense_top_k, self.collection.count()),
        }
        if title:
            kwargs["where"] = {"title": title}
        res = self.collection.query(**kwargs)
        return res["ids"][0] if res["ids"] else []

    def _sparse(self, query: str, title: str | None) -> list[str]:
        return [i for i, _ in self.sparse.search(query, top_k=self.cfg.sparse_top_k, title=title)]

    def _hydrate(self, fused: list[tuple[str, float, list[str]]]) -> list[RetrievedChunk]:
        ids = [f[0] for f in fused]
        if not ids:
            return []
        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        by_id = {i: (d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)}
        out = []
        for doc_id, score, sources in fused:
            if doc_id not in by_id:
                continue
            doc, meta = by_id[doc_id]
            out.append(
                RetrievedChunk(
                    id=doc_id,
                    title=meta["title"],
                    index=int(meta.get("index", meta.get("number", 0)) or 0) if str(meta.get("index", meta.get("number", 0))).isdigit() else 0,
                    start=float(meta["start"]),
                    end=float(meta["end"]),
                    text=doc,
                    score=score,
                    sources=sources,
                )
            )
        return out

    # ---- public API --------------------------------------------------------

    def retrieve(self, query: str, title: str | None = None) -> RetrievalResult:
        dense_ids = self._dense(query, title)
        sparse_ids = self._sparse(query, title)
        fused = rrf({"dense": dense_ids, "sparse": sparse_ids}, k=self.cfg.rrf_k)[: self.cfg.rerank_top_k]
        candidates = self._hydrate(fused)
        log.info("query=%r dense=%d sparse=%d fused=%d", query, len(dense_ids), len(sparse_ids), len(candidates))

        reranked = self.reranker.rerank(query, candidates)
        top = reranked[: self.cfg.final_top_k]
        answerable = bool(top) and top[0].score >= self.cfg.min_relevance
        return RetrievalResult(chunks=top, candidates_considered=len(candidates), answerable=answerable)
