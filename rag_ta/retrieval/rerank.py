"""Rerankers score (query, passage) pairs jointly — far more accurate than bi-encoder
similarity, but too slow to run over the whole corpus. So: retrieve ~20 candidates
cheaply, rerank them, keep the top 5.

Three backends, selected by RAG_RERANKER:
  llm       – OpenAI pointwise 0–10 relevance judgments (default; no extra dependencies)
  flashrank – local ONNX cross-encoder (ms-marco MiniLM), free & fast on CPU
  none      – pass-through (keeps RRF order)
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.models import RetrievedChunk

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoopReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # Normalise RRF scores into 0–1 so the min_relevance threshold still means something.
        if not chunks:
            return chunks
        top = max(c.score for c in chunks) or 1.0
        for c in chunks:
            c.score = c.score / top
        return chunks


class LLMReranker:
    """Asks the chat model to grade each passage's relevance on 0–10 in a single call."""

    PROMPT = """You are grading transcript passages for how well they answer a student's question.
Score each passage from 0 (irrelevant) to 10 (directly and completely answers it).
Judge only on the passage content. Return ONLY a JSON object mapping passage index to score,
e.g. {{"0": 8, "1": 2}}.

Question: {query}

Passages:
{passages}"""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        from rag_ta.llm.client import get_openai

        passages = "\n\n".join(f"[{i}] {c.text[:1500]}" for i, c in enumerate(chunks))
        resp = get_openai(self.cfg).responses.create(
            model=self.cfg.chat_model,
            input=self.PROMPT.format(query=query, passages=passages),
        )
        try:
            raw = resp.output_text.strip().strip("`")
            raw = raw[raw.find("{") : raw.rfind("}") + 1]
            grades = {int(k): float(v) for k, v in json.loads(raw).items()}
        except Exception as e:  # noqa: BLE001 — fall back gracefully, never crash a query
            log.warning("LLM reranker returned unparsable output (%s); keeping fusion order", e)
            return NoopReranker().rerank(query, chunks)
        for i, c in enumerate(chunks):
            c.score = grades.get(i, 0.0) / 10.0
        return sorted(chunks, key=lambda c: c.score, reverse=True)


class FlashRankReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker  # optional dependency

        self._ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        from flashrank import RerankRequest

        req = RerankRequest(query=query, passages=[{"id": c.id, "text": c.text} for c in chunks])
        by_id = {c.id: c for c in chunks}
        out = []
        for r in self._ranker.rerank(req):
            c = by_id[r["id"]]
            c.score = float(r["score"])
            out.append(c)
        return out


def build_reranker(cfg: Settings = default_settings) -> Reranker:
    name = cfg.reranker.lower()
    if name == "none":
        return NoopReranker()
    if name == "flashrank":
        try:
            return FlashRankReranker()
        except ImportError:
            log.warning("flashrank not installed; falling back to LLM reranker")
    return LLMReranker(cfg)
