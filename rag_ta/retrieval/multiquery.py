"""Multi-query expansion.

A student's phrasing rarely matches the lecturer's. "Why does my model do great on training
data but badly on test data?" and "overfitting" retrieve different passages. So: ask the LLM
for N alternative phrasings, retrieve for each in parallel, and fuse the rankings with RRF.
Recall goes up because a passage only has to be found by one phrasing, and RRF rewards
passages found by several.

Built as an LCEL chain (prompt | llm | parser) rather than a prebuilt retriever so the
expansion prompt stays visible and tunable.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.models import RetrievedChunk
from rag_ta.retrieval.fusion import rrf
from rag_ta.retrieval.langchain_retriever import SupportsRetrieve
from rag_ta.retrieval.retriever import RetrievalResult

log = logging.getLogger(__name__)

EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite a student's question about a lecture into {n} alternative search queries. "
            "Vary the vocabulary: include the formal/technical wording a lecturer would use, and a "
            "plain-language version. Keep each on its own line, no numbering, no extra text.",
        ),
        ("human", "{question}"),
    ]
)


def build_expansion_chain(cfg: Settings = default_settings):
    llm = ChatOpenAI(model=cfg.chat_model, timeout=cfg.openai_timeout, max_retries=cfg.openai_max_retries)
    return EXPANSION_PROMPT | llm | StrOutputParser()


def parse_queries(raw: str, original: str, limit: int) -> list[str]:
    seen = {original.strip().lower()}
    out = [original]
    for line in raw.splitlines():
        q = line.strip().lstrip("-•0123456789. ").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) > limit:
            break
    return out


class MultiQueryRetriever:
    """Wraps HybridRetriever: expand the query, retrieve for each variant, fuse with RRF."""

    def __init__(self, retriever: SupportsRetrieve, cfg: Settings = default_settings, expansion_chain=None):
        self.retriever = retriever
        self.cfg = cfg
        self._chain = expansion_chain if expansion_chain is not None else build_expansion_chain(cfg)

    def expand(self, question: str) -> list[str]:
        try:
            raw = self._chain.invoke({"question": question, "n": self.cfg.multiquery_variants})
        except Exception as e:  # noqa: BLE001 — never let expansion break a query
            log.warning("Query expansion failed (%s); using the original question only", e)
            return [question]
        return parse_queries(raw, question, self.cfg.multiquery_variants)

    def retrieve(self, question: str, title: str | None = None) -> RetrievalResult:
        queries = self.expand(question)
        log.info("Expanded into %d queries: %s", len(queries), queries)

        rankings: dict[str, list[str]] = {}
        pool: dict[str, RetrievedChunk] = {}
        for i, q in enumerate(queries):
            res = self.retriever.retrieve(q, title=title)
            rankings[f"q{i}"] = [c.id for c in res.chunks]
            for c in res.chunks:
                pool.setdefault(c.id, c)

        fused = rrf(rankings, k=self.cfg.rrf_k)[: self.cfg.final_top_k]
        # Keep each chunk's reranker score for display; RRF only decides the ordering here.
        chunks = [pool[doc_id] for doc_id, _, _ in fused]

        answerable = bool(chunks) and max((c.score for c in chunks), default=0.0) >= self.cfg.min_relevance
        return RetrievalResult(chunks=chunks, candidates_considered=len(pool), answerable=answerable)
