"""LangChain adapter over HybridRetriever.

Exposing the pipeline as a `BaseRetriever` means it drops into any LangChain component —
LCEL chains, agents, evaluation tooling — without rewriting the retrieval logic. The hybrid
search, fusion and reranking all still happen in `HybridRetriever`; this is a thin translation
layer to/from `Document`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_ta.models import RetrievedChunk
from rag_ta.retrieval.retriever import RetrievalResult


@runtime_checkable
class SupportsRetrieve(Protocol):
    """Anything with this shape works — HybridRetriever, MultiQueryRetriever, or a test double."""

    def retrieve(self, query: str, title: str | None = ...) -> RetrievalResult: ...


def chunk_to_document(chunk: RetrievedChunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "id": chunk.id,
            "title": chunk.title,
            "index": chunk.index,
            "start": chunk.start,
            "end": chunk.end,
            "timestamp": chunk.timestamp,
            "score": chunk.score,
            "sources": ",".join(chunk.sources),
        },
    )


def document_to_chunk(doc: Document) -> RetrievedChunk:
    m = doc.metadata
    return RetrievedChunk(
        id=m["id"],
        title=m["title"],
        index=int(m.get("index", 0)),
        start=float(m["start"]),
        end=float(m["end"]),
        text=doc.page_content,
        score=float(m.get("score", 0.0)),
        sources=[s for s in str(m.get("sources", "")).split(",") if s],
    )


class LectureRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by the hybrid pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Typed as Any because pydantic would otherwise pin this to one concrete class; the
    # SupportsRetrieve protocol above is the real contract, checked in __init__.
    retriever: Any
    title: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not isinstance(self.retriever, SupportsRetrieve):
            raise TypeError(f"{type(self.retriever).__name__} has no .retrieve(query, title) method")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None, **kwargs: Any
    ) -> list[Document]:
        result = self.retriever.retrieve(query, title=self.title)
        docs = [chunk_to_document(c) for c in result.chunks]
        for d in docs:
            d.metadata["answerable"] = result.answerable
        return docs
