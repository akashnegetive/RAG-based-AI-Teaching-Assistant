"""The answer path as an LCEL chain.

    retriever → format context → prompt → ChatOpenAI → parse

Composing it this way (rather than imperative calls) gives streaming, batching, async and
callback tracing for free, and makes the pipeline inspectable as a graph.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.llm import prompts
from rag_ta.retrieval.langchain_retriever import document_to_chunk

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.ANSWER_SYSTEM),
        ("human", prompts.ANSWER_USER),
    ]
)


def format_documents(docs) -> str:
    return "\n\n".join(
        f"[S{i}] ({d.metadata['title']} @ {d.metadata['timestamp']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def build_answer_chain(retriever: Runnable, cfg: Settings = default_settings) -> Runnable:
    """question -> {"answer": str, "documents": [Document]}"""
    llm = ChatOpenAI(model=cfg.chat_model, timeout=cfg.openai_timeout, max_retries=cfg.openai_max_retries)

    generate = (
        {
            "context": RunnableLambda(lambda x: format_documents(x["documents"])),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(question=RunnablePassthrough(), documents=retriever) | RunnablePassthrough.assign(
        answer=generate
    )


def stream_answer(chain: Runnable, question: str) -> Iterator[str]:
    """Token-by-token streaming for the UI."""
    for part in chain.stream(question):
        if isinstance(part, dict) and "answer" in part:
            yield part["answer"]


def chunks_from_result(result: dict):
    return [document_to_chunk(d) for d in result["documents"]]
