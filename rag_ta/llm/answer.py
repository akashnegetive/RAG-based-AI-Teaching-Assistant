from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.llm import prompts
from rag_ta.llm.client import get_openai
from rag_ta.models import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]
    grounded: bool


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[S{i}] ({c.title} @ {c.timestamp})\n{c.text}" for i, c in enumerate(chunks, start=1))


def generate_answer(
    question: str, chunks: list[RetrievedChunk], answerable: bool, cfg: Settings = default_settings
) -> Answer:
    if not chunks or not answerable:
        return Answer(
            text="I couldn't find this in the indexed lectures. Try rephrasing, or check that the "
            "relevant lecture has been uploaded.",
            sources=chunks,
            grounded=False,
        )
    resp = get_openai(cfg).responses.create(
        model=cfg.chat_model,
        instructions=prompts.ANSWER_SYSTEM,
        input=prompts.ANSWER_USER.format(context=format_context(chunks), question=question),
    )
    return Answer(text=resp.output_text, sources=chunks, grounded=True)
