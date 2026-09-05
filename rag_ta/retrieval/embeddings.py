from __future__ import annotations

import logging

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.llm.client import get_openai

log = logging.getLogger(__name__)


def embed_texts(texts: list[str], cfg: Settings = default_settings, batch_size: int = 64) -> list[list[float]]:
    client = get_openai(cfg)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=cfg.embedding_model, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def embed_query(text: str, cfg: Settings = default_settings) -> list[float]:
    return embed_texts([text], cfg)[0]
