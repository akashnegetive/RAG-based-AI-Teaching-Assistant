"""ChromaDB access. One persistent client per process."""

from __future__ import annotations

from functools import lru_cache

import chromadb

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings


@lru_cache(maxsize=4)
def _client(path: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=path)


def get_collection(cfg: Settings = default_settings):
    cfg.ensure_dirs()
    return _client(str(cfg.chroma_dir)).get_or_create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def list_titles(cfg: Settings = default_settings) -> list[str]:
    res = get_collection(cfg).get(include=["metadatas"])
    return sorted({m["title"] for m in res["metadatas"]})
