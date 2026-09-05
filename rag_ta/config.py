"""Central configuration. Everything tunable lives here and is overridable via env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # ---- Paths ----
    data_dir: Path = field(default_factory=lambda: Path(_env("RAG_DATA_DIR", str(Path.home() / "rag_data"))))

    # ---- Models ----
    chat_model: str = field(default_factory=lambda: _env("RAG_CHAT_MODEL", "gpt-5"))
    embedding_model: str = field(default_factory=lambda: _env("RAG_EMBEDDING_MODEL", "text-embedding-3-large"))
    whisper_model: str = field(default_factory=lambda: _env("RAG_WHISPER_MODEL", "whisper-1"))
    # Language hint for Whisper ("" = auto-detect). Use "hi" / "en" etc. to pin.
    whisper_language: str = field(default_factory=lambda: _env("RAG_WHISPER_LANGUAGE", ""))

    # ---- Chunking ----
    chunk_target_seconds: float = field(default_factory=lambda: _env_float("RAG_CHUNK_SECONDS", 45.0))
    chunk_max_chars: int = field(default_factory=lambda: _env_int("RAG_CHUNK_MAX_CHARS", 1200))
    chunk_overlap_seconds: float = field(default_factory=lambda: _env_float("RAG_CHUNK_OVERLAP", 10.0))

    # ---- Retrieval ----
    dense_top_k: int = field(default_factory=lambda: _env_int("RAG_DENSE_TOP_K", 20))
    sparse_top_k: int = field(default_factory=lambda: _env_int("RAG_SPARSE_TOP_K", 20))
    rerank_top_k: int = field(default_factory=lambda: _env_int("RAG_RERANK_TOP_K", 15))
    final_top_k: int = field(default_factory=lambda: _env_int("RAG_FINAL_TOP_K", 5))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    # Below this reranker score we treat the question as unanswerable from the corpus.
    min_relevance: float = field(default_factory=lambda: _env_float("RAG_MIN_RELEVANCE", 0.15))
    # "llm" (OpenAI, no extra deps), "flashrank" (local cross-encoder), or "none".
    reranker: str = field(default_factory=lambda: _env("RAG_RERANKER", "llm"))

    # ---- Audio ----
    # Whisper API hard limit is 25 MB; we split anything longer than this many seconds.
    audio_segment_seconds: int = field(default_factory=lambda: _env_int("RAG_AUDIO_SEGMENT_SECONDS", 600))

    # ---- LLM ----
    openai_timeout: float = field(default_factory=lambda: _env_float("RAG_OPENAI_TIMEOUT", 120.0))
    openai_max_retries: int = field(default_factory=lambda: _env_int("RAG_OPENAI_MAX_RETRIES", 4))
    # Summaries: transcripts longer than this (chars) go through map-reduce.
    summary_map_chars: int = field(default_factory=lambda: _env_int("RAG_SUMMARY_MAP_CHARS", 24000))

    # ---- Chroma ----
    collection_name: str = field(default_factory=lambda: _env("RAG_COLLECTION", "lecture_embeddings"))

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def audios_dir(self) -> Path:
        return self.data_dir / "audios"

    @property
    def jsons_dir(self) -> Path:
        return self.data_dir / "jsons"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_store"

    @property
    def summaries_dir(self) -> Path:
        return self.data_dir / "summaries"

    def ensure_dirs(self) -> None:
        for d in (self.videos_dir, self.audios_dir, self.jsons_dir, self.chroma_dir, self.summaries_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
