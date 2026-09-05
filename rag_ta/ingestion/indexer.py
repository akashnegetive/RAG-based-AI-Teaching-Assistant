"""Chunk -> embed -> upsert into Chroma. Also the single entry point for ingesting media."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.ingestion.chunking import chunk_segments
from rag_ta.ingestion.media import extract_audio
from rag_ta.ingestion.transcribe import load_transcript, save_transcript, transcribe_audio
from rag_ta.models import Chunk, Segment
from rag_ta.retrieval.embeddings import embed_texts
from rag_ta.store import get_collection

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def index_chunks(chunks: list[Chunk], cfg: Settings = default_settings) -> int:
    if not chunks:
        return 0
    collection = get_collection(cfg)
    embeddings = embed_texts([c.text for c in chunks], cfg)
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[c.metadata() for c in chunks],
    )
    log.info("Indexed %d chunks for '%s'", len(chunks), chunks[0].title)
    return len(chunks)


def index_segments(title: str, segments: list[Segment], cfg: Settings = default_settings) -> int:
    chunks = chunk_segments(
        segments,
        title,
        target_seconds=cfg.chunk_target_seconds,
        max_chars=cfg.chunk_max_chars,
        overlap_seconds=cfg.chunk_overlap_seconds,
    )
    return index_chunks(chunks, cfg)


def ingest_audio(audio_path: Path, cfg: Settings = default_settings, progress: ProgressFn = _noop) -> int:
    title = audio_path.stem
    progress("Transcribing audio with Whisper…")
    segments = transcribe_audio(audio_path, cfg)
    save_transcript(title, segments, cfg.jsons_dir)
    progress(f"Transcribed {len(segments)} segments. Chunking and embedding…")
    return index_segments(title, segments, cfg)


def ingest_video(video_path: Path, cfg: Settings = default_settings, progress: ProgressFn = _noop) -> int:
    progress("Extracting audio from video…")
    audio_path = extract_audio(video_path, cfg.audios_dir)
    return ingest_audio(audio_path, cfg, progress)


def reindex_from_transcript(title: str, cfg: Settings = default_settings) -> int:
    """Re-chunk and re-embed an existing transcript (e.g. after changing chunk settings)."""
    json_path = cfg.jsons_dir / f"{title}.json"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    delete_lecture(title, cfg, remove_files=False)
    _, segments = load_transcript(json_path)
    return index_segments(title, segments, cfg)


def delete_lecture(title: str, cfg: Settings = default_settings, remove_files: bool = True) -> None:
    get_collection(cfg).delete(where={"title": title})
    if remove_files:
        for p in (
            cfg.videos_dir / f"{title}.mp4",
            cfg.audios_dir / f"{title}.mp3",
            cfg.jsons_dir / f"{title}.json",
            cfg.summaries_dir / f"{title}.json",
        ):
            if p.exists():
                p.unlink()
    log.info("Deleted lecture '%s'", title)


def lecture_exists(title: str, cfg: Settings = default_settings) -> bool:
    res = get_collection(cfg).get(where={"title": title}, limit=1)
    return bool(res.get("ids"))
