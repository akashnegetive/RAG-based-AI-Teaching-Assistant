"""Whisper transcription with timestamp preservation and long-file support."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.ingestion.media import audio_duration, split_audio
from rag_ta.llm.client import get_openai
from rag_ta.models import Segment

log = logging.getLogger(__name__)


def _transcribe_file(path: Path, offset: float, cfg: Settings) -> list[Segment]:
    client = get_openai(cfg)
    kwargs = {}
    if cfg.whisper_language:
        kwargs["language"] = cfg.whisper_language
    with path.open("rb") as f:
        # NOTE: `transcriptions`, not `translations` — the old code translated everything
        # to English, which silently rewrote non-English lectures.
        resp = client.audio.transcriptions.create(
            file=f,
            model=cfg.whisper_model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            **kwargs,
        )
    segs = getattr(resp, "segments", None) or []
    out = []
    for s in segs:
        start = float(getattr(s, "start", s["start"]))
        end = float(getattr(s, "end", s["end"]))
        text = getattr(s, "text", None) if not isinstance(s, dict) else s["text"]
        out.append(Segment(start=start + offset, end=end + offset, text=(text or "").strip()))
    return out


def transcribe_audio(audio_path: Path, cfg: Settings = default_settings) -> list[Segment]:
    """Transcribe an audio file of any length, returning globally-offset segments."""
    duration = audio_duration(audio_path)
    if duration <= cfg.audio_segment_seconds:
        log.info("Transcribing %s (%.0fs) in one call", audio_path.name, duration)
        return _transcribe_file(audio_path, 0.0, cfg)

    log.info("Audio is %.0fs; splitting into %ds pieces", duration, cfg.audio_segment_seconds)
    segments: list[Segment] = []
    with tempfile.TemporaryDirectory() as tmp:
        for part, offset in split_audio(audio_path, cfg.audio_segment_seconds, Path(tmp)):
            log.info("Transcribing %s (offset %.0fs)", part.name, offset)
            segments.extend(_transcribe_file(part, offset, cfg))
    return segments


def save_transcript(title: str, segments: list[Segment], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{title}.json"
    payload = {"title": title, "segments": [s.__dict__ for s in segments]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_transcript(path: Path) -> tuple[str, list[Segment]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    title = data.get("title", path.stem)
    raw = data.get("segments") or data.get("chunks") or []  # backwards-compatible with old JSONs
    return title, [Segment(float(r["start"]), float(r["end"]), r["text"]) for r in raw]
