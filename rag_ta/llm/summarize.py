"""Lecture summaries with map-reduce for long transcripts and on-disk caching."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass

from rag_ta.config import Settings
from rag_ta.config import settings as default_settings
from rag_ta.llm import prompts
from rag_ta.llm.client import get_openai
from rag_ta.store import get_collection

log = logging.getLogger(__name__)


@dataclass
class Summary:
    quick: str
    detailed: str
    transcript_hash: str


def _complete(prompt: str, cfg: Settings) -> str:
    return get_openai(cfg).responses.create(model=cfg.chat_model, input=prompt).output_text


def lecture_text(title: str, cfg: Settings = default_settings) -> str:
    res = get_collection(cfg).get(where={"title": title}, include=["documents", "metadatas"])
    ordered = sorted(zip(res["documents"], res["metadatas"], strict=True), key=lambda x: float(x[1]["start"]))
    # Chunks overlap; de-duplicating exactly would need the raw transcript. For summarisation,
    # a little repetition is harmless and cheaper than a second read from disk.
    return "\n".join(d for d, _ in ordered)


def _split(text: str, max_chars: int) -> list[str]:
    parts, buf, size = [], [], 0
    for line in text.split("\n"):
        if size + len(line) > max_chars and buf:
            parts.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts


def summarize(title: str, cfg: Settings = default_settings, force: bool = False) -> Summary:
    text = lecture_text(title, cfg)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    cache = cfg.summaries_dir / f"{title}.json"

    if cache.exists() and not force:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("transcript_hash") == digest:
            log.info("Summary cache hit for '%s'", title)
            return Summary(**cached)

    if len(text) <= cfg.summary_map_chars:
        quick = _complete(prompts.QUICK_SUMMARY.format(title=title, text=text), cfg)
        detailed = _complete(prompts.DETAILED_NOTES.format(title=title, text=text), cfg)
    else:
        parts = _split(text, cfg.summary_map_chars)
        log.info("Map-reduce summary for '%s' over %d parts", title, len(parts))
        notes = [
            _complete(prompts.MAP_PROMPT.format(title=title, part=i, total=len(parts), text=p), cfg)
            for i, p in enumerate(parts, start=1)
        ]
        joined = "\n\n".join(notes)
        quick = _complete(prompts.REDUCE_QUICK.format(title=title, text=joined), cfg)
        detailed = _complete(prompts.REDUCE_DETAILED.format(title=title, text=joined), cfg)

    summary = Summary(quick=quick, detailed=detailed, transcript_hash=digest)
    cfg.summaries_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
