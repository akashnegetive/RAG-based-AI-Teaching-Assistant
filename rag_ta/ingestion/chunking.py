"""Timestamp-aware chunking.

Whisper returns very short segments (often a single sentence, 2–8 s). Embedding those
individually gives poor retrieval: each vector carries too little context. We merge
consecutive segments into windows of ~N seconds / M characters, with a time-based overlap
so that a concept explained across a boundary is still captured by at least one chunk.
"""

from __future__ import annotations

from rag_ta.models import Chunk, Segment


def chunk_segments(
    segments: list[Segment],
    title: str,
    target_seconds: float = 45.0,
    max_chars: int = 1200,
    overlap_seconds: float = 10.0,
) -> list[Chunk]:
    segs = [s for s in segments if s.text and s.text.strip()]
    if not segs:
        return []

    chunks: list[Chunk] = []
    i = 0
    n = len(segs)

    while i < n:
        window: list[Segment] = []
        window_start = segs[i].start
        chars = 0
        j = i
        while j < n:
            seg = segs[j]
            seg_len = len(seg.text) + 1
            duration = seg.end - window_start
            # Always take at least one segment; then stop when a limit is exceeded.
            if window and (duration > target_seconds or chars + seg_len > max_chars):
                break
            window.append(seg)
            chars += seg_len
            j += 1

        text = " ".join(s.text.strip() for s in window)
        chunks.append(Chunk(title=title, index=len(chunks), start=window[0].start, end=window[-1].end, text=text))

        if j >= n:
            break

        # Overlap: rewind to the first segment that starts within `overlap_seconds` of the
        # window end, but never rewind to (or before) where this window started.
        window_end = window[-1].end
        k = j - 1
        while k > i and segs[k].start > window_end - overlap_seconds:
            k -= 1
        next_i = k + 1 if k + 1 > i else i + 1
        i = max(next_i, i + 1)

    return chunks


def segments_from_json(data: dict) -> tuple[str, list[Segment]]:
    """Read the transcript JSON format produced by `transcribe.py`."""
    title = data.get("title") or (data["chunks"][0]["title"] if data.get("chunks") else "untitled")
    raw = data.get("segments") or data.get("chunks") or []
    return title, [Segment(start=float(r["start"]), end=float(r["end"]), text=r["text"]) for r in raw]
