"""Shared data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    """A raw ASR segment (what Whisper returns)."""

    start: float
    end: float
    text: str


@dataclass
class Chunk:
    """A retrieval unit: several merged segments with a timestamp range."""

    title: str
    index: int
    start: float
    end: float
    text: str

    @property
    def id(self) -> str:
        return f"{self.title}__{self.index:04d}__{int(self.start * 1000)}"

    def metadata(self) -> dict:
        return {"title": self.title, "index": self.index, "start": self.start, "end": self.end}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedChunk:
    id: str
    title: str
    index: int
    start: float
    end: float
    text: str
    score: float = 0.0  # final (fused or reranked) score
    sources: list[str] = field(default_factory=list)  # e.g. ["dense", "sparse"]

    @property
    def timestamp(self) -> str:
        return f"{_fmt(self.start)}–{_fmt(self.end)}"


def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
