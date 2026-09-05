import pytest

from rag_ta.models import RetrievedChunk, Segment


@pytest.fixture
def segments() -> list[Segment]:
    # 40 segments, 5 s each, 200 s total
    return [
        Segment(start=i * 5.0, end=i * 5.0 + 5.0, text=f"sentence number {i} about topic {i // 8}") for i in range(40)
    ]


def make_chunk(cid: str, text: str, title: str = "L1", start: float = 0.0, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(id=cid, title=title, index=0, start=start, end=start + 10, text=text, score=score)
