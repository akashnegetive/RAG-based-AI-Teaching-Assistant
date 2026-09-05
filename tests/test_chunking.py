from rag_ta.ingestion.chunking import chunk_segments
from rag_ta.models import Segment


def test_merges_small_segments_into_windows(segments):
    chunks = chunk_segments(segments, "L1", target_seconds=45, max_chars=10_000, overlap_seconds=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.end - c.start <= 45 + 5  # at most one segment over the target
        assert c.title == "L1"
    # coverage: first chunk starts at 0, last ends at 200
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 200.0


def test_overlap_repeats_boundary_segments(segments):
    no_overlap = chunk_segments(segments, "L1", target_seconds=45, max_chars=10_000, overlap_seconds=0)
    overlap = chunk_segments(segments, "L1", target_seconds=45, max_chars=10_000, overlap_seconds=10)
    assert len(overlap) >= len(no_overlap)
    # second chunk should start before the first one ends
    assert overlap[1].start < overlap[0].end
    assert overlap[0].end - overlap[1].start <= 10 + 5


def test_max_chars_limit_respected():
    segs = [Segment(i, i + 1, "x" * 300) for i in range(10)]
    chunks = chunk_segments(segs, "L1", target_seconds=1000, max_chars=700, overlap_seconds=0)
    assert all(len(c.text) <= 700 for c in chunks)
    assert len(chunks) == 5  # 2 segments per chunk


def test_single_long_segment_is_still_emitted():
    segs = [Segment(0, 120, "y" * 5000)]
    chunks = chunk_segments(segs, "L1", target_seconds=45, max_chars=1200)
    assert len(chunks) == 1 and chunks[0].text.startswith("y")


def test_empty_and_blank_segments_are_dropped():
    segs = [Segment(0, 1, "  "), Segment(1, 2, ""), Segment(2, 3, "real")]
    chunks = chunk_segments(segs, "L1")
    assert len(chunks) == 1 and chunks[0].text == "real"
    assert chunk_segments([], "L1") == []


def test_indices_and_ids_are_unique(segments):
    chunks = chunk_segments(segments, "Lecture A", target_seconds=30)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)


def test_terminates_when_overlap_exceeds_window():
    # pathological: overlap larger than the window must not loop forever
    segs = [Segment(i, i + 1, f"s{i}") for i in range(50)]
    chunks = chunk_segments(segs, "L1", target_seconds=5, max_chars=10_000, overlap_seconds=100)
    assert chunks and chunks[-1].end == 50
