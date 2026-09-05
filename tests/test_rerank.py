from rag_ta.retrieval.rerank import NoopReranker
from tests.conftest import make_chunk


def test_noop_normalises_scores_to_unit():
    chunks = [make_chunk("a", "t", score=0.05), make_chunk("b", "t", score=0.025)]
    out = NoopReranker().rerank("q", chunks)
    assert out[0].score == 1.0 and abs(out[1].score - 0.5) < 1e-9


def test_noop_handles_empty():
    assert NoopReranker().rerank("q", []) == []
