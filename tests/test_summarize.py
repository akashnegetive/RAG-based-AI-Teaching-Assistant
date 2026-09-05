from rag_ta.llm.summarize import _split


def test_split_respects_limit_and_preserves_text():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(100))
    parts = _split(text, 1000)
    assert len(parts) > 1
    assert all(len(p) <= 1000 + 60 for p in parts)
    assert "\n".join(parts) == text
