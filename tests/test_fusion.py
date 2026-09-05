from rag_ta.retrieval.fusion import rrf


def test_doc_in_both_lists_outranks_single_list_docs():
    fused = rrf({"dense": ["a", "b", "c"], "sparse": ["b", "d"]}, k=60)
    ids = [f[0] for f in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}
    b = next(f for f in fused if f[0] == "b")
    assert sorted(b[2]) == ["dense", "sparse"]


def test_scores_follow_formula():
    fused = dict((d, s) for d, s, _ in rrf({"x": ["a"], "y": ["a"]}, k=10))
    assert abs(fused["a"] - 2 / 11) < 1e-9


def test_empty_rankings():
    assert rrf({}) == []
    assert rrf({"dense": [], "sparse": []}) == []
