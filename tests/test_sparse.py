from rag_ta.retrieval.sparse import SparseIndex, tokenize


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The Kadane's algorithm is O(n)") == ["kadane", "algorithm", "o", "n"]


def test_search_exact_term_wins():
    idx = SparseIndex(
        ids=["1", "2", "3"],
        documents=["ridge regression adds an L2 penalty", "lasso uses L1 penalty", "dynamic programming memoization"],
        metadatas=[{"title": "ML"}, {"title": "ML"}, {"title": "DSA"}],
    )
    res = idx.search("what is memoization", top_k=5)
    assert res[0][0] == "3"


def test_title_filter():
    idx = SparseIndex(
        ["1", "2", "3"],
        ["penalty term", "penalty term", "unrelated graph traversal"],
        [{"title": "A"}, {"title": "B"}, {"title": "B"}],
    )
    assert [i for i, _ in idx.search("penalty", title="B")] == ["2"]


def test_empty_index():
    assert SparseIndex([], [], []).search("anything") == []
