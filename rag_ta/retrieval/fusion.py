"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = Σ_r 1 / (k + rank_r(d))   over each ranking r the document appears in.
Rank-based, so it needs no score normalisation between BM25 and cosine distance.
"""

from __future__ import annotations


def rrf(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float, list[str]]]:
    """rankings: {"dense": [id, id, ...], "sparse": [...]} (each already sorted best-first).

    Returns [(id, fused_score, [source names])] sorted desc.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, ids in rankings.items():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(doc_id, []).append(name)
    return sorted(((d, s, sources[d]) for d, s in scores.items()), key=lambda x: x[1], reverse=True)
