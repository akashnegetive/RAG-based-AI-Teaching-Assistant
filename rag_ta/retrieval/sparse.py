"""BM25 keyword index over the same chunks Chroma holds.

Dense embeddings are great at paraphrase but weak on exact tokens — variable names,
acronyms, "Kadane", "L1 vs L2". BM25 covers that gap; RRF merges the two rankings.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "was",
    "were",
    "it",
    "this",
    "that",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "be",
    "we",
    "you",
    "i",
    "so",
    "if",
    "then",
    "than",
    "but",
}


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t.endswith("'s"):  # "kadane's" -> "kadane" so the bare name matches
            t = t[:-2]
        if t and t not in _STOP:
            out.append(t)
    return out


class SparseIndex:
    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self._bm25 = BM25Okapi([tokenize(d) for d in documents]) if documents else None

    @classmethod
    def from_collection(cls, collection) -> SparseIndex:
        res = collection.get(include=["documents", "metadatas"])
        return cls(res["ids"], res["documents"], res["metadatas"])

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, top_k: int = 20, title: str | None = None) -> list[tuple[str, float]]:
        """Return [(id, score)] sorted desc. `title` restricts to one lecture."""
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = []
        for i, s in enumerate(scores):
            if s <= 0:
                continue
            if title and self.metadatas[i].get("title") != title:
                continue
            ranked.append((self.ids[i], float(s)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def lookup(self, doc_id: str) -> tuple[str, dict]:
        i = self.ids.index(doc_id)
        return self.documents[i], self.metadatas[i]
