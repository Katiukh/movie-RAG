"""Adapt the application's LLM relevance filter to the common recall benchmark."""

from movie_rag.evaluation.runner import Retriever
from movie_rag.generation import Recommender
from movie_rag.generation.recommender import MAX_CANDIDATES


class LLMFilteredRetriever:
    def __init__(self, retriever: Retriever, recommender: Recommender):
        self.retriever = retriever
        self.recommender = recommender

    def search(self, query: str, *, top_k: int) -> list[dict]:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not query.strip() or top_k == 0:
            return []
        # Metric cutoffs must not change the application's ten-candidate budget.
        candidates = list(self.retriever.search(query, top_k=MAX_CANDIDATES))[
            :MAX_CANDIDATES
        ]
        selected = self.recommender.recommend(query=query, candidates=candidates)
        return selected[:top_k]
