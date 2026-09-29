"""Combine lexical and semantic rankings using Reciprocal Rank Fusion."""

import argparse
import logging
from pathlib import Path
from typing import Protocol


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict]: ...


def fuse_rankings(
    bm25_results: list[dict], dense_results: list[dict], rrf_k: int = 60
) -> list[dict]:
    """Fuse by URL, copying metadata and counting each URL once per ranking.

    Ranks are one-based positions in the original rankings. A duplicate within
    one ranking keeps its first position and score. Raw scores never affect RRF.
    """
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    candidates: dict[str, dict] = {}
    for source, ranking in (("bm25", bm25_results), ("dense", dense_results)):
        for rank, movie in enumerate(ranking, start=1):
            url = movie["url"]
            if url not in candidates:
                result = dict(movie)
                result.pop("score", None)
                result.update(
                    rrf_score=0.0,
                    bm25_rank=None,
                    bm25_score=None,
                    dense_rank=None,
                    dense_score=None,
                )
                candidates[url] = result
            result = candidates[url]
            if result[f"{source}_rank"] is not None:
                continue
            result[f"{source}_rank"] = rank
            result[f"{source}_score"] = movie["score"]
            result["rrf_score"] += 1.0 / (rrf_k + rank)
    return sorted(
        candidates.values(),
        key=lambda movie: (
            -movie["rrf_score"],
            movie["dense_rank"] or float("inf"),
            movie["bm25_rank"] or float("inf"),
            movie["url"],
        ),
    )


class HybridRetriever:
    """Use existing retrievers without constructing indexes or loading models."""

    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not query.strip() or top_k == 0:
            return []
        depth = max(self.candidate_k, top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=depth)
        dense_results = self.dense_retriever.search(query, top_k=depth)
        return fuse_rankings(bm25_results, dense_results, self.rrf_k)[:top_k]


def main() -> None:
    # Keep model loading and corpus access out of the fusion layer.
    from movie_rag.retrieval.bm25 import (
        DEFAULT_CORPUS,
        EXAMPLE_QUERIES,
        BM25Retriever,
        load_movies,
    )
    from movie_rag.retrieval.dense import DEFAULT_CACHE, DenseRetriever

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query", nargs="?", help="Omit to compare eight benchmark queries"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.top_k < 0 or args.candidate_k < 1 or args.rrf_k < 0 or args.batch_size < 1:
        parser.error("top-k/rrf-k must be >= 0; candidate-k/batch-size must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    movies = load_movies(args.corpus)
    bm25 = BM25Retriever(movies)
    dense = DenseRetriever(
        movies,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        batch_size=args.batch_size,
    )
    depth = max(args.candidate_k, args.top_k)
    print(
        f"Corpus: {args.corpus} ({len(movies)} movies); "
        f"candidate_k={args.candidate_k}, depth={depth}, rrf_k={args.rrf_k}"
    )
    for query in [args.query] if args.query is not None else EXAMPLE_QUERIES:
        print(f"\nQUERY: {query}")
        # Search once per backend so all three columns share exactly the same
        # rankings and comparison does not encode the query a second time.
        lexical = bm25.search(query, top_k=depth) if args.top_k else []
        semantic = dense.search(query, top_k=depth) if args.top_k else []
        hybrid = fuse_rankings(lexical, semantic, args.rrf_k)
        for label, results in (
            ("BM25", lexical),
            ("DENSE", semantic),
            ("HYBRID", hybrid),
        ):
            print(f"\n{label}")
            if not results:
                print("Нет результатов.")
            for rank, movie in enumerate(results[: args.top_k], start=1):
                print(f"{rank}. {movie['title']} ({movie['year']})")
                if label == "HYBRID":
                    print(
                        f"   RRF: {movie['rrf_score']:.6f}; "
                        f"BM25 rank: {movie['bm25_rank']}; "
                        f"Dense rank: {movie['dense_rank']}"
                    )
                    print(
                        f"   BM25 score: {movie['bm25_score']}; "
                        f"Dense score: {movie['dense_score']}"
                    )
                else:
                    print(f"   Score: {movie['score']:.6f}")
                print(f"   URL: {movie['url']}")


if __name__ == "__main__":
    main()
