"""Rerank Hybrid candidates with BAAI/bge-reranker-v2-m3."""

import argparse
import json
import logging
import os
import threading
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from movie_rag.retrieval.dense import build_search_text
from movie_rag.retrieval.hybrid import Retriever

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EXAMPLE_QUERIES = (
    "хоррор",
    "неглупый ужастик",
    "мрачный криминальный триллер",
    "уютное осеннее кино",
    "одиночество",
    "роуд-муви",
    "программист хакер",
    "что-нибудь странное про интернет",
    "медленное тревожное кино про одиночество",
    "про человека, которому тяжело вернуться к нормальной жизни после войны",
    "мрачная история о разваливающихся отношениях",
    "дорожное кино про дружбу",
)
logger = logging.getLogger(__name__)


def build_rerank_text(movie: dict) -> str:
    """Use the same title/what/about/why passage as Dense; exclude not_for."""
    return build_search_text(movie)


class Scorer(Protocol):
    device: str

    def score(self, pairs: list[list[str]]) -> list[float]: ...


class TransformersScorer:
    """Batched query-passage classification using the model's trained head."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        batch_size: int = 4,
        max_length: int = 512,
        device: str | None = None,
    ):
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be >= 1")
        if device not in (None, "cpu", "cuda"):
            raise ValueError("device must be cpu, cuda, or None (automatic)")
        # Avoid Transformers' unrelated background conversion worker for .bin
        # models, as in the existing Dense loader. Lazy imports keep tests light.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading %s on %s (%s)", model_name, self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def score(self, pairs: list[list[str]]) -> list[float]:
        import torch

        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                inputs = self.tokenizer(
                    pairs[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.reshape(-1)
                # Sigmoid maps relevance logits to [0, 1], not calibrated odds.
                scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
        return scores


class Reranker:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        scorer: Scorer | None = None,
        batch_size: int = 4,
        max_length: int = 512,
        device: str | None = None,
    ):
        self._scorer = (
            scorer
            if scorer is not None
            else TransformersScorer(
                model_name=model_name,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            )
        )
        self.device = self._scorer.device
        # Streamlit shares this resource across sessions, including its tokenizer.
        self._score_lock = threading.Lock()

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not query.strip() or not candidates or top_k == 0:
            return []
        unique = {}
        for rank, movie in enumerate(candidates, start=1):
            if movie["url"] not in unique:
                unique[movie["url"]] = dict(movie, hybrid_rank=rank)
        results = list(unique.values())
        pairs = [[query, build_rerank_text(movie)] for movie in results]
        with self._score_lock:
            scores = np.asarray(self._scorer.score(pairs), dtype=float).reshape(-1)
        if len(scores) != len(results) or not np.isfinite(scores).all():
            raise ValueError("scorer must return one finite score per candidate")
        for movie, score in zip(results, scores, strict=True):
            movie["reranker_score"] = float(score)
        results.sort(key=lambda m: (-m["reranker_score"], m["hybrid_rank"], m["url"]))
        return results[:top_k]


class RerankedHybridRetriever:
    def __init__(
        self, hybrid_retriever: Retriever, reranker: Reranker, candidate_k: int = 20
    ):
        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not query.strip() or top_k == 0:
            return []
        candidates = self.hybrid_retriever.search(query, top_k=self.candidate_k)
        return self.reranker.rerank(query, candidates[: self.candidate_k], top_k=top_k)


def main() -> None:
    from movie_rag.retrieval.bm25 import DEFAULT_CORPUS, BM25Retriever, load_movies
    from movie_rag.retrieval.dense import DEFAULT_CACHE, DenseRetriever
    from movie_rag.retrieval.hybrid import HybridRetriever

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Omit to compare 12 example queries")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4, help="Reranker batch size")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Reranker device")
    parser.add_argument("--output-json", type=Path, help="Save results and timings")
    args = parser.parse_args()
    if (
        args.candidate_k < 1
        or args.top_k < 0
        or args.batch_size < 1
        or args.max_length < 1
    ):
        parser.error(
            "candidate-k/batch-size/max-length must be >= 1; top-k must be >= 0"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    started = perf_counter()
    movies = load_movies(args.corpus)
    hybrid = HybridRetriever(
        BM25Retriever(movies), DenseRetriever(movies, cache_dir=args.cache_dir)
    )
    reranker = Reranker(
        batch_size=args.batch_size, max_length=args.max_length, device=args.device
    )
    setup_ms = (perf_counter() - started) * 1000
    report = {
        "model": MODEL_NAME,
        "device": reranker.device,
        "corpus": str(args.corpus),
        "corpus_size": len(movies),
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "setup_ms": setup_ms,
        "queries": [],
    }
    print(
        f"Corpus: {len(movies)} movies; reranker: {reranker.device}; setup: {setup_ms:.0f} ms"
    )
    for query in [args.query] if args.query is not None else EXAMPLE_QUERIES:
        start = perf_counter()
        candidates = hybrid.search(query, top_k=args.candidate_k) if args.top_k else []
        hybrid_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        results = reranker.rerank(query, candidates, top_k=args.top_k)
        rerank_ms = (perf_counter() - start) * 1000
        report["queries"].append(
            {
                "query": query,
                "hybrid_ms": hybrid_ms,
                "rerank_ms": rerank_ms,
                "hybrid_candidates": candidates,
                "reranked": results,
            }
        )
        print(f"\nQUERY: {query}\n\nHYBRID")
        if not candidates:
            print("Нет результатов.")
        for rank, movie in enumerate(candidates[: args.top_k], start=1):
            print(
                f"{rank}. {movie['title']} ({movie['year']}); RRF: {movie['rrf_score']:.6f}"
            )
        print("\nRERANKED")
        if not results:
            print("Нет результатов.")
        for rank, movie in enumerate(results, start=1):
            print(
                f"{rank}. {movie['title']} ({movie['year']}); "
                f"reranker score: {movie['reranker_score']:.6f}; "
                f"hybrid rank = {movie['hybrid_rank']} -> reranked rank = {rank}"
            )
        print(f"Latency: Hybrid {hybrid_ms:.1f} ms; reranker {rerank_ms:.1f} ms")
    if args.output_json:
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
