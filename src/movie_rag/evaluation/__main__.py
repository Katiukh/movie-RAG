"""Validate the retrieval gold dataset or evaluate an existing search backend."""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from movie_rag.evaluation.dataset import (
    DEFAULT_CANDIDATES,
    DEFAULT_DATASET,
    QUERY_TYPES,
    dataset_statistics,
    load_eval_dataset,
)
from movie_rag.evaluation.runner import DEFAULT_KS, evaluate_retriever
from movie_rag.retrieval.bm25 import (
    DEFAULT_CORPUS,
    BM25Retriever,
    build_search_text,
    load_movies,
)
from movie_rag.retrieval.dense import DEFAULT_CACHE

DEFAULT_OUTPUT = Path("artifacts/eval/retrieval_results.jsonl")
DEFAULT_SUMMARY = Path("artifacts/eval/retrieval_summary.json")


def _same_file(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def _build_retriever(args, movies):
    if args.retriever == "bm25":
        return BM25Retriever(movies)
    from movie_rag.retrieval.dense import DenseRetriever

    dense = DenseRetriever(movies, cache_dir=args.cache_dir, batch_size=args.batch_size)
    if args.retriever == "dense":
        return dense
    from movie_rag.retrieval.hybrid import HybridRetriever

    hybrid = HybridRetriever(
        BM25Retriever(movies), dense, candidate_k=args.candidate_k, rrf_k=args.rrf_k
    )
    if args.retriever == "hybrid":
        return hybrid
    from movie_rag.retrieval.reranker import RerankedHybridRetriever, Reranker

    return RerankedHybridRetriever(
        hybrid,
        Reranker(
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            device=args.reranker_device,
        ),
        candidate_k=args.candidate_k,
    )


def _print_summary(result):
    _print_statistics(result["dataset_statistics"])
    print(f"Queries: {result['overall']['count']}")
    groups = [("Overall", result["overall"])] + [
        (
            f"{kind} ({result['by_query_type'][kind]['count']})",
            result["by_query_type"][kind],
        )
        for kind in QUERY_TYPES
    ]
    for label, group in groups:
        print(f"\n{label}")
        for k in result["ks"]:
            value = group[f"recall@{k}"]
            print(
                f"Recall@{k}: {value:.2f}" if value is not None else f"Recall@{k}: n/a"
            )


def _print_statistics(stats):
    print(f"queries: {stats['queries']}")
    print(
        f"queries with 1 relevant document: {stats['queries_with_one_relevant_document']}"
    )
    print(
        f"queries with 2+ relevant documents: {stats['queries_with_multiple_relevant_documents']}"
    )
    print(
        f"average relevant documents per query: {stats['average_relevant_documents_per_query']:.2f}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--retriever", choices=("bm25", "dense", "hybrid", "reranker"), default="bm25"
    )
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if not args.ks or min(args.ks) < 1 or len(set(args.ks)) != len(args.ks):
        parser.error("--ks must contain distinct positive integers")
    if args.batch_size < 1 or args.candidate_k < 1 or args.rrf_k < 0:
        parser.error("batch-size/candidate-k must be >= 1; rrf-k must be >= 0")
    if args.reranker_batch_size < 1 or args.reranker_max_length < 1:
        parser.error("reranker-batch-size/reranker-max-length must be >= 1")
    protected = {
        path.resolve()
        for path in (args.dataset, args.corpus, DEFAULT_DATASET, DEFAULT_CANDIDATES)
    }
    if not args.validate_only and (
        any(
            _same_file(output, path)
            for output in (args.output, args.summary_output)
            for path in protected
        )
        or _same_file(args.output, args.summary_output)
    ):
        parser.error(
            "Output paths must differ and must not overwrite datasets or corpus"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    started_at_utc = datetime.now(UTC).isoformat()
    try:
        movies = load_movies(args.corpus)
        # Always validate before backend construction (especially model loading).
        dataset = load_eval_dataset(args.dataset, movies)
        dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        if args.validate_only:
            counts = {
                kind: sum(row["query_type"] == kind for row in dataset)
                for kind in QUERY_TYPES
            }
            print(
                f"Valid: {len(dataset)} queries; corpus: {len(movies)} documents; {counts}"
            )
            _print_statistics(dataset_statistics(dataset))
            return
        retriever = _build_retriever(args, movies)
        result = evaluate_retriever(retriever, dataset, ks=args.ks)
        indexed = [
            {"url": movie["url"], "text": build_search_text(movie)} for movie in movies
        ]
        result["run"] = {
            "retriever": args.retriever,
            "corpus": str(args.corpus),
            "corpus_size": len(movies),
            "dataset": str(args.dataset),
            "started_at_utc": started_at_utc,
            "dataset_sha256": dataset_sha256,
            "indexed_corpus_sha256": hashlib.sha256(
                json.dumps(indexed, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "ks": args.ks,
        }
        if args.retriever != "bm25":
            from movie_rag.retrieval.dense import MODEL_NAME

            result["run"].update(
                embedding_model=MODEL_NAME,
                batch_size=args.batch_size,
                cache_dir=str(args.cache_dir),
            )
        if args.retriever in ("hybrid", "reranker"):
            result["run"].update(candidate_k=args.candidate_k, rrf_k=args.rrf_k)
        if args.retriever == "reranker":
            from movie_rag.retrieval.reranker import MODEL_NAME as RERANKER_MODEL_NAME

            result["run"].update(
                reranker_model=RERANKER_MODEL_NAME,
                rerank_candidate_k=args.candidate_k,
                reranker_batch_size=args.reranker_batch_size,
                reranker_max_length=args.reranker_max_length,
                reranker_device=retriever.reranker.device,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as target:
            for row in result["per_query"]:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {key: value for key, value in result.items() if key != "per_query"}
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    _print_summary(result)
    print(f"\nPer-query: {args.output}\nSummary: {args.summary_output}")


if __name__ == "__main__":
    main()
