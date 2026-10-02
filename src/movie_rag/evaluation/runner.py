"""Backend-independent evaluation with macro averages and per-query details."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from movie_rag.evaluation.dataset import (
    QUERY_TYPES,
    dataset_statistics,
    validate_eval_dataset,
)
from movie_rag.evaluation.metrics import recall_at_k

DEFAULT_KS = (1, 3, 5, 10, 20)


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int) -> Iterable[Any]: ...


def movie_id(hit: dict) -> str:
    return hit["url"]


def evaluate_retriever(
    retriever: Retriever,
    eval_dataset: Sequence[dict],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    id_getter: Callable[[Any], str] = movie_id,
) -> dict:
    """Search once per query at max(ks); do not suppress backend failures.

    Existing backends return movie dictionaries identified by URL. Supply
    id_getter to adapt other results to the same gold document IDs. Corpus
    membership is checked by load_eval_dataset before running evaluation.
    """
    ks = tuple(ks)
    if (
        not ks
        or any(type(k) is not int or k < 1 for k in ks)
        or len(set(ks)) != len(ks)
    ):
        raise ValueError("ks must contain distinct positive integers")
    records = list(eval_dataset)
    validate_eval_dataset(records)
    depth = max(ks)
    per_query = []
    for row in records:
        hits = list(retriever.search(row["query"], top_k=depth))[:depth]
        ids = [id_getter(hit) for hit in hits]
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError(
                f"{row['query_id']}: retriever returned an invalid document ID"
            )
        per_query.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "query_type": row["query_type"],
                "relevant_ids": list(row["relevant_ids"]),
                "retrieved_ids": ids,
                **{f"recall@{k}": recall_at_k(ids, row["relevant_ids"], k) for k in ks},
            }
        )

    def aggregate(rows: list[dict]) -> dict:
        return {
            "count": len(rows),
            **{
                f"recall@{k}": sum(row[f"recall@{k}"] for row in rows) / len(rows)
                if rows
                else None
                for k in ks
            },
        }

    return {
        "ks": list(ks),
        "dataset_statistics": dataset_statistics(records),
        "overall": aggregate(per_query),
        "by_query_type": {
            kind: aggregate([row for row in per_query if row["query_type"] == kind])
            for kind in QUERY_TYPES
        },
        "per_query": per_query,
    }
