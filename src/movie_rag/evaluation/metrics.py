"""Recall is the fraction of labeled positives found within the first K hits."""

from collections.abc import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    if type(k) is not int or k < 0:
        raise ValueError("k must be an integer >= 0")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    # Slice before deduplicating: repeated hits still occupy ranking positions.
    return len(relevant.intersection(retrieved_ids[:k])) / len(relevant)
