"""Gold JSONL loading and validation against the current corpus URLs."""

import json
from collections.abc import Iterable
from pathlib import Path

QUERY_TYPES = ("near_exact", "paraphrase", "semantic")
DEFAULT_DATASET = Path("data/eval/retrieval_eval.jsonl")
DEFAULT_CANDIDATES = Path("data/eval/retrieval_eval_candidates.jsonl")


def _validate_record(row: dict, seen: set[str], corpus_ids: set[str] | None) -> None:
    identity = (
        row.get("query_id", "<missing query_id>")
        if isinstance(row, dict)
        else "<unknown>"
    )

    def fail(message: str) -> None:
        raise ValueError(f"{identity}: {message}")

    if not isinstance(row, dict):
        fail("expected a query object")
    if not isinstance(identity, str) or not identity.strip():
        fail("query_id must be a nonempty string")
    if "query_id" not in row:
        fail("query_id is required")
    if identity in seen:
        fail("duplicate query_id")
    seen.add(identity)
    if not isinstance(row.get("query"), str) or not row["query"].strip():
        fail("query must be a nonempty string")
    positives = row.get("relevant_ids")
    if not isinstance(positives, list) or not positives:
        fail("relevant_ids must be a nonempty list")
    if any(not isinstance(value, str) or not value.strip() for value in positives):
        fail("relevant_ids must contain nonempty strings")
    if len(set(positives)) != len(positives):
        fail("duplicate IDs in relevant_ids")
    if corpus_ids is not None and (missing := set(positives) - corpus_ids):
        fail(f"relevant_ids absent from corpus: {', '.join(sorted(missing))}")
    if row.get("query_type") not in QUERY_TYPES:
        fail(f"query_type must be one of {', '.join(QUERY_TYPES)}")


def validate_eval_dataset(
    records: list[dict], movies: Iterable[dict] | None = None
) -> None:
    """Validate shape and, when supplied, membership in the current corpus."""
    if not records:
        raise ValueError("Evaluation dataset is empty")
    corpus_ids = {movie["url"] for movie in movies} if movies is not None else None
    seen: set[str] = set()
    for row in records:
        _validate_record(row, seen, corpus_ids)


def load_eval_dataset(path: Path, movies: Iterable[dict]) -> list[dict]:
    corpus_ids = {movie["url"] for movie in movies}
    records = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                _validate_record(row, seen, corpus_ids)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(row)
    if not records:
        raise ValueError(f"{path}: Evaluation dataset is empty")
    return records


def dataset_statistics(records: list[dict]) -> dict:
    """Describe label cardinality independently of any search results."""
    validate_eval_dataset(records)
    sizes = [len(row["relevant_ids"]) for row in records]
    return {
        "queries": len(records),
        "queries_with_one_relevant_document": sizes.count(1),
        "queries_with_multiple_relevant_documents": sum(size > 1 for size in sizes),
        "average_relevant_documents_per_query": sum(sizes) / len(sizes),
    }
