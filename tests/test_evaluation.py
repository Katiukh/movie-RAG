import copy
import json

import pytest

from movie_rag.evaluation.dataset import (
    dataset_statistics,
    load_eval_dataset,
    validate_eval_dataset,
)
from movie_rag.evaluation.metrics import recall_at_k
from movie_rag.evaluation.runner import evaluate_retriever


def record(**changes):
    return dict(
        query_id="q001",
        query="история о семье",
        relevant_ids=["a"],
        query_type="paraphrase",
        notes="about: семья",
        **changes,
    )


@pytest.mark.parametrize(
    "ranked,positives,k,want",
    [
        (["x", "a", "b"], ["a", "b"], 1, 0.0),
        (["x", "a", "b"], ["a", "b"], 2, 0.5),
        (["x", "a", "b"], ["a", "b"], 10, 1.0),
        (["a", "a", "b"], ["a", "b"], 2, 0.5),
        ([], ["a"], 10, 0.0),
        (["a"], ["a"], 0, 0.0),
    ],
)
def test_recall_counts_unique_hits_within_original_rank_positions(
    ranked, positives, k, want
):
    assert recall_at_k(ranked, positives, k) == want


@pytest.mark.parametrize("positives,k", [([], 1), (["a"], -1), (["a"], True)])
def test_recall_rejects_undefined_inputs(positives, k):
    with pytest.raises(ValueError):
        recall_at_k([], positives, k)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"query_id": " "}, "query_id"),
        ({"query": "\n "}, "query"),
        ({"query": 12}, "query"),
        ({"relevant_ids": []}, "relevant_ids"),
        ({"relevant_ids": "a"}, "relevant_ids"),
        ({"relevant_ids": [12]}, "relevant_ids"),
        ({"relevant_ids": ["missing"]}, "missing"),
        ({"relevant_ids": ["a", "a"]}, "duplicate"),
        ({"query_type": "hard"}, "query_type"),
    ],
)
def test_validator_reports_query_identity(change, match):
    row = record()
    row.update(change)
    with pytest.raises(ValueError, match=match) as error:
        validate_eval_dataset([row], [{"url": "a"}])
    if row["query_id"].strip():
        assert "q001" in str(error.value)


def test_duplicate_queries_and_empty_dataset_rejected():
    with pytest.raises(ValueError, match="q001.*duplicate"):
        validate_eval_dataset([record(), record()], [{"url": "a"}])
    with pytest.raises(ValueError, match="empty"):
        validate_eval_dataset([], [{"url": "a"}])


def test_dataset_statistics_counts_labels_without_retrieval():
    rows = [
        record(),
        record() | {"query_id": "q002", "relevant_ids": ["a", "b"]},
        record() | {"query_id": "q003", "relevant_ids": ["a", "b", "c"]},
    ]
    assert dataset_statistics(rows) == {
        "queries": 3,
        "queries_with_one_relevant_document": 1,
        "queries_with_multiple_relevant_documents": 2,
        "average_relevant_documents_per_query": 2.0,
    }


def test_jsonl_loader_preserves_rows_and_reports_lines(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n" + json.dumps(record(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert load_eval_dataset(path, [{"url": "a"}]) == [record()]
    path.write_text(json.dumps(record()) + "\n{broken", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:.*JSON"):
        load_eval_dataset(path, [{"url": "a"}])
    path.write_text(
        json.dumps(record() | {"relevant_ids": ["missing"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r":1:.*q001"):
        load_eval_dataset(path, [{"url": "a"}])


class RankedRetriever:
    def search(self, query, *, top_k):
        assert top_k == 20
        return [
            {"url": value}
            for value in ({"first": ["x", "a", "b"], "second": ["a"]}[query])
        ]


def test_runner_macro_averages_and_breakdown_preserve_query_details():
    dataset = [
        record() | {"query": "first", "relevant_ids": ["a", "b"]},
        record() | {"query_id": "q002", "query": "second", "query_type": "semantic"},
    ]
    before = copy.deepcopy(dataset)
    result = evaluate_retriever(RankedRetriever(), dataset)
    assert result["overall"] == {
        "count": 2,
        "recall@1": 0.5,
        "recall@3": 1.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "recall@20": 1.0,
    }
    assert result["by_query_type"]["paraphrase"]["recall@1"] == 0.0
    assert result["by_query_type"]["semantic"]["recall@1"] == 1.0
    assert result["by_query_type"]["near_exact"]["count"] == 0
    assert result["by_query_type"]["near_exact"]["recall@1"] is None
    assert result["per_query"][0]["retrieved_ids"] == ["x", "a", "b"]
    assert result["per_query"][0]["relevant_ids"] == ["a", "b"]
    assert dataset == before


def test_custom_adapter_and_empty_search_results():
    class Retriever:
        def search(self, query, *, top_k):
            assert top_k == 3
            return ["a"] if query == "hit" else []

    rows = [
        record() | {"query": "hit"},
        record() | {"query_id": "q002", "query": "miss"},
    ]
    result = evaluate_retriever(Retriever(), rows, ks=[1, 3], id_getter=lambda hit: hit)
    assert result["overall"] == {"count": 2, "recall@1": 0.5, "recall@3": 0.5}
    assert result["per_query"][1]["retrieved_ids"] == []


@pytest.mark.parametrize("ks", [[], [0], [-1], [1, 1], [True]])
def test_bad_cutoffs_fail_before_search(ks):
    with pytest.raises(ValueError, match="ks"):
        evaluate_retriever(None, [record()], ks=ks)


def test_backend_errors_are_not_misses():
    class Broken:
        def search(self, query, *, top_k):
            raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        evaluate_retriever(Broken(), [record()])
