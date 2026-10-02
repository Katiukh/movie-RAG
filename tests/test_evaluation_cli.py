import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from movie_rag.evaluation.candidates import generate_candidates

ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path):
    corpus = tmp_path / "movies.jsonl"
    dataset = tmp_path / "gold.jsonl"
    movies = [
        {
            "url": "a",
            "title": "История",
            "what": "драма",
            "about": "семья и дружба",
            "why": "",
        }
    ]
    rows = [
        {
            "query_id": "q001",
            "query": "семья",
            "query_type": "paraphrase",
            "relevant_ids": ["a"],
        }
    ]
    corpus.write_text(
        json.dumps(movies[0], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dataset.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    return corpus, dataset


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "movie_rag.evaluation", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_bm25_cli_writes_query_rows_summary_and_run_identity(tmp_path):
    corpus, dataset = inputs(tmp_path)
    output, summary = tmp_path / "results.jsonl", tmp_path / "summary.json"
    result = run_cli(
        "--corpus",
        corpus,
        "--dataset",
        dataset,
        "--retriever",
        "bm25",
        "--output",
        output,
        "--summary-output",
        summary,
    )
    assert result.returncode == 0, result.stderr
    assert "Queries: 1" in result.stdout
    assert "paraphrase (1)" in result.stdout
    assert "near_exact (0)" in result.stdout
    assert "Recall@10: 1.00" in result.stdout
    assert "Recall@20: 1.00" in result.stdout
    row = json.loads(output.read_text())
    assert row["retrieved_ids"] == ["a"]
    assert row["recall@1"] == 1.0
    assert row["recall@20"] == 1.0
    report = json.loads(summary.read_text())
    assert report["ks"] == [1, 3, 5, 10, 20]
    assert report["overall"]["recall@1"] == 1.0
    assert report["run"]["retriever"] == "bm25"
    assert len(report["run"]["indexed_corpus_sha256"]) == 64
    assert len(report["run"]["dataset_sha256"]) == 64
    assert report["run"]["corpus_size"] == 1
    assert report["dataset_statistics"]["queries_with_one_relevant_document"] == 1


def test_validate_only_does_not_write_artifacts_or_load_dense(tmp_path):
    corpus, dataset = inputs(tmp_path)
    output = tmp_path / "results.jsonl"
    result = run_cli(
        "--corpus",
        corpus,
        "--dataset",
        dataset,
        "--retriever",
        "dense",
        "--validate-only",
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    assert "Valid" in result.stdout
    assert not output.exists()


def test_validate_only_reports_multi_positive_statistics(tmp_path):
    corpus, dataset = inputs(tmp_path)
    with corpus.open("a", encoding="utf-8") as target:
        target.write(json.dumps({"url": "b", "title": "Другой фильм"}) + "\n")
    with dataset.open("a", encoding="utf-8") as target:
        target.write(
            json.dumps(
                {
                    "query_id": "q002",
                    "query": "драмы",
                    "query_type": "semantic",
                    "relevant_ids": ["a", "b"],
                }
            )
            + "\n"
        )
    result = run_cli("--corpus", corpus, "--dataset", dataset, "--validate-only")
    assert result.returncode == 0, result.stderr
    assert "queries with 1 relevant document: 1" in result.stdout
    assert "queries with 2+ relevant documents: 1" in result.stdout
    assert "average relevant documents per query: 1.50" in result.stdout


def test_invalid_labels_fail_before_dense_initialization(tmp_path):
    corpus, dataset = inputs(tmp_path)
    dataset.write_text(
        json.dumps(
            {
                "query_id": "qBAD",
                "query": "test",
                "query_type": "semantic",
                "relevant_ids": ["missing"],
            }
        )
    )
    result = run_cli("--corpus", corpus, "--dataset", dataset, "--retriever", "dense")
    assert result.returncode == 2
    assert "qBAD" in result.stderr and "missing" in result.stderr
    assert "Loading" not in result.stderr


@pytest.mark.parametrize(
    "target", ["gold", "corpus", "same_outputs", "canonical_gold", "candidates"]
)
def test_outputs_cannot_overwrite_inputs_or_each_other(tmp_path, target):
    corpus, dataset = inputs(tmp_path)
    output = {
        "gold": dataset,
        "corpus": corpus,
        "canonical_gold": ROOT / "data/eval/retrieval_eval.jsonl",
        "candidates": ROOT / "data/eval/retrieval_eval_candidates.jsonl",
    }.get(target, tmp_path / "out")
    summary = output if target == "same_outputs" else tmp_path / "summary"
    original = dataset.read_bytes(), corpus.read_bytes()
    result = run_cli(
        "--corpus",
        corpus,
        "--dataset",
        dataset,
        "--output",
        output,
        "--summary-output",
        summary,
    )
    assert result.returncode == 2
    assert (dataset.read_bytes(), corpus.read_bytes()) == original


def test_output_hardlink_cannot_overwrite_gold(tmp_path):
    corpus, dataset = inputs(tmp_path)
    alias = tmp_path / "gold-hardlink.jsonl"
    alias.hardlink_to(dataset)
    original = dataset.read_bytes()
    result = run_cli(
        "--corpus",
        corpus,
        "--dataset",
        dataset,
        "--output",
        alias,
        "--summary-output",
        tmp_path / "summary.json",
    )
    assert result.returncode == 2
    assert dataset.read_bytes() == original


def test_candidates_are_deterministic_corpus_only_starters_and_do_not_touch_movies():
    movies = [
        {"url": str(i), "title": "name", "about": "об истории " + "x" * i}
        for i in range(60)
    ]
    rows = generate_candidates(movies, count=50)
    assert rows == generate_candidates(movies, count=50)
    assert len(rows) == 50
    assert len({row["relevant_ids"][0] for row in rows}) == 50
    assert all(row["query_type"] == "near_exact" and row["notes"] for row in rows)
    assert movies[0] == {"url": "0", "title": "name", "about": "об истории "}


def test_candidate_cli_refuses_gold_and_existing_files(tmp_path):
    corpus, gold = inputs(tmp_path)
    before = gold.read_bytes()
    for output in [gold, ROOT / "data/eval/retrieval_eval.jsonl", corpus]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "movie_rag.evaluation.candidates",
                "--corpus",
                str(corpus),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
    assert gold.read_bytes() == before


def test_candidate_cli_creates_valid_jsonl_without_search(tmp_path):
    corpus, _ = inputs(tmp_path)
    output = tmp_path / "candidates.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "movie_rag.evaluation.candidates",
            "--corpus",
            str(corpus),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["relevant_ids"] == ["a"]


def test_reranker_cli_evaluates_real_wrapper_and_records_inference_settings(
    tmp_path, monkeypatch
):
    from movie_rag.evaluation import __main__ as cli
    from movie_rag.retrieval import dense, reranker

    corpus, dataset = inputs(tmp_path)
    corpus.write_text(
        "".join(
            json.dumps(
                {
                    "url": value,
                    "title": value,
                    "what": "драма",
                    "about": "семья",
                    "why": "",
                }
            )
            + "\n"
            for value in ["a", "b"]
        )
    )
    dataset.write_text(
        json.dumps(
            {
                "query_id": "q001",
                "query": "семья",
                "query_type": "semantic",
                "relevant_ids": ["b"],
            }
        )
    )

    class EmbeddingModel:
        device = "cpu"

        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0]] * len(texts), dtype=np.float32)

    class Scorer:
        def __init__(self, *, model_name, batch_size, max_length, device):
            assert model_name == "BAAI/bge-reranker-v2-m3"
            assert (batch_size, max_length, device) == (5, 256, "cpu")
            self.device = device

        def score(self, pairs):
            return [0.9 if passage.startswith("b ") else 0.1 for _, passage in pairs]

    monkeypatch.setattr(dense, "load_embedding_model", lambda name: EmbeddingModel())
    monkeypatch.setattr(reranker, "TransformersScorer", Scorer)
    output, summary = tmp_path / "results.jsonl", tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "--retriever",
            "reranker",
            "--corpus",
            str(corpus),
            "--dataset",
            str(dataset),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--reranker-batch-size",
            "5",
            "--reranker-max-length",
            "256",
            "--reranker-device",
            "cpu",
        ],
    )
    cli.main()
    row = json.loads(output.read_text())
    assert row["retrieved_ids"] == ["b", "a"]
    assert row["recall@1"] == 1.0
    config = json.loads(summary.read_text())["run"]
    assert config["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert config["reranker_batch_size"] == 5
    assert config["reranker_max_length"] == 256
    assert config["reranker_device"] == "cpu"
    assert config["candidate_k"] == config["rerank_candidate_k"] == 20
