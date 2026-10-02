import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from movie_rag.evaluation.runner import evaluate_retriever
from movie_rag.generation import Recommender, RecommenderError


def movie(value):
    return {
        "url": value,
        "title": value,
        "what": "драма",
        "about": "семья",
        "why": "",
        "not_for": "",
    }


def gold(query_id="q001", query="семья", relevant_ids=None, kind="semantic"):
    return {
        "query_id": query_id,
        "query": query,
        "query_type": kind,
        "relevant_ids": relevant_ids or ["a"],
    }


def mock_ollama(monkeypatch, selections):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        user = json.loads(kwargs["json"]["messages"][1]["content"])
        content = json.dumps({"selected_ids": selections[user["query"]]})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": content},
            },
        )

    monkeypatch.setattr(requests, "post", post)
    return calls


def test_llm_evaluation_uses_top_ten_once_preserves_order_and_macro_metrics(
    monkeypatch,
):
    from movie_rag.evaluation.llm import LLMFilteredRetriever

    depths = []

    class Reranked:
        def search(self, query, *, top_k):
            depths.append(top_k)
            return [movie(value) for value in ["x", "a", "b", "c"]][:top_k]

    calls = mock_ollama(monkeypatch, {"first": ["b", "a"], "second": []})
    backend = LLMFilteredRetriever(Reranked(), Recommender())
    result = evaluate_retriever(
        backend,
        [
            gold(query="first", relevant_ids=["a", "b"], kind="paraphrase"),
            gold("q002", "second"),
        ],
    )
    assert result["overall"] == {
        "count": 2,
        "recall@1": 0.25,
        "recall@3": 0.5,
        "recall@5": 0.5,
        "recall@10": 0.5,
        "recall@20": 0.5,
    }
    assert result["by_query_type"]["paraphrase"]["recall@10"] == 1.0
    assert result["by_query_type"]["semantic"]["recall@10"] == 0.0
    assert result["per_query"][0]["retrieved_ids"] == ["a", "b"]
    assert result["per_query"][1]["retrieved_ids"] == []
    assert depths == [10, 10]
    assert len(calls) == 2


@pytest.mark.parametrize("ks", [[1], [1, 3, 5, 10, 20], [20]])
def test_metric_cutoff_does_not_change_ten_candidates_sent_to_llm(monkeypatch, ks):
    from movie_rag.evaluation.llm import LLMFilteredRetriever

    class Reranked:
        def search(self, query, *, top_k):
            return [movie(str(i)) for i in range(25)][:top_k]

    calls = mock_ollama(monkeypatch, {"семья": ["9"]})
    result = evaluate_retriever(
        LLMFilteredRetriever(Reranked(), Recommender()),
        [gold(relevant_ids=["9"])],
        ks=ks,
    )
    assert result["overall"][f"recall@{ks[0]}"] == 1.0
    assert result["per_query"][0]["retrieved_ids"] == ["9"]
    assert len(calls) == 1
    user = json.loads(calls[0]["messages"][1]["content"])
    assert [candidate["movie_id"] for candidate in user["candidates"]] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]


def test_llm_eval_propagates_invalid_output_instead_of_scoring_a_miss(monkeypatch):
    from movie_rag.evaluation.llm import LLMFilteredRetriever

    class Reranked:
        def search(self, query, *, top_k):
            return [movie("a")]

    mock_ollama(monkeypatch, {"семья": ["unknown"]})
    with pytest.raises(RecommenderError):
        evaluate_retriever(LLMFilteredRetriever(Reranked(), Recommender()), [gold()])


def cli_inputs(tmp_path):
    corpus, dataset = tmp_path / "movies.jsonl", tmp_path / "gold.jsonl"
    corpus.write_text(
        "".join(json.dumps(movie(str(i))) + "\n" for i in range(25)),
        encoding="utf-8",
    )
    dataset.write_text(json.dumps(gold(relevant_ids=["19"])) + "\n", encoding="utf-8")
    return corpus, dataset


def cli_args(monkeypatch, corpus, dataset, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "--retriever",
            "llm",
            "--corpus",
            str(corpus),
            "--dataset",
            str(dataset),
            *map(str, extra),
        ],
    )


def test_llm_cli_real_pipeline_budget_separate_outputs_and_config(
    tmp_path, monkeypatch, capsys
):
    from movie_rag.evaluation import __main__ as cli
    from movie_rag.retrieval import dense, reranker

    corpus, dataset = cli_inputs(tmp_path)
    scored_counts = []

    class EmbeddingModel:
        device = "cpu"

        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0]] * len(texts), dtype=np.float32)

    class Scorer:
        def __init__(self, **kwargs):
            self.device = "cpu"

        def score(self, pairs):
            scored_counts.append(len(pairs))
            # Distinct scores give a stable, descending numeric order.
            return [float(passage.split()[0]) for _, passage in pairs]

    monkeypatch.setattr(dense, "load_embedding_model", lambda name: EmbeddingModel())
    monkeypatch.setattr(reranker, "TransformersScorer", Scorer)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "75")
    monkeypatch.chdir(tmp_path)
    old_output = tmp_path / "artifacts/eval/retrieval_results.jsonl"
    old_output.parent.mkdir(parents=True)
    old_output.write_text("existing retrieval benchmark\n")
    calls = mock_ollama(monkeypatch, {"семья": ["19"]})
    cli_args(monkeypatch, corpus, dataset, "--cache-dir", tmp_path / "cache")
    cli.main()

    row = json.loads((tmp_path / "artifacts/eval/llm_results.jsonl").read_text())
    report = json.loads((tmp_path / "artifacts/eval/llm_summary.json").read_text())
    assert row["retrieved_ids"] == ["19"]
    assert row["recall@1"] == row["recall@20"] == 1.0
    assert report["overall"]["recall@10"] == 1.0
    assert scored_counts == [20]
    assert len(calls) == 1
    user = json.loads(calls[0]["messages"][1]["content"])
    assert len(user["candidates"]) == 10
    assert user["candidates"][0]["movie_id"] == "19"
    config = report["run"]
    assert config["retriever"] == "llm"
    assert config["llm_model"] == "qwen3:8b"
    assert config["llm_base_url"] == "http://localhost:11434"
    assert config["llm_timeout_seconds"] == 75
    assert config["llm_candidate_k"] == 10
    assert config["candidate_k"] == config["rerank_candidate_k"] == 20
    assert config["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert config["reranker_device"] == "cpu"
    assert len(config["llm_prompt_sha256"]) == 64
    assert len(config["llm_corpus_sha256"]) == 64
    assert old_output.read_text() == "existing retrieval benchmark\n"
    assert "Recall@20: 1.00" in capsys.readouterr().out


def test_llm_validate_only_loads_no_models_or_ollama(tmp_path, monkeypatch, capsys):
    from movie_rag.evaluation import __main__ as cli

    corpus, dataset = cli_inputs(tmp_path)
    monkeypatch.setenv("LLM_BASE_URL", "https://external.invalid")
    cli_args(monkeypatch, corpus, dataset, "--validate-only")
    cli.main()
    assert "Valid: 1 queries" in capsys.readouterr().out


def test_llm_cli_rejects_changed_rerank_budget_before_loading(
    tmp_path, monkeypatch, capsys
):
    from movie_rag.evaluation import __main__ as cli

    corpus, dataset = cli_inputs(tmp_path)
    cli_args(monkeypatch, corpus, dataset, "--candidate-k", "30")
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--candidate-k 20" in capsys.readouterr().err


def test_invalid_llm_config_fails_before_loading_models(tmp_path, monkeypatch, capsys):
    from movie_rag.evaluation import __main__ as cli
    from movie_rag.retrieval import dense

    corpus, dataset = cli_inputs(tmp_path)
    monkeypatch.setenv("LLM_BASE_URL", "https://external.invalid")

    def unexpected_load(name):
        pytest.fail("Dense must not load when LLM config is invalid")

    monkeypatch.setattr(dense, "load_embedding_model", unexpected_load)
    cli_args(monkeypatch, corpus, dataset)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "локальный Ollama" in capsys.readouterr().err
