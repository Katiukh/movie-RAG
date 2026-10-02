import copy
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module():
    return importlib.import_module("movie_rag.retrieval.reranker")


def movie(name):
    return {
        "title": name,
        "year": 2000,
        "url": f"https://example.com/{name}",
        "what": " драма ",
        "about": " друзья ",
        "why": " дорога ",
        "not_for": "НЕ ИСПОЛЬЗОВАТЬ",
        "rrf_score": 0.03,
        "bm25_rank": 1,
        "dense_rank": None,
        "bm25_score": 7.2,
        "dense_score": None,
    }


class FakeScorer:
    device = "cpu"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, pairs):
        self.calls.append(copy.deepcopy(pairs))
        return self.scores


class FakeHybrid:
    def __init__(self, movies):
        self.movies = movies
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.movies[:top_k]


def test_reranking_preserves_metadata_original_rank_and_inputs(module):
    candidates = [movie("A"), movie("B"), movie("C")]
    before = copy.deepcopy(candidates)
    scorer = FakeScorer([0.3, 0.2, 0.9])
    results = module.Reranker(scorer=scorer).rerank("query", candidates, top_k=2)
    assert [r["title"] for r in results] == ["C", "A"]
    assert [r["hybrid_rank"] for r in results] == [3, 1]
    assert [r["reranker_score"] for r in results] == pytest.approx([0.9, 0.3])
    assert results[0]["rrf_score"] == 0.03
    assert results[0]["bm25_score"] == 7.2
    assert results[0]["dense_rank"] is None
    assert scorer.calls == [
        [
            ["query", "A драма друзья дорога"],
            ["query", "B драма друзья дорога"],
            ["query", "C драма друзья дорога"],
        ]
    ]
    results[0]["title"] = "changed"
    assert candidates == before


def test_text_uses_only_dense_fields_and_handles_missing_values(module):
    assert module.build_rerank_text(movie(" Film ")) == "Film драма друзья дорога"
    assert (
        module.build_rerank_text(
            {"title": " X ", "what": None, "about": "  ", "not_for": "exclude"}
        )
        == "X"
    )


def test_dedup_by_url_keeps_first_original_rank_not_title(module):
    a, b = movie("A"), movie("B")
    b["title"] = "A"
    scorer = FakeScorer([0.5, 0.9])
    results = module.Reranker(scorer=scorer).rerank("q", [a, dict(a), b])
    assert [r["url"] for r in results] == [b["url"], a["url"]]
    assert [r["hybrid_rank"] for r in results] == [3, 1]
    assert len(scorer.calls[0]) == 2


def test_ties_preserve_hybrid_order(module):
    results = module.Reranker(scorer=FakeScorer([0.5, 0.5])).rerank(
        "q", [movie("Z"), movie("A")]
    )
    assert [r["title"] for r in results] == ["Z", "A"]


@pytest.mark.parametrize("scores", [0.7, [0.7], np.array([[0.7]])])
def test_one_candidate_and_scalar_score(module, scores):
    result = module.Reranker(scorer=FakeScorer(scores)).rerank("q", [movie("A")])
    assert len(result) == 1
    assert result[0]["reranker_score"] == pytest.approx(0.7)
    assert result[0]["hybrid_rank"] == 1


def test_empty_query_candidates_and_zero_top_k_skip_scoring(module):
    scorer = FakeScorer([])
    reranker = module.Reranker(scorer=scorer)
    assert reranker.rerank("q", []) == []
    assert reranker.rerank(" \n", [movie("A")]) == []
    assert reranker.rerank("q", [movie("A")], top_k=0) == []
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("q", [], top_k=-1)
    assert scorer.calls == []


@pytest.mark.parametrize("scores", [[], [0.1, 0.2], [float("nan")], [float("inf")]])
def test_bad_scorer_output_fails_explicitly(module, scores):
    with pytest.raises(ValueError, match="score"):
        module.Reranker(scorer=FakeScorer(scores)).rerank("q", [movie("A")])


def test_wrapper_scores_only_twenty_hybrid_candidates(module):
    hybrid = FakeHybrid([movie(str(i)) for i in range(500)])
    scorer = FakeScorer([i / 20 for i in range(20)])
    retriever = module.RerankedHybridRetriever(hybrid, module.Reranker(scorer=scorer))
    results = retriever.search("q")
    assert hybrid.calls == [("q", 20)]
    assert len(scorer.calls[0]) == 20
    assert [r["title"] for r in results] == ["19", "18", "17", "16", "15"]
    assert len(retriever.search("q", top_k=50)) == 20


def test_wrapper_validates_limits_and_skips_empty_queries(module):
    hybrid = FakeHybrid([movie("A")])
    reranker = module.Reranker(scorer=FakeScorer([0.7]))
    with pytest.raises(ValueError, match="candidate_k"):
        module.RerankedHybridRetriever(hybrid, reranker, candidate_k=0)
    retriever = module.RerankedHybridRetriever(hybrid, reranker, candidate_k=1)
    assert retriever.search(" ") == []
    assert retriever.search("q", top_k=0) == []
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("q", top_k=-1)
    assert hybrid.calls == []
    assert len(retriever.search("q", top_k=5)) == 1


@pytest.mark.parametrize(
    "cuda,device,expected",
    [(False, None, "cpu"), (True, None, "cuda"), (True, "cpu", "cpu")],
)
def test_transformers_backend_batches_pairs_and_normalizes_logits(
    module, monkeypatch, cuda, device, expected
):
    import torch

    calls = {"batches": [], "dtype": None}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

    class Inputs(dict):
        def to(self, target):
            assert target == expected
            return self

    class Tokenizer:
        def __call__(self, pairs, **kwargs):
            assert kwargs == {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 512,
            }
            calls["batches"].append(pairs)
            return Inputs(count=len(pairs))

    class Model:
        def to(self, target):
            assert target == expected
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def __call__(self, count, return_dict):
            assert return_dict is True
            assert torch.is_inference_mode_enabled()
            return SimpleNamespace(logits=torch.zeros((count, 1)))

    def load_tokenizer(name):
        assert name == "BAAI/bge-reranker-v2-m3"
        return Tokenizer()

    def load_model(name, **kwargs):
        assert name == "BAAI/bge-reranker-v2-m3"
        calls["dtype"] = kwargs["dtype"]
        return Model()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForSequenceClassification=SimpleNamespace(
                from_pretrained=load_model
            ),
        ),
    )
    scorer = module.TransformersScorer(batch_size=2, device=device)
    pairs = [["q", str(i)] for i in range(3)]
    assert scorer.score(pairs) == [0.5, 0.5, 0.5]
    assert calls["batches"] == [pairs[:2], pairs[2:]]
    assert calls["dtype"] == (torch.float16 if expected == "cuda" else torch.float32)
    assert calls["eval"] is True


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"max_length": 0}, {"device": "invalid"}]
)
def test_invalid_inference_configuration_before_loading(module, kwargs):
    with pytest.raises(ValueError):
        module.TransformersScorer(**kwargs)


def test_cli_compares_twelve_queries_and_exports_timings(
    module, monkeypatch, capsys, tmp_path
):
    import json

    from movie_rag.retrieval import bm25, dense

    class Model:
        device = "cpu"

        def score(self, pairs):
            return [0.7] * len(pairs)

    monkeypatch.setattr(module, "TransformersScorer", lambda **kwargs: Model())
    monkeypatch.setattr(bm25, "load_movies", lambda path: [dict(movie("A"), score=0.5)])
    monkeypatch.setattr(bm25, "BM25Retriever", lambda movies: FakeHybrid(movies))
    monkeypatch.setattr(
        dense, "DenseRetriever", lambda movies, **kwargs: FakeHybrid(movies)
    )
    output_path = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["reranker", "--output-json", str(output_path)])
    module.main()
    out = capsys.readouterr().out
    assert out.count("\nQUERY:") == 12
    assert out.count("\nHYBRID\n") == 12
    assert out.count("\nRERANKED\n") == 12
    assert "hybrid rank = 1 -> reranked rank = 1" in out
    data = json.loads(output_path.read_text())
    assert len(data["queries"]) == 12
    assert all(q["rerank_ms"] >= 0 for q in data["queries"])
    assert all(len(q["reranked"]) == 1 for q in data["queries"])


@pytest.mark.parametrize(
    "args", [["--candidate-k", "0"], ["--top-k", "-1"], ["--batch-size", "0"]]
)
def test_cli_rejects_bad_limits_before_model_loading(module, monkeypatch, args):
    monkeypatch.setattr(
        module, "TransformersScorer", lambda **kw: pytest.fail("loaded model")
    )
    monkeypatch.setattr(sys, "argv", ["reranker", *args])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
