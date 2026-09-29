import copy
import importlib
import sys

import pytest


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def movie(name, score=0.5, *, title=None):
    return {
        "title": title or name,
        "year": 2000,
        "url": f"https://example.com/{name}",
        "what": "драма",
        "about": "люди",
        "why": "история",
        "score": score,
    }


@pytest.fixture
def hybrid_module():
    return importlib.import_module("movie_rag.retrieval.hybrid")


def test_rrf_fuses_by_url_preserves_metadata_and_does_not_mutate(hybrid_module):
    lexical = [movie("A", 8.4), movie("B", 6.0), movie("C", 3.42)]
    semantic = [movie("C", 0.74), movie("D", 0.72), movie("A", 0.70)]
    before = copy.deepcopy((lexical, semantic))
    retriever = hybrid_module.HybridRetriever(
        FakeRetriever(lexical), FakeRetriever(semantic)
    )
    results = retriever.search("query")
    assert [r["title"] for r in results] == ["C", "A", "D", "B"]
    assert [r["rrf_score"] for r in results] == pytest.approx(
        [
            0.032266458495966696,
            0.032266458495966696,
            0.016129032258064516,
            0.016129032258064516,
        ]
    )
    assert results[0]["bm25_rank"] == 3
    assert results[0]["dense_rank"] == 1
    assert results[0]["bm25_score"] == 3.42
    assert results[0]["dense_score"] == 0.74
    assert results[2]["bm25_rank"] is None
    assert results[2]["bm25_score"] is None
    assert results[3]["dense_rank"] is None
    assert results[3]["dense_score"] is None
    assert results[0]["what"] == "драма"
    assert retriever.search("query") == results
    results[0]["title"] = "changed"
    assert (lexical, semantic) == before


@pytest.mark.parametrize(
    "lexical,semantic,expected",
    [
        ([], ["A", "B", "C"], ["A", "B", "C"]),
        (["A", "B", "C"], [], ["A", "B", "C"]),
        ([], [], []),
        (["A"], ["B"], ["B", "A"]),
    ],
)
def test_empty_and_disjoint_rankings(hybrid_module, lexical, semantic, expected):
    retriever = hybrid_module.HybridRetriever(
        FakeRetriever([movie(n) for n in lexical]),
        FakeRetriever([movie(n) for n in semantic]),
    )
    assert [r["title"] for r in retriever.search("query")] == expected


def test_same_title_different_urls_remain_distinct(hybrid_module):
    retriever = hybrid_module.HybridRetriever(
        FakeRetriever([movie("A", title="same")]),
        FakeRetriever([movie("B", title="same")]),
    )
    assert len(retriever.search("query")) == 2


def test_duplicate_url_in_one_ranking_only_contributes_once(hybrid_module):
    retriever = hybrid_module.HybridRetriever(
        FakeRetriever([movie("A", 9), movie("A", 8), movie("B", 7)]),
        FakeRetriever([]),
    )
    results = retriever.search("query")
    assert [r["title"] for r in results] == ["A", "B"]
    assert results[0]["rrf_score"] == pytest.approx(1 / 61)
    assert results[0]["bm25_score"] == 9
    assert results[1]["bm25_rank"] == 3


def test_candidate_depth_grows_with_top_k_and_limits_final_results(hybrid_module):
    lexical = FakeRetriever([movie(str(i)) for i in range(30)])
    semantic = FakeRetriever([])
    retriever = hybrid_module.HybridRetriever(lexical, semantic, candidate_k=2)
    assert len(retriever.search("query", top_k=5)) == 5
    assert lexical.calls == [("query", 5)]
    assert semantic.calls == [("query", 5)]
    assert len(retriever.search("query", top_k=1)) == 1
    assert lexical.calls[-1] == ("query", 2)


def test_default_candidate_depth_and_custom_rrf_k(hybrid_module):
    lexical = FakeRetriever([movie("A")])
    semantic = FakeRetriever([movie("A")])
    retriever = hybrid_module.HybridRetriever(lexical, semantic, rrf_k=10)
    assert retriever.search("query")[0]["rrf_score"] == pytest.approx(2 / 11)
    assert lexical.calls == [("query", 20)]
    assert semantic.calls == [("query", 20)]


def test_blank_query_zero_and_negative_top_k(hybrid_module):
    lexical, semantic = FakeRetriever([]), FakeRetriever([])
    retriever = hybrid_module.HybridRetriever(lexical, semantic)
    assert retriever.search(" \n ") == []
    assert retriever.search("query", top_k=0) == []
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("query", top_k=-1)
    assert lexical.calls == semantic.calls == []


@pytest.mark.parametrize(
    "kwargs", [{"candidate_k": 0}, {"candidate_k": -1}, {"rrf_k": -1}]
)
def test_invalid_configuration(hybrid_module, kwargs):
    with pytest.raises(ValueError):
        hybrid_module.HybridRetriever(FakeRetriever([]), FakeRetriever([]), **kwargs)


def test_cli_compares_all_benchmark_queries(hybrid_module, monkeypatch, capsys):
    from movie_rag.retrieval import bm25, dense

    monkeypatch.setattr(sys, "argv", ["hybrid", "--top-k", "1"])
    monkeypatch.setattr(bm25, "load_movies", lambda path: [movie("A")])
    monkeypatch.setattr(bm25, "BM25Retriever", lambda movies: FakeRetriever(movies))
    monkeypatch.setattr(
        dense, "DenseRetriever", lambda movies, **kw: FakeRetriever(movies)
    )
    hybrid_module.main()
    output = capsys.readouterr().out
    for query in bm25.EXAMPLE_QUERIES:
        assert f"QUERY: {query}" in output
    assert output.count("\nBM25\n") == 8
    assert output.count("\nDENSE\n") == 8
    assert output.count("\nHYBRID\n") == 8
    assert "RRF:" in output and "BM25 rank:" in output and "Dense rank:" in output


@pytest.mark.parametrize(
    "args", [["--top-k", "-1"], ["--candidate-k", "0"], ["--rrf-k", "-1"]]
)
def test_cli_rejects_invalid_limits_before_loading_model(
    hybrid_module, monkeypatch, args
):
    from movie_rag.retrieval import dense

    def must_not_load(*args, **kwargs):
        pytest.fail("Invalid arguments must be rejected before loading BGE-M3")

    monkeypatch.setattr(sys, "argv", ["hybrid", *args])
    monkeypatch.setattr(dense, "DenseRetriever", must_not_load)
    with pytest.raises(SystemExit) as error:
        hybrid_module.main()
    assert error.value.code == 2
