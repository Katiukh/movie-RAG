import copy
import json

import numpy as np
import pytest

from movie_rag.retrieval import dense


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def get_embedding_dimension(self):
        return 2

    def encode(self, texts, *, normalize_embeddings, convert_to_numpy, **kwargs):
        assert normalize_embeddings is True
        assert convert_to_numpy is True
        self.calls.append(copy.deepcopy(texts))
        vectors = {
            "Запад драма дорога дружба": [1.0, 0.0],
            "Космос фантастика планеты будущее": [0.0, 1.0],
            "Сумерки хоррор лес страх": [-1.0, 0.0],
            "дорога": [1.0, 0.0],
            "страх": [-1.0, 0.0],
        }
        return np.asarray(
            [vectors.get(text, [0.6, 0.8]) for text in texts], dtype=np.float32
        )


@pytest.fixture
def model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(dense, "load_embedding_model", lambda name: model)
    return model


@pytest.fixture
def movies():
    return [
        {
            "title": title,
            "year": 2000,
            "what": what,
            "about": about,
            "why": why,
            "not_for": "НЕ ИНДЕКСИРОВАТЬ",
            "url": f"https://vk.ru/wall-1_{i}",
        }
        for i, (title, what, about, why) in enumerate(
            [
                ("Запад", "драма", "дорога", "дружба"),
                ("Космос", "фантастика", "планеты", "будущее"),
                ("Сумерки", "хоррор", "лес", "страх"),
            ]
        )
    ]


def test_search_text(movies):
    assert dense.build_search_text(movies[0]) == "Запад драма дорога дружба"
    assert (
        dense.build_search_text({"title": " Film ", "what": None, "not_for": "hidden"})
        == "Film"
    )


def test_ranked_cosine_results_are_copies_and_only_queries_reencoded(model, movies):
    before = copy.deepcopy(movies)
    retriever = dense.DenseRetriever(movies, cache_dir=None)
    results = retriever.search("дорога", top_k=5)
    assert [m["title"] for m in results] == ["Запад", "Космос", "Сумерки"]
    assert [m["score"] for m in results] == pytest.approx([1.0, 0.0, -1.0])
    assert isinstance(results[0]["score"], float)
    assert results[0]["not_for"] == "НЕ ИНДЕКСИРОВАТЬ"
    results[0]["title"] = "changed"
    assert movies == before
    assert len(retriever.search("страх", top_k=1)) == 1
    assert len(model.calls) == 3
    assert model.calls[1:] == [["дорога"], ["страх"]]


def test_empty_queries_and_limits_do_not_encode(model, movies):
    retriever = dense.DenseRetriever(movies, cache_dir=None)
    assert retriever.search("  ") == []
    assert retriever.search("дорога", top_k=0) == []
    with pytest.raises(ValueError):
        retriever.search("дорога", top_k=-1)
    assert len(model.calls) == 1
    assert dense.DenseRetriever([], cache_dir=None).search("дорога") == []
    assert len(model.calls) == 1


def test_disk_cache_avoids_corpus_reencoding(model, movies, tmp_path):
    dense.DenseRetriever(movies, cache_dir=tmp_path)
    assert len(model.calls) == 1
    reloaded = dense.DenseRetriever(movies, cache_dir=tmp_path)
    assert len(model.calls) == 1
    assert reloaded.search("дорога")[0]["title"] == "Запад"
    assert model.calls[-1] == ["дорога"]
    metadata = json.loads((tmp_path / "bge_m3_metadata.json").read_text())
    assert metadata["model_name"] == "BAAI/bge-m3"
    assert metadata["document_urls"] == [m["url"] for m in movies]
    assert metadata["fingerprint"]


@pytest.mark.parametrize("change", ["text", "order", "url", "model", "force"])
def test_cache_invalidates_on_identity_changes(model, movies, tmp_path, change):
    dense.DenseRetriever(movies, cache_dir=tmp_path)
    kwargs = {}
    if change == "text":
        movies[0]["about"] = "new description"
    elif change == "order":
        movies.reverse()
    elif change == "url":
        movies[0]["url"] = "https://vk.ru/wall-1_99"
    elif change == "model":
        kwargs["model_name"] = "other-model"
    else:
        kwargs["rebuild_cache"] = True
    dense.DenseRetriever(movies, cache_dir=tmp_path, **kwargs)
    assert len(model.calls) == 2


def test_metadata_only_changes_do_not_reencode(model, movies, tmp_path):
    dense.DenseRetriever(movies, cache_dir=tmp_path)
    movies[0]["not_for"] = "new metadata"
    retriever = dense.DenseRetriever(movies, cache_dir=tmp_path)
    assert len(model.calls) == 1
    assert retriever.search("дорога")[0]["not_for"] == "new metadata"


@pytest.mark.parametrize("broken", ["npy", "json", "shape", "nonfinite"])
def test_corrupt_cache_is_rebuilt(model, movies, tmp_path, broken):
    dense.DenseRetriever(movies, cache_dir=tmp_path)
    if broken == "npy":
        (tmp_path / "bge_m3_embeddings.npy").write_bytes(b"broken")
    elif broken == "json":
        (tmp_path / "bge_m3_metadata.json").write_text("{broken")
    else:
        np.save(
            tmp_path / "bge_m3_embeddings.npy",
            np.zeros((1, 3)) if broken == "shape" else np.full((3, 2), np.nan),
        )
    retriever = dense.DenseRetriever(movies, cache_dir=tmp_path)
    assert len(model.calls) == 2
    assert retriever.search("дорога")[0]["title"] == "Запад"


def test_model_loader_disables_unneeded_remote_conversion_and_uses_cpu(monkeypatch):
    import os
    import sys
    from types import SimpleNamespace

    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    def constructor(model_name, *, device):
        # Transformers reads this flag before starting its non-daemon remote
        # conversion worker; it must be set before model initialization.
        assert os.environ.get("DISABLE_SAFETENSORS_CONVERSION") == "1"
        assert model_name == "BAAI/bge-m3"
        return SimpleNamespace(device=device)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=constructor),
    )
    assert dense.load_embedding_model("BAAI/bge-m3").device == "cpu"
