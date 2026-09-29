from pathlib import Path

import pytest


@pytest.fixture
def app_test(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from movie_rag.retrieval import bm25, dense

    calls = {"dense_init": 0, "bm25_init": 0, "queries": []}

    class FakeDense:
        device = "cpu"

        def __init__(self, movies, **kwargs):
            calls["dense_init"] += 1
            self.movies = movies

        def search(self, query, top_k=5):
            calls["queries"].append(("dense", query, top_k))
            return [dict(m, score=0.75) for m in self.movies[:top_k]]

    class FakeBM25(FakeDense):
        def __init__(self, movies):
            calls["bm25_init"] += 1
            self.movies = movies

        def search(self, query, top_k=5):
            calls["queries"].append(("bm25", query, top_k))
            return [dict(m, score=4.25) for m in self.movies[:top_k]]

    monkeypatch.setattr(dense, "DenseRetriever", FakeDense)
    monkeypatch.setattr(bm25, "BM25Retriever", FakeBM25)
    st.cache_resource.clear()
    yield (
        AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20
        ),
        calls,
    )
    st.cache_resource.clear()


def test_chat_switches_method_and_preserves_independent_history(app_test):
    app, calls = app_test
    app.run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Dense — BGE-M3"
    assert app.sidebar.selectbox[0].value == 5
    app.chat_input[0].set_value("хочу хоррор").run()
    assert not app.exception
    assert calls["queries"] == [("dense", "хочу хоррор", 5)]
    assert len(app.session_state.messages) == 2
    app.sidebar.radio[0].set_value("BM25").run()
    app.sidebar.selectbox[0].set_value(3).run()
    app.chat_input[0].set_value("а что-нибудь про интернет?").run()
    assert not app.exception
    assert calls["queries"][-1] == ("bm25", "а что-нибудь про интернет?", 3)
    assert len(app.session_state.messages) == 4
    assert len(app.session_state.messages[1]["results"]) == 5
    assert app.session_state.messages[1]["method"] == "Dense — BGE-M3"
    assert len(app.session_state.messages[3]["results"]) == 3
    assert app.session_state.messages[3]["method"] == "BM25"
    assert calls["dense_init"] == 1
    assert calls["bm25_init"] == 1
    assert len(app.chat_message) == 4
    assert len(app.expander) == 8
    captions = " ".join(c.value for c in app.caption)
    assert "Similarity" in captions and "BM25 score" in captions


def test_reruns_reuse_dense_resource(app_test):
    app, calls = app_test
    app.run()
    app.run()
    app.chat_input[0].set_value("дорога").run()
    app.chat_input[0].set_value("интернет").run()
    assert not app.exception
    assert calls["dense_init"] == 1
    assert calls["queries"] == [("dense", "дорога", 5), ("dense", "интернет", 5)]
