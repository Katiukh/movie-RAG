import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def app_test(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from movie_rag.retrieval import bm25, dense, reranker

    calls = {
        "dense_init": 0,
        "bm25_init": 0,
        "reranker_init": 0,
        "queries": [],
        "rerank_pairs": [],
        "llm": [],
        "selected_count": 2,
    }

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

    class Scorer:
        device = "cpu"

        def __init__(self, **kwargs):
            calls["reranker_init"] += 1

        def score(self, pairs):
            calls["rerank_pairs"].append(pairs)
            return [i / len(pairs) for i in range(len(pairs))]

    def fake_post(url, **kwargs):
        calls["llm"].append(kwargs["json"])
        candidates = json.loads(kwargs["json"]["messages"][1]["content"])["candidates"]
        reply = Mock()
        content = calls.get(
            "content",
            json.dumps(
                {
                    "selected_ids": [
                        m["movie_id"] for m in candidates[: calls["selected_count"]]
                    ]
                }
            ),
        )
        reply.json.return_value = {
            "done": True,
            "done_reason": "stop",
            "message": {"content": content},
        }
        return reply

    monkeypatch.setattr(dense, "DenseRetriever", FakeDense)
    monkeypatch.setattr(bm25, "BM25Retriever", FakeBM25)
    monkeypatch.setattr(reranker, "TransformersScorer", Scorer)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
    st.cache_resource.clear()
    yield (
        AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20
        ),
        calls,
    )
    st.cache_resource.clear()


@pytest.mark.parametrize("count", [0, 1, 4, 10])
def test_chat_filters_ten_candidates_and_displays_only_original_corpus_text(
    app_test, count
):
    app, calls = app_test
    calls["selected_count"] = count
    app.run()
    assert not app.exception
    assert not app.sidebar.selectbox
    assert not app.sidebar.radio
    app.chat_input[0].set_value("про космос без хоррора").run()
    assert not app.exception
    assert not app.error
    assert calls["queries"] == [
        ("bm25", "про космос без хоррора", 20),
        ("dense", "про космос без хоррора", 20),
    ]
    assert len(calls["rerank_pairs"][0]) == 20
    payload = json.loads(calls["llm"][0]["messages"][1]["content"])
    assert len(payload["candidates"]) == 10
    assert len(calls["llm"]) == 1
    results = app.session_state.messages[-1]["results"]
    assert [m["url"] for m in results] == [
        m["movie_id"] for m in payload["candidates"][:count]
    ]
    assert len(app.expander) == count
    assert len(app.get("link_button")) == count
    assert [t.value for t in app.text] == [
        movie[field]
        for movie in results
        for field in ("what", "about", "why", "not_for")
        if movie.get(field)
    ]
    for movie in results:
        assert (
            not {
                "reason",
                "evidence",
                "requirements",
                "status",
                "include",
                "evaluations",
            }
            & movie.keys()
        )
    rendered = " ".join(m.value for m in app.markdown)
    assert "Почему подходит:" not in rendered
    assert "Из описания:" not in rendered
    if count:
        assert "О чём:" in rendered
        assert "Почему стоит посмотреть:" in rendered
        assert "Кому может не подойти:" in rendered
    else:
        assert "Среди найденных фильмов не нашлось вариантов" in rendered
    captions = " ".join(c.value for c in app.caption)
    for internal in ("Reranker score", "BM25", "RRF", "device"):
        assert internal not in captions


def test_reruns_reuse_resources_and_queries_have_no_conversation_context(app_test):
    app, calls = app_test
    app.run()
    app.chat_input[0].set_value("дорога").run()
    app.run()
    app.chat_input[0].set_value("интернет").run()
    assert not app.exception
    assert calls["dense_init"] == calls["bm25_init"] == calls["reranker_init"] == 1
    assert len(calls["llm"]) == 2
    second = calls["llm"][1]["messages"]
    assert [m["role"] for m in second] == ["system", "user"]
    assert json.loads(second[1]["content"])["query"] == "интернет"
    assert len(app.session_state.messages) == 4
    assert len(app.chat_message) == 4
    assert len(app.expander) == 4
    app.sidebar.button[0].click().run()
    assert not app.session_state.messages
    assert not app.chat_message


@pytest.mark.parametrize(
    "content",
    [
        "",
        '{"selected_ids": ["outside"]}',
        '{"selected_ids": [1]}',
        '{"selected_ids": ["outside", "outside"]}',
        '{"selected_ids": [], "reason": "Придуманный фильм"}',
        '{"evaluations": []}',
    ],
)
def test_invalid_llm_response_shows_safe_error_without_retrieval_fallback(
    app_test, content
):
    app, calls = app_test
    calls["content"] = content
    app.run()
    app.chat_input[0].set_value("космос").run()
    assert not app.exception
    assert app.error
    assert "results" not in app.session_state.messages[-1]
    assert not app.expander
    assert not app.get("link_button")
    assert "outside" not in app.error[0].value


@pytest.mark.parametrize(
    "failure", [requests.Timeout(), requests.HTTPError("private error details")]
)
def test_api_failure_is_a_ui_error_without_traceback_or_fallback(
    app_test, monkeypatch, failure
):
    app, _calls = app_test
    monkeypatch.setattr(requests, "post", Mock(side_effect=failure))
    app.run()
    app.chat_input[0].set_value("космос").run()
    assert not app.exception
    assert app.error
    assert not app.expander
    assert "results" not in app.session_state.messages[-1]
    assert "private error details" not in app.error[0].value


def test_external_endpoint_is_rejected_before_llm_or_search(app_test, monkeypatch):
    app, calls = app_test
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid")
    app.run()
    app.chat_input[0].set_value("космос").run()
    assert not app.exception
    assert "локаль" in app.error[0].value
    assert not calls["llm"]
    assert not calls["queries"]
    assert not app.expander


def test_existing_history_is_reset_when_switching_to_corpus_only_display(app_test):
    app, calls = app_test
    app.session_state.recommendation_flow_version = 3
    app.session_state.messages = [
        {
            "role": "assistant",
            "results": [
                {
                    "url": "corpus:old",
                    "title": "Старый результат",
                    "year": 2000,
                    "reason": "Сгенерированное описание старого формата.",
                }
            ],
        }
    ]
    app.run()
    assert not app.exception
    assert not app.session_state.messages
    assert not app.expander
    assert not calls["llm"]


def test_optional_empty_corpus_fields_do_not_crash_or_generate_filler(
    app_test, monkeypatch
):
    from movie_rag.generation import Recommender

    recommend = Recommender.recommend

    def missing_fields(self, **kwargs):
        results = recommend(self, **kwargs)
        for movie in results:
            movie.update(what=None, about="", why=None, not_for="")
        return results

    monkeypatch.setattr(Recommender, "recommend", missing_fields)
    app, calls = app_test
    calls["selected_count"] = 1
    app.run()
    app.chat_input[0].set_value("приключение").run()
    assert not app.exception
    assert not app.error
    assert len(app.expander) == 1
    assert not app.text
    rendered = " ".join(m.value for m in app.markdown)
    for heading in ("О чём:", "Почему стоит посмотреть:", "Кому может не подойти:"):
        assert heading not in rendered
    assert len(app.get("link_button")) == 1


def test_old_generated_explanations_are_not_used_even_if_present_in_session(app_test):
    app, _calls = app_test
    app.run()
    app.session_state.messages = [
        {
            "role": "assistant",
            "results": [
                {
                    "url": "corpus:selected",
                    "title": "Фильм из corpus",
                    "year": 2000,
                    "what": "драма",
                    "about": "Исходное описание.",
                    "why": "Исходная рекомендация автора.",
                    "not_for": "",
                    "reason": "Придуманная характеристика.",
                    "requirements": [
                        {
                            "requirement": "Невидимое условие",
                            "evidence": {"field": "about", "quote": "Невидимая цитата"},
                        }
                    ],
                }
            ],
        }
    ]
    app.run()
    assert not app.exception
    assert [t.value for t in app.text] == [
        "драма",
        "Исходное описание.",
        "Исходная рекомендация автора.",
    ]
    all_text = " ".join(m.value for m in app.markdown) + " ".join(
        t.value for t in app.text
    )
    assert "Придуманная характеристика" not in all_text
    assert "Невидим" not in all_text


def test_original_corpus_markdown_is_displayed_literally(app_test, monkeypatch):
    from movie_rag.generation import Recommender

    recommend = Recommender.recommend

    def text_with_markdown(self, **kwargs):
        results = recommend(self, **kwargs)
        for movie in results:
            movie["about"] = (
                "Точный *текст* с [ссылкой](https://example.invalid) и\nновой строкой."
            )
        return results

    monkeypatch.setattr(Recommender, "recommend", text_with_markdown)
    app, calls = app_test
    calls["selected_count"] = 1
    app.run()
    app.chat_input[0].set_value("описание").run()
    assert not app.exception
    assert "Точный *текст* с [ссылкой](https://example.invalid) и\nновой строкой." in [
        t.value for t in app.text
    ]
