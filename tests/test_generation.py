import copy
import json
from unittest.mock import Mock

import pytest
import requests

from movie_rag.generation.recommender import Recommender, RecommenderError
from movie_rag.generation.schemas import output_schema, validate_output


def movies(count=10):
    return [
        {
            "url": f"https://vk.ru/wall-1_{i}",
            "title": f"Фильм {i}",
            "year": "2000",
            "what": "фантастическая драма",
            "about": "экипаж исследует космос",
            "why": "светлая история",
            "not_for": "тем, кто не любит спокойное кино",
            "reranker_score": 0.9,
        }
        for i in range(count)
    ]


def response(content):
    reply = Mock()
    reply.json.return_value = {
        "done": True,
        "done_reason": "stop",
        "message": {"content": content, "thinking": "private reasoning"},
    }
    return reply


@pytest.fixture
def post(monkeypatch):
    call = Mock()
    monkeypatch.setattr(requests, "post", call)
    return call


@pytest.mark.parametrize("count", [0, 1, 4, 10])
def test_variable_count_returns_only_selected_corpus_movies_in_reranker_order(
    post, count
):
    candidates = movies()
    before = copy.deepcopy(candidates)
    selected = list(reversed([m["url"] for m in candidates[:count]]))
    post.return_value = response(json.dumps({"selected_ids": selected}))
    result = Recommender().recommend(query="космос", candidates=candidates)
    assert result == candidates[:count]
    for returned, original in zip(result, candidates, strict=False):
        assert returned is not original
        assert (
            not {"reason", "requirements", "evidence", "status", "include"}
            & returned.keys()
        )
    assert candidates == before
    post.assert_called_once()


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        '```json\n{"selected_ids": []}\n```',
        "null",
        "[]",
        "{}",
        '{"selected_ids": null}',
        '{"selected_ids": "a"}',
        '{"selected_ids": {}}',
        '{"selected_ids": [null]}',
        '{"selected_ids": [1]}',
        '{"selected_ids": [[]]}',
        '{"selected_ids": [{}]}',
        '{"selected_ids": ["unknown"]}',
        '{"selected_ids": ["https://vk.ru/wall-1_0", "https://vk.ru/wall-1_0"]}',
        '{"selected_ids": [], "reason": "Придуманное объяснение"}',
        '{"selected_ids": [], "evidence": []}',
        '{"selected_ids": [], "scores": []}',
        '{"evaluations": []}',
        '{"recommendations": []}',
    ],
)
def test_malformed_unknown_duplicate_or_extra_output_is_rejected(post, content):
    post.return_value = response(content)
    with pytest.raises(RecommenderError):
        Recommender().recommend(query="космос", candidates=movies())


def test_valid_selected_ids_and_empty_selection_are_valid():
    assert validate_output('{"selected_ids": ["a", "b"]}', ["a", "b", "c"]) == [
        "a",
        "b",
    ]
    assert validate_output('{"selected_ids": []}', ["a", "b"]) == []


def test_unknown_ids_are_rejected_even_alongside_valid_ids(post):
    post.return_value = response(
        json.dumps({"selected_ids": [movies(1)[0]["url"], "unknown"]})
    )
    with pytest.raises(RecommenderError):
        Recommender().recommend(query="космос", candidates=movies())


def test_schema_only_accepts_distinct_candidate_ids():
    schema = output_schema(["a", "b"])
    assert schema["required"] == ["selected_ids"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"selected_ids"}
    ids = schema["properties"]["selected_ids"]
    assert ids["type"] == "array"
    assert ids["items"] == {"type": "string", "enum": ["a", "b"]}
    assert ids["uniqueItems"] is True


def test_display_fields_are_preserved_verbatim_without_generation(post):
    candidate = dict(
        movies(1)[0],
        about="Оригинальный текст с *Markdown* и\nновой строкой.",
        why="Зачем смотреть: human-written.",
        not_for=None,
    )
    post.return_value = response(json.dumps({"selected_ids": [candidate["url"]]}))
    result = Recommender().recommend(query="описание", candidates=[candidate])
    assert result == [candidate]
    assert result[0]["about"] == "Оригинальный текст с *Markdown* и\nновой строкой."
    assert result[0]["not_for"] is None


def test_prompt_covers_all_compound_conditions_and_corpus_only_decisions(post):
    post.return_value = response('{"selected_ids": []}')
    Recommender().recommend(query="романтика но грустная", candidates=movies(1))
    prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "только what, about, why, not_for" in prompt
    assert "Не используй знания из pretraining" in prompt
    assert "ВСЕ существенные условия" in prompt
    assert "романтика но грустная" in prompt.casefold()
    assert "романтической линии недостаточно" in prompt
    assert "мало стрельбы но она есть" in prompt.casefold()
    assert "фантастика без хоррора" in prompt.casefold()
    assert "отсутствие слова" in prompt
    assert "selected_ids" in prompt
    assert "Не возвращай reason, evidence" in prompt


def test_request_uses_local_ollama_schema_and_only_corpus_fields(post):
    post.return_value = response('{"selected_ids": []}')
    Recommender().recommend(query="космос", candidates=movies())
    url = post.call_args.args[0]
    args = post.call_args.kwargs
    assert url == "http://localhost:11434/api/chat"
    assert "Authorization" not in args.get("headers", {})
    assert args["allow_redirects"] is False
    assert args["proxies"] == {"http": "", "https": ""}
    assert args["timeout"] == (5, 60)
    payload = args["json"]
    assert payload["model"] == "qwen3:8b"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] >= 8192
    assert payload["options"]["num_predict"] == 4096
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    data = json.loads(payload["messages"][1]["content"])
    assert payload["format"] == data["output_schema"]
    assert data["query"] == "космос"
    assert len(data["candidates"]) == 10
    assert set(data["candidates"][0]) == {
        "movie_id",
        "title",
        "what",
        "about",
        "why",
        "not_for",
    }
    assert data["candidates"][0]["movie_id"] == movies()[0]["url"]
    assert "private reasoning" not in json.dumps(payload)


def test_empty_candidates_need_no_api_call(post):
    assert Recommender().recommend(query="космос", candidates=[]) == []
    post.assert_not_called()


@pytest.mark.parametrize("candidates", [movies(11), [movies()[0]] * 2, [{"url": ""}]])
def test_invalid_candidates_are_rejected_before_api(post, candidates):
    with pytest.raises(ValueError):
        Recommender().recommend(query="космос", candidates=candidates)
    post.assert_not_called()


@pytest.mark.parametrize("failure", [requests.Timeout(), requests.ConnectionError()])
def test_network_errors_are_safe_and_do_not_fall_back(post, failure):
    post.side_effect = failure
    with pytest.raises(RecommenderError) as error:
        Recommender().recommend(query="космос", candidates=movies())
    assert "private error details" not in str(error.value)


def test_connection_timeout_explains_that_ollama_must_be_running(post):
    post.side_effect = requests.ConnectTimeout()
    with pytest.raises(RecommenderError, match="запущен") as error:
        Recommender().recommend(query="космос", candidates=movies())
    assert "LLM_TIMEOUT_SECONDS" not in str(error.value)
    post.assert_called_once()


def test_http_error_does_not_expose_ollama_response(post):
    post.return_value = response('{"selected_ids": []}')
    post.return_value.raise_for_status.side_effect = requests.HTTPError(
        "private error details"
    )
    with pytest.raises(RecommenderError) as error:
        Recommender().recommend(query="космос", candidates=movies())
    assert "private error details" not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"error": "private error details"},
        {"done": False, "message": {"content": '{"selected_ids": []}'}},
        {
            "done": True,
            "done_reason": "length",
            "message": {"content": '{"selected_ids": []}'},
        },
        {"done": True, "done_reason": "stop", "message": {"content": None}},
    ],
)
def test_incomplete_or_malformed_api_envelope_is_an_error(post, body):
    post.return_value.json.return_value = body
    with pytest.raises(RecommenderError):
        Recommender().recommend(query="космос", candidates=movies())


def test_defaults_need_no_credentials(post, monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    recommender = Recommender.from_env()
    assert recommender.model == "qwen3:8b"
    assert recommender.base_url == "http://localhost:11434"
    post.assert_not_called()


def test_configurable_local_endpoint_model_and_timeout(post, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11435/")
    monkeypatch.setenv("LLM_MODEL", "qwen3:4b")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    post.return_value = response('{"selected_ids": []}')
    Recommender.from_env().recommend(query="космос", candidates=movies())
    assert post.call_args.args[0] == "http://127.0.0.1:11435/api/chat"
    assert post.call_args.kwargs["timeout"] == (5, 30)
    assert post.call_args.kwargs["json"]["model"] == "qwen3:4b"
    assert "Authorization" not in post.call_args.kwargs.get("headers", {})


@pytest.mark.parametrize("url", ["https://example.invalid", "http://192.168.0.2:11434"])
def test_external_endpoints_are_rejected_without_any_call(post, url):
    with pytest.raises(RecommenderError, match="локаль"):
        Recommender(base_url=url)
    post.assert_not_called()


@pytest.mark.parametrize(
    "model", ["qwen3:8b-cloud", "gemma4:cloud", " GPT-OSS:120B-CLOUD "]
)
def test_cloud_models_are_rejected_before_inference(post, monkeypatch, model):
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434")
    post.return_value = response('{"selected_ids": []}')
    with pytest.raises(RecommenderError, match="локаль"):
        Recommender.from_env().recommend(query="космос", candidates=movies())
    post.assert_not_called()


def test_ipv6_loopback_is_supported(post):
    post.return_value = response('{"selected_ids": []}')
    Recommender(base_url="http://[::1]:11434").recommend(
        query="космос", candidates=movies()
    )
    assert post.call_args.args[0] == "http://[::1]:11434/api/chat"
