import copy
import json

import pytest

from movie_rag.retrieval.bm25 import (
    BM25Retriever,
    build_search_text,
    load_movies,
    tokenize,
)


@pytest.fixture
def movies():
    return [
        {
            "title": title,
            "year": 2000,
            "what": what,
            "about": about,
            "why": why,
            "not_for": "секретныймаркер",
            "url": f"https://vk.ru/wall-1_{i}",
            "date": "сегодня",
        }
        for i, (title, what, about, why) in enumerate(
            [
                ("Киберлабиринт", "триллер", "хакер взламывает компьютер", "загадка"),
                ("Туман", "хоррор", "заброшенный дом", "страх"),
                ("Путь домой", "роуд-муви", "дорога", "приключения"),
                ("Лето", "комедия", "друзья", "отдых"),
                ("Орбита", "фантастика", "космос", "открытия"),
            ]
        )
    ]


def test_load_jsonl(tmp_path, movies):
    path = tmp_path / "movies.jsonl"
    path.write_text(
        "\n\n".join(json.dumps(m, ensure_ascii=False) for m in movies) + "\n",
        encoding="utf-8",
    )
    assert load_movies(path) == movies


def test_build_text_uses_only_positive_fields(movies):
    assert (
        build_search_text(movies[0])
        == "Киберлабиринт триллер хакер взламывает компьютер загадка"
    )
    assert build_search_text({"title": "Фильм", "what": None}) == "Фильм"


def test_tokenizer():
    assert tokenize("ЁЖ, МРАЧНЫЙ роуд-муви! Hacker_2002…") == [
        "ёж",
        "мрачный",
        "роуд",
        "муви",
        "hacker",
        "2002",
    ]
    assert tokenize(" — !!! ") == []


def test_search_ranking_metadata_and_no_mutation(movies):
    original = copy.deepcopy(movies)
    results = BM25Retriever(movies).search("ХАКЕР, триллер хоррор", top_k=2)
    assert len(results) == 2
    assert results[0]["title"] == "Киберлабиринт"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["not_for"] == "секретныймаркер"
    assert isinstance(results[0]["score"], float)
    assert {k: v for k, v in results[0].items() if k != "score"} == movies[0]
    results[0]["title"] = "changed"
    assert movies == original


@pytest.mark.parametrize("query", ["КИБЕРЛАБИРИНТ", "компьютер"])
def test_exact_title_or_rare_word_ranks_first(movies, query):
    assert BM25Retriever(movies).search(query)[0]["title"] == "Киберлабиринт"


@pytest.mark.parametrize("query", ["", "???", "несуществующееслово", "секретныймаркер"])
def test_no_lexical_matches_returns_empty(movies, query):
    assert BM25Retriever(movies).search(query) == []


def test_top_k_limits_and_empty_corpus(movies):
    retriever = BM25Retriever(movies)
    query = "хоррор триллер комедия"
    assert len(retriever.search(query, top_k=1)) == 1
    assert len(retriever.search(query, top_k=100)) == 3
    assert retriever.search(query, top_k=0) == []
    with pytest.raises(ValueError):
        retriever.search(query, top_k=-1)
    assert BM25Retriever([]).search(query) == []
    assert BM25Retriever([{"title": "!!!"}]).search(query) == []


def test_single_document_match_is_not_discarded_for_nonpositive_idf(movies):
    assert BM25Retriever(movies[:1]).search("хакер")[0]["title"] == "Киберлабиринт"
