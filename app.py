"""Run with: uv run streamlit run app.py."""

import hashlib
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from movie_rag.generation import Recommender, RecommenderError
from movie_rag.retrieval.bm25 import BM25Retriever, load_movies
from movie_rag.retrieval.dense import DenseRetriever
from movie_rag.retrieval.hybrid import HybridRetriever
from movie_rag.retrieval.reranker import RerankedHybridRetriever, Reranker

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data/processed/movies.jsonl"
DENSE = "Dense — BGE-M3"
HYBRID = "Hybrid"
RERANKED = "Hybrid + Reranker"


@st.cache_resource(show_spinner=False)
def get_reranker():
    return Reranker()


@st.cache_resource(show_spinner=False)
def get_retriever(method: str, corpus_path: str, corpus_fingerprint: str):
    # The fingerprint is part of Streamlit's cache key, so corpus edits invalidate
    # the in-memory retriever. DenseRetriever separately validates its disk cache.
    if method == RERANKED:
        return RerankedHybridRetriever(
            get_retriever(HYBRID, corpus_path, corpus_fingerprint), get_reranker()
        )
    if method == HYBRID:
        return HybridRetriever(
            get_retriever("BM25", corpus_path, corpus_fingerprint),
            get_retriever(DENSE, corpus_path, corpus_fingerprint),
        )
    movies = load_movies(Path(corpus_path))
    if method == DENSE:
        return DenseRetriever(movies, cache_dir=ROOT / "data/embeddings")
    return BM25Retriever(movies)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
            return
        if "error" in message:
            st.error(message["error"])
            return
        results = message["results"]
        st.caption(f"Подходящих фильмов: {len(results)}")
        if not results:
            st.write(
                "Среди найденных фильмов не нашлось вариантов, которые достаточно "
                "хорошо соответствуют запросу."
            )
        for rank, movie in enumerate(results, 1):
            with st.expander(
                f"{rank}. {movie['title']} ({movie['year']})", expanded=rank == 1
            ):
                if movie.get("what"):
                    st.text(movie["what"])
                for field, label in (
                    ("about", "О чём"),
                    ("why", "Почему стоит посмотреть"),
                    ("not_for", "Кому может не подойти"),
                ):
                    if movie.get(field):
                        st.markdown(f"**{label}:**")
                        st.text(movie[field])
                st.link_button("Открыть источник", movie["url"])


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    st.set_page_config(page_title="Movie RAG", page_icon="🎬")
    st.title("Movie RAG")
    st.write("Найди фильм по описанию, настроению или сюжету.")
    st.caption("Каждый запрос — новый поиск. История ниже не влияет на результаты.")
    if st.sidebar.button("Очистить историю"):
        st.session_state.messages = []
    # Discard older sessions with model-generated explanations.
    if st.session_state.get("recommendation_flow_version") != 4:
        st.session_state.messages = []
        st.session_state.recommendation_flow_version = 4
    for message in st.session_state.messages:
        render_message(message)

    query = st.chat_input("Какое кино хочется посмотреть?")
    if query and query.strip():
        query = query.strip()
        user_message = {"role": "user", "content": query}
        st.session_state.messages.append(user_message)
        render_message(user_message)
        try:
            recommender = Recommender.from_env()
            fingerprint = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
            with st.spinner("Подготавливаю поиск…"):
                retriever = get_retriever(RERANKED, str(CORPUS), fingerprint)
            with st.spinner("Ищу фильмы…"):
                candidates = retriever.search(query, top_k=10)
            with st.spinner("Проверяю соответствие запросу…"):
                results = recommender.recommend(query=query, candidates=candidates[:10])
            answer = {"role": "assistant", "results": results}
        except RecommenderError as exc:
            answer = {"role": "assistant", "error": str(exc)}
        except Exception:
            logging.getLogger(__name__).exception("Search failed")
            answer = {
                "role": "assistant",
                "error": "Поиск не выполнился. Проверьте корпус и загрузку моделей или попробуйте ещё раз.",
            }
        st.session_state.messages.append(answer)
        render_message(answer)


if __name__ == "__main__":
    main()
