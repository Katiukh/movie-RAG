"""Run with: uv run streamlit run app.py."""

import hashlib
import logging
from pathlib import Path

import streamlit as st

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
        st.caption(f"{message['method']} · Найдено фильмов: {len(results)}")
        if not results:
            st.write("По этому запросу ничего не найдено. Попробуйте другое описание.")
        score_label = "Similarity" if message["method"] == DENSE else "BM25 score"
        for rank, movie in enumerate(results, 1):
            with st.expander(
                f"{rank}. {movie['title']} ({movie['year']})", expanded=rank == 1
            ):
                if message["method"] == RERANKED:
                    st.caption(f"Reranker score: {movie['reranker_score']:.6f}")
                    with st.expander("Ранги до reranking"):
                        st.caption(
                            f"Hybrid rank: {movie['hybrid_rank']} · "
                            f"RRF: {movie['rrf_score']:.6f} · "
                            f"BM25 rank: {movie['bm25_rank'] or '—'} · "
                            f"Dense rank: {movie['dense_rank'] or '—'}"
                        )
                elif message["method"] == HYBRID:
                    st.caption(
                        f"RRF: {movie['rrf_score']:.6f} · "
                        f"BM25 rank: {movie['bm25_rank'] or '—'} · "
                        f"Dense rank: {movie['dense_rank'] or '—'}"
                    )
                    with st.expander("Исходные оценки"):
                        for label, field in (
                            ("BM25 score", "bm25_score"),
                            ("Dense score", "dense_score"),
                        ):
                            score = movie[field]
                            value = "—" if score is None else f"{score:.4f}"
                            st.caption(f"{label}: {value}")
                else:
                    st.caption(f"{score_label}: {movie['score']:.4f}")
                for label, field in [
                    ("ЧТО", "what"),
                    ("О ЧЁМ", "about"),
                    ("ЗАЧЕМ", "why"),
                ]:
                    st.markdown(f"**{label}:**")
                    st.write(movie.get(field) or "—")
                st.link_button("Открыть источник", movie["url"])


def main() -> None:
    st.set_page_config(page_title="Movie RAG", page_icon="🎬")
    st.title("Movie RAG")
    st.write("Найди фильм по описанию, настроению или сюжету.")
    st.caption("Каждый запрос — новый поиск. История ниже не влияет на результаты.")
    method = st.sidebar.radio("Retrieval method", [DENSE, "BM25", HYBRID, RERANKED])
    top_k = st.sidebar.selectbox("Number of results", [3, 5, 10], index=1)
    if st.sidebar.button("Очистить историю"):
        st.session_state.messages = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        render_message(message)

    try:
        fingerprint = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
        with st.spinner("Подготавливаю поиск…"):
            retriever = get_retriever(method, str(CORPUS), fingerprint)
    except Exception:
        logging.getLogger(__name__).exception("Could not initialize retrieval")
        st.error(
            "Не удалось подготовить поиск. Проверьте файл корпуса и соединение "
            "при первой загрузке модели. Можно попробовать другой метод в боковой панели."
        )
        st.stop()
    if method in (DENSE, HYBRID, RERANKED):
        base = retriever.hybrid_retriever if method == RERANKED else retriever
        dense = base if method == DENSE else base.dense_retriever
        st.sidebar.caption(f"Embedding device: {dense.device}")
    if method == RERANKED:
        st.sidebar.caption(f"Reranker device: {retriever.reranker.device}")

    query = st.chat_input("Какое кино хочется посмотреть?")
    if query and query.strip():
        query = query.strip()
        user_message = {"role": "user", "content": query}
        st.session_state.messages.append(user_message)
        render_message(user_message)
        try:
            with st.spinner("Ищу фильмы…"):
                results = retriever.search(query, top_k=top_k)
            answer = {"role": "assistant", "method": method, "results": results}
        except Exception:
            logging.getLogger(__name__).exception("Search failed")
            answer = {
                "role": "assistant",
                "error": "Поиск не выполнился. Попробуйте ещё раз или смените метод.",
            }
        st.session_state.messages.append(answer)
        render_message(answer)


if __name__ == "__main__":
    main()
