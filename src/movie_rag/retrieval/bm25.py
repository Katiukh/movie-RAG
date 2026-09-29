"""Raw-token BM25 baseline: one document per movie, no semantic expansion."""

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

DEFAULT_CORPUS = Path("data/processed/movies.jsonl")
EXAMPLE_QUERIES = (
    "неглупый ужастик",
    "хоррор",
    "мрачный криминальный триллер",
    "уютное осеннее кино",
    "одиночество",
    "роуд-муви",
    "программисты и хакеры",
    "программист хакер",
)
_TOKEN_PATTERN = re.compile(r"[а-яёa-z0-9]+")
_SEARCH_FIELDS = ("title", "what", "about", "why")


def load_movies(path: Path) -> list[dict]:
    """Load UTF-8 JSONL, ignoring blank lines; malformed rows report a line."""
    movies = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                movie = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(movie, dict):
                raise TypeError(f"{path}:{line_number}: expected a movie object")
            movies.append(movie)
    return movies


def build_search_text(movie: dict) -> str:
    """Exclude metadata and the negatively phrased not_for field."""
    return " ".join(movie.get(field) or "" for field in _SEARCH_FIELDS).strip()


def tokenize(text: str) -> list[str]:
    """Same lowercase Russian/Latin/number tokens for documents and queries.

    No stemming, lemmatization, stop-word removal or synonym expansion.
    Hyphens split words; ё and е remain distinct.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, movies: list[dict]):
        self._movies = [dict(movie) for movie in movies]
        self._tokens = [tokenize(build_search_text(movie)) for movie in self._movies]
        # rank-bm25 cannot initialize an empty corpus/vocabulary.
        self._index = BM25Okapi(self._tokens) if any(self._tokens) else None
        self._token_sets = [set(tokens) for tokens in self._tokens]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Rank lexical matches by unmodified BM25 scores, descending.

        Return fresh dictionaries including all original metadata. Ties retain
        corpus order. A score is not a probability; on tiny corpora Okapi IDF
        can be zero/negative, so candidates are selected by token overlap,
        never by score > 0. Unknown/empty queries return no results.
        """
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        query_tokens = tokenize(query)
        if not query_tokens or top_k == 0 or self._index is None:
            return []
        scores = self._index.get_scores(query_tokens)
        query_terms = set(query_tokens)
        candidates = [
            i for i, terms in enumerate(self._token_sets) if terms & query_terms
        ]
        ranked = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)
        return [dict(self._movies[i], score=float(scores[i])) for i in ranked[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query", nargs="?", help="Omit to run the baseline example queries"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    movies = load_movies(args.corpus)
    retriever = BM25Retriever(movies)
    print(f"Corpus: {args.corpus} ({len(movies)} movies); raw tokens, BM25Okapi")
    queries = [args.query] if args.query is not None else EXAMPLE_QUERIES
    for query in queries:
        print(f"\nQuery: {query}\n")
        results = retriever.search(query, top_k=args.top_k)
        if not results:
            print("Нет лексических совпадений.")
        for rank, movie in enumerate(results, start=1):
            print(f"{rank}. {movie['title']} ({movie['year']})")
            print(f"   BM25: {movie['score']:.4f}")
            print(f"   Что: {movie.get('what') or ''}")
            print(f"   Зачем: {movie.get('why') or ''}")
            print(f"   URL: {movie.get('url') or ''}")


if __name__ == "__main__":
    main()
