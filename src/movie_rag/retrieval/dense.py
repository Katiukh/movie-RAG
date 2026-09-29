"""BGE-M3 dense movie retrieval with a small, validated on-disk cache."""

import argparse
import hashlib
import io
import json
import logging
import os
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np

from movie_rag.retrieval.bm25 import DEFAULT_CORPUS, load_movies

MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CACHE = Path("data/embeddings")
EXAMPLE_QUERIES = (
    "неглупый ужастик",
    "хочу тревожное и медленное кино про одиночество",
    "про человека, которому тяжело вернуться к нормальной жизни после войны",
    "что-нибудь странное про интернет",
    "мрачная история о разваливающихся отношениях",
    "дорожное кино про дружбу",
)
logger = logging.getLogger(__name__)


def build_search_text(movie: dict) -> str:
    """Keep original language and casing; omit negatively phrased not_for."""
    return " ".join(
        part.strip()
        for field in ("title", "what", "about", "why")
        if (part := movie.get(field)) and part.strip()
    )


def load_embedding_model(model_name: str):
    # BGE-M3 ships .bin weights. Transformers otherwise starts an unrelated
    # remote safetensors conversion worker that can keep the CLI alive.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    # Lazy imports keep fake-model tests and BM25-only UI use lightweight.
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class DenseRetriever:
    def __init__(
        self,
        movies: list[dict],
        model_name: str = MODEL_NAME,
        *,
        cache_dir: Path | None = DEFAULT_CACHE,
        rebuild_cache: bool = False,
        batch_size: int = 8,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._movies = [dict(movie) for movie in movies]
        self._encode_lock = threading.Lock()
        self.device = "not loaded"
        self.cache_hit = False
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        if not self._movies:
            return
        self._model = load_embedding_model(model_name)
        self.device = str(self._model.device)
        texts = [build_search_text(movie) for movie in self._movies]
        urls = [movie["url"] for movie in self._movies]
        fingerprint = hashlib.sha256(
            json.dumps(
                {"texts": texts, "urls": urls},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        dimension = self._model.get_embedding_dimension()
        manifest = {
            "format_version": 1,
            "model_name": model_name,
            "fingerprint": fingerprint,
            "document_urls": urls,
            "dimension": dimension,
            "normalized": True,
        }
        cache_dir = Path(cache_dir) if cache_dir is not None else None
        embeddings = None
        if cache_dir is not None and not rebuild_cache:
            embeddings = self._read_cache(cache_dir, manifest)
        if embeddings is not None:
            self.cache_hit = True
            logger.info("Loaded cached embeddings for %s movies", len(movies))
        else:
            logger.info("Encoding %s movies (once)", len(movies))
            embeddings = np.asarray(
                self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=True,
                ),
                dtype=np.float32,
            )
            if not self._valid_matrix(embeddings, len(movies), dimension):
                raise ValueError(
                    "Model returned invalid or unnormalized movie embeddings"
                )
            if cache_dir is not None:
                self._write_cache(cache_dir, manifest, embeddings)
        self._embeddings = embeddings

    @staticmethod
    def _valid_matrix(matrix: np.ndarray, rows: int, dimension: int) -> bool:
        return (
            matrix.shape == (rows, dimension)
            and np.issubdtype(matrix.dtype, np.floating)
            and bool(np.isfinite(matrix).all())
            and bool(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4))
        )

    @classmethod
    def _read_cache(cls, directory: Path, manifest: dict) -> np.ndarray | None:
        try:
            metadata = json.loads(
                (directory / "bge_m3_metadata.json").read_text(encoding="utf-8")
            )
            if not isinstance(metadata, dict) or any(
                metadata.get(k) != v for k, v in manifest.items()
            ):
                return None
            payload = (directory / "bge_m3_embeddings.npy").read_bytes()
            # Also detects interrupted/concurrent writes of the two-file cache.
            if hashlib.sha256(payload).hexdigest() != metadata.get("matrix_sha256"):
                return None
            matrix = np.load(io.BytesIO(payload), allow_pickle=False)
            if not cls._valid_matrix(
                matrix, len(manifest["document_urls"]), manifest["dimension"]
            ):
                return None
            return matrix.astype(np.float32, copy=False)
        except (OSError, ValueError, EOFError):
            return None

    @staticmethod
    def _write_cache(directory: Path, manifest: dict, matrix: np.ndarray) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        payload = buffer.getvalue()
        metadata = {**manifest, "matrix_sha256": hashlib.sha256(payload).hexdigest()}
        _atomic_write(directory / "bge_m3_embeddings.npy", payload)
        _atomic_write(
            directory / "bge_m3_metadata.json",
            (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return copied metadata sorted by cosine similarity; blank query -> []."""
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not query.strip() or top_k == 0 or not self._movies:
            return []
        # st.cache_resource shares this retriever between Streamlit sessions.
        with self._encode_lock:
            query_embedding = self._model.encode(
                [query.strip()],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        scores = self._embeddings @ query_embedding
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [dict(self._movies[i], score=float(scores[i])) for i in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Omit to run six example queries")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.top_k < 0 or args.batch_size < 1:
        parser.error("--top-k must be >= 0 and --batch-size must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    movies = load_movies(args.corpus)
    retriever = DenseRetriever(
        movies,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        batch_size=args.batch_size,
    )
    print(
        f"{MODEL_NAME}: {len(movies)} movies; device={retriever.device}; cache_hit={retriever.cache_hit}"
    )
    for query in [args.query] if args.query is not None else EXAMPLE_QUERIES:
        print(f"\nQuery: {query}")
        for rank, movie in enumerate(retriever.search(query, args.top_k), 1):
            print(
                f"{rank}. {movie['title']} ({movie['year']}) — similarity {movie['score']:.4f}"
            )
            print(f"   Что: {movie.get('what') or ''}")
            print(f"   Зачем: {movie.get('why') or ''}")
            print(f"   URL: {movie['url']}")


if __name__ == "__main__":
    main()
