"""Call local Ollama with Qwen3 8B; never run retrieval here."""

import ipaddress
import math
import os
from urllib.parse import urlparse

import requests

from movie_rag.generation.prompts import build_messages
from movie_rag.generation.schemas import output_schema, validate_output

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
MAX_CANDIDATES = 10


class RecommenderError(RuntimeError):
    """Safe, user-facing generation/configuration error without server details."""


def _is_loopback(hostname: str | None) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class Recommender:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 60,
    ):
        endpoint = urlparse(base_url)
        if (
            endpoint.scheme not in ("https", "http")
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
            or not model.strip()
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise RecommenderError(
                "Проверьте LLM_BASE_URL, LLM_MODEL и LLM_TIMEOUT_SECONDS."
            )
        if not _is_loopback(endpoint.hostname):
            raise RecommenderError(
                "LLM_BASE_URL должен указывать на локальный Ollama "
                "(localhost, 127.0.0.1 или ::1)."
            )
        if model.strip().lower().endswith((":cloud", "-cloud")):
            raise RecommenderError(
                "LLM_MODEL должен указывать на локальную модель Ollama, без cloud-тега."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls):
        try:
            timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
        except ValueError:
            raise RecommenderError(
                "LLM_TIMEOUT_SECONDS должен быть положительным числом."
            ) from None
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            timeout_seconds=timeout,
        )

    def recommend(self, *, query: str, candidates: list[dict]) -> list[dict]:
        """Return copies of selected corpus movies, preserving reranker order."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError("Pass at most 10 reranked candidates")
        ids = []
        for movie in candidates:
            if not isinstance(movie, dict):
                raise TypeError("Invalid candidate")
            movie_id = movie.get("url")
            if not isinstance(movie_id, str) or not movie_id.strip() or movie_id in ids:
                raise ValueError("Candidates must have distinct nonempty URLs")
            if not isinstance(movie.get("title"), str) or not movie["title"].strip():
                raise ValueError("Candidate title must come from corpus")
            if any(
                movie.get(field) is not None and not isinstance(movie[field], str)
                for field in ("what", "about", "why", "not_for")
            ):
                raise ValueError("Candidate descriptions must be text")
            ids.append(movie_id)
        if not candidates:
            return []
        payload = {
            "model": self.model,
            "messages": build_messages(query.strip(), candidates),
            "format": output_schema(ids),
            "options": {"temperature": 0.2, "num_predict": 4096, "num_ctx": 8192},
            "think": False,
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(5, self.timeout_seconds),
                allow_redirects=False,
                proxies={"http": "", "https": ""},
            )
            response.raise_for_status()
            body = response.json()
        except requests.ConnectTimeout:
            raise RecommenderError(
                "Не удалось подключиться к локальному Ollama. "
                "Проверьте, что сервер запущен и доступен по LLM_BASE_URL."
            ) from None
        except requests.Timeout:
            raise RecommenderError(
                "Ollama не ответил вовремя. Попробуйте ещё раз или увеличьте LLM_TIMEOUT_SECONDS."
            ) from None
        except requests.RequestException:
            raise RecommenderError(
                "Не удалось обратиться к Ollama. Проверьте, что он запущен и модель загружена."
            ) from None
        except ValueError:
            raise RecommenderError(
                "Ollama вернул некорректный ответ. Попробуйте ещё раз."
            ) from None
        try:
            if not isinstance(body, dict) or body.get("error"):
                raise ValueError("API error or invalid envelope")
            if body.get("done") is not True or body.get("done_reason") != "stop":
                raise ValueError("Incomplete completion")
            selected = validate_output(body["message"]["content"], ids)
        except (ValueError, TypeError, KeyError, IndexError):
            raise RecommenderError(
                "Ollama вернул некорректный ответ. Попробуйте ещё раз."
            ) from None
        selected_ids = set(selected)
        return [dict(movie) for movie in candidates if movie["url"] in selected_ids]
