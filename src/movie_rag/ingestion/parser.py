import json
import re
from pathlib import Path

TITLE_PATTERN = re.compile(
    r"(?P<title>.+?)\s*\((?P<year>\d{4})\)"
)


def extract_section(
    text: str,
    start_marker: str,
    end_markers: list[str],
) -> str | None:
    start = text.find(start_marker)

    if start == -1:
        return None

    start += len(start_marker)

    end_positions = []

    for marker in end_markers:
        pos = text.find(marker, start)

        if pos != -1:
            end_positions.append(pos)

    end = min(end_positions) if end_positions else len(text)

    return text[start:end].strip()


def parse_movie_post(post: dict) -> dict | None:
    text = post["text"]

    title_match = TITLE_PATTERN.search(text)

    if title_match is None:
        return None

    title = title_match.group("title").strip()
    year = int(title_match.group("year"))

    what = extract_section(
        text,
        "ЧТО:",
        ["О ЧЁМ:", "О ЧЕМ:"],
    )

    about_marker = (
        "О ЧЁМ:"
        if "О ЧЁМ:" in text
        else "О ЧЕМ:"
    )

    about = extract_section(
        text,
        about_marker,
        ["ЗАЧЕМ:"],
    )

    why = extract_section(
        text,
        "ЗАЧЕМ:",
        ["КОМУ НЕЛЬЗЯ:"],
    )

    not_for = extract_section(
        text,
        "КОМУ НЕЛЬЗЯ:",
        [
            "Автор:",
            "Показать ещё",
            "Отправить реакцию",
        ],
    )

    return {
        "title": title,
        "year": year,
        "what": what,
        "about": about,
        "why": why,
        "not_for": not_for,
        "url": post["url"],
        "date": post["date"],
    }


def save_movies(movies: list[dict], path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for movie in movies:
            file.write(
                json.dumps(
                    movie,
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    from movie_rag.ingestion.vk_browser import URL, fetch_posts

    posts = fetch_posts(URL)

    movies = []

    for post in posts:
        movie = parse_movie_post(post)

        if movie is not None:
            movies.append(movie)

    print(f"Распарсено фильмов: {len(movies)}")

    save_movies(
        movies,
        Path("data/processed/movies.jsonl"),
    )

    for movie in movies:
        print("=" * 80)
        print(movie)