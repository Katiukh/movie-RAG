"""Create unreviewed corpus-only starter queries; never write gold labels."""

import argparse
import json
from pathlib import Path

from movie_rag.evaluation.dataset import DEFAULT_CANDIDATES, DEFAULT_DATASET
from movie_rag.retrieval.bm25 import DEFAULT_CORPUS, load_movies


def generate_candidates(movies: list[dict], count: int = 50) -> list[dict]:
    """Sample across about-field lengths, independently of any retriever.

    These are deliberately rough near_exact drafts. A person must rewrite
    paraphrase/semantic queries, check all positives and promote rows manually.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    eligible = sorted(
        [movie for movie in movies if (movie.get("about") or "").strip()],
        key=lambda movie: (len(movie["about"]), movie["url"]),
    )
    if not eligible:
        raise ValueError("Corpus has no nonempty about fields")
    size = min(count, len(eligible))
    selected = [eligible[i * len(eligible) // size] for i in range(size)]
    return [
        {
            "query_id": f"q{i:03d}",
            "query": f"фильм {movie['about'].strip()}",
            "relevant_ids": [movie["url"]],
            "query_type": "near_exact",
            "notes": "UNREVIEWED: about-based draft; rewrite, check indexed evidence and other positives.",
        }
        for i, movie in enumerate(selected, 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    if args.output.resolve() in {DEFAULT_DATASET.resolve(), args.corpus.resolve()}:
        parser.error("Candidate output must not overwrite gold or corpus")
    try:
        rows = generate_candidates(load_movies(args.corpus), args.count)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects existing manually edited candidates.
        with args.output.open("x", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"Unreviewed candidates: {len(rows)} -> {args.output}; gold is promoted manually"
    )


if __name__ == "__main__":
    main()
