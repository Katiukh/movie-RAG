"""Validate a relevance filter's selection against the provided candidate IDs."""

import json


def output_schema(candidate_ids: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_ids"],
        "properties": {
            "selected_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": candidate_ids},
            }
        },
    }


def validate_output(content: str, candidate_ids: list[str]) -> list[str]:
    """Reject malformed output, unknown IDs and duplicate IDs; allow no matches."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty model output")
    data = json.loads(content)
    if not isinstance(data, dict) or set(data) != {"selected_ids"}:
        raise ValueError("Expected selected_ids object")
    selected = data["selected_ids"]
    if not isinstance(selected, list):
        raise TypeError("Expected selected_ids array")
    allowed = set(candidate_ids)
    seen = set()
    for movie_id in selected:
        if not isinstance(movie_id, str) or movie_id not in allowed:
            raise ValueError("Movie ID outside candidate set")
        if movie_id in seen:
            raise ValueError("Duplicate movie ID")
        seen.add(movie_id)
    return selected
