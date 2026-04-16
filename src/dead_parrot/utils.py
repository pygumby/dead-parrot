"""Utility functions."""

import json
import os
import re
from datetime import datetime

import pypdf


def load_pdf(path: str) -> list[str]:
    """Load the texts from a PDF file."""
    reader = pypdf.PdfReader(stream=path)
    return [reader.pages[i].extract_text() for i in range(len(reader.pages))]


def load_json(path: str) -> list[dict[str, str]]:
    """Load a list of dicts from a JSON file."""
    with open(file=path) as file:
        data = json.load(fp=file)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"JSON in {path} doesn't contain a list of dicts.")

    return data


def _normalize_name(name: str) -> str:
    """Normalize a name to a lowercase, underscore-separated, alphanumeric string."""
    if not re.search(pattern=r"[a-zA-Z0-9]", string=name):
        raise ValueError(
            f"Name must contain letters or numbers, but '{name}' does not."
        )

    return re.sub(
        pattern=r"[^a-z0-9_]",
        repl="",
        string=re.sub(
            pattern=r"[ -]",
            repl="_",
            string=name.lower(),
        ),
    )


def _create_timestamp() -> str:
    """Create a lexically sortable timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S%f")


def _get_latest_subpath(path: str, suffix: str) -> str | None:
    """Get the latest directory or file in the path that ends with the suffix."""
    datetime_strs: list[str] = [
        dir_name.removesuffix(suffix)
        for dir_name in os.listdir(path=path)
        if dir_name.endswith(suffix)
    ]

    if not datetime_strs:
        return None

    latest_datetime_str: str = max(datetime_strs)
    return f"{latest_datetime_str}{suffix}"
