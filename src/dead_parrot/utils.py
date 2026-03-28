"""Utility functions."""

import json
import os
from datetime import datetime

import pypdf


def load_pages_from_pdf(path: str) -> list[str]:
    """Load the text of each page from a PDF file."""
    reader = pypdf.PdfReader(stream=path)
    return [reader.pages[i].extract_text() for i in range(len(reader.pages))]


def load_dicts_from_json(path: str) -> list[dict[str, str]]:
    """Load a list of dicts from a JSON file."""
    with open(file=path) as file:
        data = json.load(fp=file)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"JSON in {path} doesn't contain a list of dicts.")

    return data


def create_timestamp() -> str:
    """Create a lexically sortable timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S%f")


def get_latest_subpath(path: str, suffix: str) -> str | None:
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
