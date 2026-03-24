"""Utility functions."""

import json
import os
from datetime import datetime

import pypdf


def load_corpus_from_pdf(
    name: str,
    path: str,
    chunk_size: int = 1000,
    prepend_metadata: bool = True,
) -> list[str]:
    """Load text from a PDF file and split it into chunks for retrieval."""
    reader = pypdf.PdfReader(stream=path)
    corpus: list[str] = []

    for i in range(len(reader.pages)):
        page_text: str = reader.pages[i].extract_text()
        for j in range(0, len(page_text), chunk_size):
            chunk = page_text[j : j + chunk_size]
            corpus.append(
                f"Document: {name}\nPage: {i}\n{chunk}" if prepend_metadata else chunk
            )

    return corpus


def load_examples_from_json(path: str) -> list[tuple[str, str]]:
    """Load examples from a JSON file."""
    with open(file=path, mode="r") as file:
        examples: list[list[str]] = json.load(fp=file)
    return [(example[0], example[1]) for example in examples]


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
