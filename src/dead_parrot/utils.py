"""Utility functions for loading and processing data."""

import json

import pypdf


def load_corpus_from_pdf(
    name: str,
    path: str,
    chunk_size: int = 1000,
    prepend_metadata: bool = True,
) -> list[str]:
    """Load text from a PDF file and split it into chunks for retrieval."""
    reader = pypdf.PdfReader(path)
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
    with open(path, "r") as file:
        examples: list[list[str]] = json.load(fp=file)

    return [(example[0], example[1]) for example in examples]
