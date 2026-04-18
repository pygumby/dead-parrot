"""Types."""

from dataclasses import dataclass
from typing import Protocol, TypedDict


@dataclass
class Models:
    """Models of an agent."""

    task: str
    teacher: str
    embedding: str


@dataclass
class Document:
    """Document in the corpus of an agent."""

    name: str
    pages: list[str]
    chunk_size: int = 1000


@dataclass
class Examples:
    """Examples in the dataset of an agent."""

    qa_pairs: list[dict[str, str]]
    question_key: str = "question"
    answer_key: str = "answer"


class MetricResult(TypedDict):
    """Result of a metric's scoring."""

    score: float
    rationale: str | None


class Metric(Protocol):
    """Metric."""

    def score(
        self,
        question: str,
        example_answer: str,
        prediction_answer: str,
    ) -> MetricResult:
        """Score the predicted answer given the question and example answer."""
        ...
