"""Protocols."""

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


class MetricResult(TypedDict):
    """TypedDict for the result of a metric's scoring."""

    score: float
    rationale: str | None


class Metric(Protocol):
    """Protocol for a metric."""

    def score(
        self,
        question: str,
        example_answer: str,
        prediction_answer: str,
    ) -> MetricResult:
        """Score the predicted answer given the question and example answer."""
        ...


class AiAssistant(Protocol):
    """Protocol for an AI assistant."""

    @property
    def name(self) -> str:
        """Return the name of the AI assistant."""
        ...

    def ask(self, question: str) -> str:
        """Answer the question using the RAG pipeline."""
        ...

    def evaluate(self, metric: str, use_testset: bool = False) -> float:
        """Evaluate the RAG pipeline based on the devset or testset."""
        ...

    def optimize(
        self,
        metric: str,
        effort: Literal["light", "medium", "heavy"],
    ) -> None:
        """Optimize the RAG pipeline based on the trainset."""
        ...


@dataclass
class Corpus:
    """Dataclass for the corpus of an AI assistant."""

    name: str
    pages: list[str]
    chunk_size: int


class AiAssistantClass(Protocol):
    """Protocol for the instantiation of an AI assistant."""

    def __call__(
        self,
        name: str,
        task_model: str,
        teacher_model: str,
        embedding_model: str,
        corpus: Corpus,
        examples: list[tuple[str, str]],
        metrics: dict[str, Metric],
    ) -> AiAssistant:
        """Instantiate the AI assistant."""
        ...
