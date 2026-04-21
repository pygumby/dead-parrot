"""Types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from .expert_agent import ExpertAgent
    from .expert_agent_client import ExpertAgentClient
    from .triage_agent import TriageAgent


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


class ExpertAgentClass(Protocol):
    """Protocol for the expert agent class."""

    def __call__(  # noqa: D102
        self,
        name: str,
        models: Models,
        corpus: Document | list[Document],
        dataset: Examples | list[Examples],
        metrics: dict[str, Metric],
    ) -> ExpertAgent: ...


class ExpertAgentClientClass(Protocol):
    """Protocol for the expert agent client class."""

    def __call__(  # noqa: D102
        self,
        scheme: Literal["http", "https"],
        host: str,
        port: int,
        ask_endpoint: str = "ask",
        card_endpoint: str = "card",
        timeout: int = 60,
    ) -> ExpertAgentClient: ...


class TriageAgentClass(Protocol):
    """Protocol for the triage agent class."""

    def __call__(  # noqa: D102
        self,
        name: str,
        task_model: str,
        expert_agent_clients: list[ExpertAgentClient],
        dataset: Examples | list[Examples],
        metrics: dict[str, Metric],
    ) -> TriageAgent: ...
