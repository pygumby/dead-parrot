"""Protocols."""

from collections.abc import Callable
from typing import Literal, Protocol


class AiAssistant(Protocol):
    """Protocol for an AI assistant."""

    @property
    def name(self) -> str:
        """Return the name of the AI assistant."""
        ...

    def ask(self, question: str) -> str:
        """Answer the question using the RAG pipeline."""
        ...

    def evaluate(self, use_testset: bool = False) -> float:
        """Evaluate the RAG pipeline based on the devset or testset."""
        ...

    def optimize(self) -> None:
        """Optimize the RAG pipeline based on the trainset."""
        ...


class AiAssistantType(Protocol):
    """Protocol for the instantiation of an AI assistant."""

    def __call__(
        self,
        name: str,
        task_model: str,
        teacher_model: str,
        embedding_model: str,
        corpus: list[str],
        examples: list[tuple[str, str]],
        metric: Callable[[str, str], float] | None = None,
        optimization_effort: Literal["light", "medium", "heavy"] = "light",
    ) -> AiAssistant:
        """Instantiate the AI assistant."""
        ...
