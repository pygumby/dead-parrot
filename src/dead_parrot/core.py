"""Core implementation of dead-parrot."""

import os
from collections.abc import Callable
from typing import Any, Protocol

import dspy


class AnswerGroundedInContext(dspy.Signature):
    """Signature for answering a question based on retrieved context."""

    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Rag(dspy.Module):
    """Module for a simple retrieval-augmented generation (RAG) pipeline."""

    def __init__(self, lm: dspy.LM, retriever: dspy.retrievers.Embeddings) -> None:
        """Initialize the module."""
        self._retriever: dspy.retrievers.Embeddings = retriever
        self._answer_grounded_in_context: dspy.ChainOfThought = dspy.ChainOfThought(
            signature=AnswerGroundedInContext,
        )
        self._answer_grounded_in_context.set_lm(lm=lm)

    def forward(self, question: str) -> dspy.Prediction:
        """Answer the question based on retrieved context."""
        context: list[str] = self._retriever(question).passages
        return self._answer_grounded_in_context(context=context, question=question)


class ExtractKeyStatements(dspy.Signature):
    """Signature for extracting key statements from an example answer."""

    example_answer: str = dspy.InputField()
    key_statements: list[str] = dspy.OutputField()


class AssessCoverage(dspy.Signature):
    """Signature for assessing the coverage of key statements in a prediction answer."""

    prediction_answer: str = dspy.InputField()
    ratio_of_key_statements_covered: float = dspy.OutputField()


class Recall(dspy.Module):
    """Module for a simple recall metric."""

    def __init__(self, lm: dspy.LM) -> None:
        """Initialize the module."""
        self._extract_key_statements = dspy.ChainOfThought(
            signature=ExtractKeyStatements
        )
        self._extract_key_statements.set_lm(lm=lm)
        self._assess_coverage = dspy.ChainOfThought(signature=AssessCoverage)
        self._assess_coverage.set_lm(lm=lm)

    def forward(self, example_answer: str, prediction_answer: str) -> float:
        """Compute the recall metric for a given example and prediction."""
        key_statements: list[str] = self._extract_key_statements(
            example_answer=example_answer
        ).key_statements

        coverage: float = self._assess_coverage(
            key_statements=key_statements,
            prediction_answer=prediction_answer,
        ).ratio_of_key_statements_covered

        assert 0 <= coverage <= 1
        return coverage


class AiAssistant(Protocol):
    """Protocol for an AI Assistant."""

    def ask(self, question: str) -> str:
        """Answer the question based on the AI Assistant's knowledge."""
        ...

    def eval(self, testset: bool = False, display_progress: bool = True) -> float:
        """Evaluate the AI Assistant based on the devset or testset."""
        ...


class AiAssistantType(Protocol):
    """Protocol for the instantiation of an AI Assistant."""

    def __call__(
        self,
        lm: str,
        embedder: str,
        corpus: list[str],
        examples: list[tuple[str, str]],
        metric: Callable[[str, str], float] | None = None,
    ) -> AiAssistant:
        """Instantiate an AI Assistant."""
        ...


class DspyAiAssistant:
    """Implementation of the protocol for an AI Assistant using dspy."""

    def __init__(
        self,
        lm: str,
        embedder: str,
        corpus: list[str],
        examples: list[tuple[str, str]],
        metric: Callable[[str, str], float] | None = None,
    ):
        """Instantiate an AI Assistant."""
        self._lm: dspy.LM = dspy.LM(model=lm)
        self._embedder: dspy.Embedder = dspy.Embedder(model=embedder, dimensions=512)
        self._corpus: list[str] = corpus
        self._trainset: list[dspy.Example]
        self._devset: list[dspy.Example]
        self._testset: list[dspy.Example]
        self._trainset, self._devset, self._testset = self._init_examples(
            examples=examples
        )
        self._metric: Callable[[dspy.Example, dspy.Prediction, Any], float] = (
            self._init_metric(metric)
        )
        self._retriever: dspy.retrievers.Embeddings = self._init_retriever(
            path="embeddings",
            k=3,
        )
        self._rag: Rag = Rag(lm=self._lm, retriever=self._retriever)

    def _init_retriever(self, path: str, k: int) -> dspy.retrievers.Embeddings:
        if os.path.exists(path):
            retriever = dspy.retrievers.Embeddings.from_saved(
                path=path,
                embedder=self._embedder,
            )
        else:
            retriever = dspy.retrievers.Embeddings(
                embedder=self._embedder,
                corpus=self._corpus,
                k=k,
            )
            retriever.save(path=path)

        return retriever

    def _init_examples(
        self, examples: list[tuple[str, str]]
    ) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
        dspy_examples = [
            dspy.Example(question=example[0], answer=example[1]).with_inputs("question")
            for example in examples
        ]

        n = len(dspy_examples)
        if n < 4:
            raise ValueError(
                f"At least 4 examples are required, but only {n} were provided."
            )

        i = n // 2
        j = n * 3 // 4

        return dspy_examples[:i], dspy_examples[i:j], dspy_examples[j:]

    def _init_metric(
        self, metric: Callable[[str, str], float] | None
    ) -> Callable[[dspy.Example, dspy.Prediction, Any], float]:
        if metric is None:
            recall = Recall(lm=self._lm)

        def dspy_metric(
            example: dspy.Example, prediction: dspy.Prediction, trace: Any = None
        ) -> float | bool:
            score: float = (
                metric(example.answer, prediction.answer)
                if metric
                else float(recall(example.answer, prediction.answer))
            )

            if trace is not None:
                return score == 1
            else:
                return score

        return dspy_metric

    def ask(self, question: str) -> str:
        """Answer the question based on the AI Assistant's knowledge."""
        pred: dspy.Prediction = self._rag(question)
        return str(pred.answer)

    def eval(self, testset: bool = False, display_progress: bool = True) -> float:
        """Evaluate the AI Assistant based on the devset or testset."""
        evaluate = dspy.Evaluate(
            devset=self._devset if not testset else self._testset,
            metric=self._metric,
            display_progress=display_progress,
        )
        result: dspy.EvaluationResult = evaluate(program=self._rag)
        return float(result.score)
