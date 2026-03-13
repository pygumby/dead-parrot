"""Metrics."""

import dspy


class ExtractKeyStatements(dspy.Signature):
    # TODO DSPy assigns meaning to signature docstrings -> review
    """Signature for extracting key statements from an example answer."""

    example_answer: str = dspy.InputField()
    key_statements: list[str] = dspy.OutputField()


class AssessCoverage(dspy.Signature):
    # TODO DSPy assigns meaning to signature docstrings -> review
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

        if not 0 <= coverage <= 1:
            raise ValueError(f"Coverage must be between 0 and 1, but got {coverage}.")

        return coverage
