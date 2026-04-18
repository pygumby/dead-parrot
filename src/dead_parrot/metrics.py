"""Metrics."""

import functools
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import dspy

from .types import Metric, MetricResult


class _DspyMetric(Protocol):
    def __call__(
        self,
        question: str,
        example_answer: str,
        prediction_answer: str,
    ) -> MetricResult: ...


def _as_metric[**P](dspy_metric_class: Callable[P, _DspyMetric]) -> Callable[P, Metric]:
    class WrappedMetric:
        def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
            """Initialize the metric."""
            self._dspy_metric: _DspyMetric = dspy_metric_class(*args, **kwargs)

        def score(
            self,
            question: str,
            example_answer: str,
            prediction_answer: str,
        ) -> MetricResult:
            """Score the prediction answer against the example answer."""
            result: MetricResult = self._dspy_metric(
                question=question,
                example_answer=example_answer,
                prediction_answer=prediction_answer,
            )
            score = float(result["score"])
            rationale: str | None = result["rationale"]

            indent = " " * 2

            def wrap(text: str) -> str:
                return textwrap.fill(text=text, subsequent_indent=indent)

            print(
                f"Question: {wrap(question)}\n"
                f"Example answer: {wrap(example_answer)}\n"
                f"Predicted answer: {wrap(prediction_answer)}\n"
                f"Rationale: {wrap(rationale) if rationale else 'None given.'}\n"
                f"Score: {score}\n"
            )

            return {
                "score": score,
                "rationale": rationale,
            }

    functools.update_wrapper(
        wrapper=WrappedMetric,
        wrapped=dspy_metric_class,
        updated=[],
    )

    return WrappedMetric


class _GetRecallScore(dspy.Signature):
    # In DSPy, the signature docstring is used as the instruction for the LM.
    """Score recall of prediction answer against example answer given question."""

    question: str = dspy.InputField()
    example_answer: str = dspy.InputField()
    prediction_answer: str = dspy.InputField()
    recall_score: float = dspy.OutputField()
    recall_score_rationale: str = dspy.OutputField()


@_as_metric
class SimpleRecall(dspy.Module):
    """Simple recall metric."""

    def __init__(self, judge_model: str) -> None:
        """Initialize the metric."""
        self._get_recall_score = dspy.ChainOfThought(signature=_GetRecallScore)
        self._get_recall_score.set_lm(lm=dspy.LM(judge_model))

    def forward(
        self,
        question: str,
        example_answer: str,
        prediction_answer: str,
    ) -> MetricResult:
        """Score the prediction answer against the example answer."""
        prediction: dspy.Prediction = self._get_recall_score(
            question=question,
            example_answer=example_answer,
            prediction_answer=prediction_answer,
        )
        return {
            "score": prediction.recall_score,
            "rationale": prediction.recall_score_rationale,
        }


class _GetSourcesCoverage(dspy.Signature):
    # In DSPy, the signature docstring is used as the instruction for the LM.
    """Compute ratio of sources from example answer also cited in prediction answer."""

    example_answer: str = dspy.InputField()
    prediction_answer: str = dspy.InputField()
    sources_ratio: float = dspy.OutputField()
    sources_ratio_rationale: str = dspy.OutputField()


@_as_metric
class SimpleSourcesCoverage(dspy.Module):
    """Simple sources coverage metric."""

    def __init__(self, judge_model: str) -> None:
        """Initialize the metric."""
        self._get_sources_coverage = dspy.ChainOfThought(
            signature=_GetSourcesCoverage,
        )
        self._get_sources_coverage.set_lm(lm=dspy.LM(judge_model))

    def forward(
        self,
        question: str,
        example_answer: str,
        prediction_answer: str,
    ) -> MetricResult:
        """Score the prediction answer against the example answer."""
        prediction: dspy.Prediction = self._get_sources_coverage(
            example_answer=example_answer,
            prediction_answer=prediction_answer,
        )
        return {
            "score": prediction.sources_ratio,
            "rationale": prediction.sources_ratio_rationale,
        }


# Verify that protocols are correctly implemented.
if TYPE_CHECKING:
    _1: Callable[..., _DspyMetric] = SimpleRecall.__wrapped__  # Not enforced by mypy.
    _2: Callable[..., Metric] = SimpleRecall
