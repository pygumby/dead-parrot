"""Metrics."""

import functools
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import dspy

from .types import Metric, MetricResult


def _round_score(score: float, decimals: int = 2) -> float:
    return round(score, decimals)


def _print_score(
    question: str,
    example_answer: str,
    prediction_answer: str,
    score: float,
    rationale: str | None,
) -> None:
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
            print_output: bool = True,
        ) -> MetricResult:
            """Score the prediction answer against the example answer."""
            result: MetricResult = self._dspy_metric(
                question=question,
                example_answer=example_answer,
                prediction_answer=prediction_answer,
            )
            score = _round_score(result["score"])
            rationale: str | None = result["rationale"]

            if print_output:
                _print_score(
                    question=question,
                    example_answer=example_answer,
                    prediction_answer=prediction_answer,
                    score=score,
                    rationale=rationale,
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
class Recall(dspy.Module):
    """Recall score metric."""

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
class Sources(dspy.Module):
    """Sources coverage metric."""

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


class Length(Metric):
    """Length similarity metric."""

    def score(
        self,
        question: str,
        example_answer: str,
        prediction_answer: str,
        print_output: bool = True,
    ) -> MetricResult:
        """Score the prediction answer against the example answer."""
        example_answer_len = len(example_answer.split())
        prediction_answer_len = len(prediction_answer.split())

        if example_answer_len == 0 or prediction_answer_len == 0:
            ratio = 0.0
        else:
            min_len: int = min(example_answer_len, prediction_answer_len)
            max_len: int = max(example_answer_len, prediction_answer_len)
            ratio = min_len / max_len

        score: float = _round_score(ratio)
        rationale = (
            f"Example answer length: {example_answer_len} words, "
            f"prediction answer length: {prediction_answer_len} words."
        )

        if print_output:
            _print_score(
                question=question,
                example_answer=example_answer,
                prediction_answer=prediction_answer,
                score=score,
                rationale=rationale,
            )

        return {"score": score, "rationale": rationale}


class Composite(Metric):
    """Composite metric."""

    def __init__(
        self,
        judge_model: str,
        recall_weight: float = 0.5,
        sources_weight: float = 0.3,
        length_weight: float = 0.2,
    ) -> None:
        """Initialize the metric."""
        if recall_weight + sources_weight + length_weight != 1.0:
            raise ValueError("Weights must sum to 1.")

        self._recall = Recall(judge_model=judge_model)
        self._sources = Sources(judge_model=judge_model)
        self._length = Length()
        self._recall_weight = recall_weight
        self._sources_weight = sources_weight
        self._length_weight = length_weight

    def score(
        self,
        question: str,
        example_answer: str,
        prediction_answer: str,
        print_output: bool = True,
    ) -> MetricResult:
        """Score the prediction answer against the example answer."""
        recall_score = self._recall.score(
            question=question,
            example_answer=example_answer,
            prediction_answer=prediction_answer,
            print_output=False,
        )["score"]
        sources_score = self._sources.score(
            question=question,
            example_answer=example_answer,
            prediction_answer=prediction_answer,
            print_output=False,
        )["score"]
        length_score = self._length.score(
            question=question,
            example_answer=example_answer,
            prediction_answer=prediction_answer,
            print_output=False,
        )["score"]

        score = _round_score(
            self._recall_weight * recall_score
            + self._sources_weight * sources_score
            + self._length_weight * length_score
        )
        rationale = (
            f"Recall score: {recall_score}, "
            f"sources score: {sources_score}, "
            f"length score: {length_score}."
        )

        if print_output:
            _print_score(
                question=question,
                example_answer=example_answer,
                prediction_answer=prediction_answer,
                score=score,
                rationale=rationale,
            )

        return {"score": score, "rationale": rationale}


# Verify that protocols are correctly implemented.
if TYPE_CHECKING:
    _1: Callable[..., Metric] = Recall
    _2: Callable[..., Metric] = Sources
    _3: Callable[..., Metric] = Length
    _4: Callable[..., Metric] = Composite
