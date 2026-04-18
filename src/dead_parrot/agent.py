"""Agent."""

import contextlib
import random
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import dspy

from . import utils
from .types import Examples, Metric


class Agent(ABC):
    """Agent."""

    def __init__(
        self,
        name: str,
        dataset: Examples | list[Examples],
        metrics: dict[str, Metric],
    ) -> None:
        """Initialize the agent."""
        self._name: str
        self._init_name(name=name)

        self._trainset: list[dspy.Example]
        self._devset: list[dspy.Example]
        self._testset: list[dspy.Example]
        self._init_dataset(dataset=dataset)

        self._metrics: dict[
            str,
            Callable[[dspy.Example, dspy.Prediction, Any], float | bool],
        ]
        self._init_metrics(metrics=metrics)

    def _init_name(self, name: str) -> None:
        self._name = utils._normalize_name(name)

    def _init_dataset(self, dataset: Examples | list[Examples]) -> None:
        self._log(msg="Initializing dataset")
        dataset = dataset if isinstance(dataset, list) else [dataset]

        self._trainset = []
        self._devset = []
        self._testset = []

        for idx, examples in enumerate(dataset):
            self._log(msg=f"Ingesting examples: Set {idx + 1}", sub=True)
            n = len(examples.qa_pairs)

            if n < 4:
                raise ValueError(
                    f"At least 4 examples are required, but only {n} were provided."
                )

            dspy_examples = [
                dspy.Example(
                    question=example[examples.question_key],
                    answer=example[examples.answer_key],
                ).with_inputs("question")
                for example in examples.qa_pairs
            ]
            random.Random(self._name).shuffle(dspy_examples)

            i = n // 2
            j = n * 3 // 4
            self._trainset.extend(dspy_examples[:i])
            self._devset.extend(dspy_examples[i:j])
            self._testset.extend(dspy_examples[j:])

        n_train = len(self._trainset)
        n_dev = len(self._devset)
        n_test = len(self._testset)
        n_total = n_train + n_dev + n_test
        self._log(msg=f"Train examples: {n_train}", sub=True)
        self._log(msg=f"Dev examples: {n_dev}", sub=True)
        self._log(msg=f"Test examples: {n_test}", sub=True)
        self._log(msg=f"Total examples: {n_total}", sub=True)

    def _init_metrics(self, metrics: dict[str, Metric]) -> None:
        self._log(msg="Initializing metrics")

        def make_dspy_metric(
            metric: Metric,
        ) -> Callable[[dspy.Example, dspy.Prediction, Any], float | bool]:
            def dspy_metric(
                example: dspy.Example,
                prediction: dspy.Prediction,
                trace: Any = None,
            ) -> float | bool:
                result = metric.score(
                    question=example.question,
                    example_answer=example.answer,
                    prediction_answer=prediction.answer,
                )
                score: float = result["score"]

                if not 0 <= score <= 1:
                    raise ValueError(f"Score must be between 0 and 1, but got {score}.")

                if trace is not None:
                    return score >= 0.95
                else:
                    return score

            return dspy_metric

        self._metrics = {}
        for name, metric in metrics.items():
            self._log(msg=f"Metric: {name}", sub=True)
            self._metrics[name] = make_dspy_metric(metric=metric)

    def _log(self, msg: str, sub: bool = False, indent: int = 2) -> None:
        indent_str = " " * indent
        if not sub:
            text = f"[{self.name}] {msg}"
            sub_indent_str = indent_str
        else:
            text = f"{indent_str}{msg}"
            sub_indent_str = indent_str * 2
        print(textwrap.fill(text=text, subsequent_indent=sub_indent_str))

    @abstractmethod
    def _get_task_model(self) -> dspy.LM: ...

    @abstractmethod
    def _get_lm_program(self) -> dspy.Module: ...

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self._name

    def ask(self, question: str) -> dict[str, Any]:
        """Query the agent with a question."""
        self._log(msg="Querying the agent")
        self._log(msg=f"Question: {question}", sub=True)

        with dspy.context(lm=self._get_task_model()):
            pred_dict: dict[str, Any] = self._get_lm_program()(
                question=question
            ).toDict()

        self._log(msg=f"Answer: {pred_dict['answer']}", sub=True)
        return pred_dict

    def evaluate(self, metric: str, use_testset: bool = False) -> float:
        """Evaluate the agent based on the devset or testset."""
        self._log(msg="Evaluating the agent")
        self._log(msg=f"Metric: {metric}", sub=True)

        if metric not in self._metrics:
            raise ValueError(
                f"Metric '{metric}' not found in metrics:\n{list(self._metrics.keys())}"
            )

        dspy_evaluate = dspy.Evaluate(
            devset=self._devset if not use_testset else self._testset,
            metric=self._metrics[metric],
        )

        evaluation_log_file = f"{utils._create_timestamp()}_evaluation.log"
        evaluation_log_file_path = f"{self.name}/{evaluation_log_file}"
        self._log(msg=f"Logging to: {evaluation_log_file_path}", sub=True)
        with (
            open(file=evaluation_log_file_path, mode="w") as log,
            contextlib.redirect_stdout(new_target=log),
            contextlib.redirect_stderr(new_target=log),
            dspy.context(lm=self._get_task_model()),
        ):
            result: dspy.EvaluationResult = dspy_evaluate(
                program=self._get_lm_program()
            )

        score = float(result.score)
        self._log(msg=f"Score: {score}", sub=True)

        return score
