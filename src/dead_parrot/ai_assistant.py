"""AI Assistant."""

import contextlib
import os
import re
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import dspy

from . import utils
from .metrics import Recall
from .protocols import AiAssistant, AiAssistantType
from .rag import Rag


class DspyAiAssistant(AiAssistant):
    """Implementation of the protocol for an AI assistant using dspy."""

    def __init__(
        self,
        name: str,
        task_model: str,
        teacher_model: str,
        embedding_model: str,
        corpus: list[str],
        examples: list[tuple[str, str]],
        metric: Callable[[str, str], float] | None = None,
        optimization_effort: Literal["light", "medium", "heavy"] = "light",
    ):
        """Initialize the AI assistant."""
        self._name: str
        self._init_name(name=name)

        self._task_model: dspy.LM
        self._teacher_model: dspy.LM
        self._embedding_model: dspy.Embedder
        self._init_models(
            task_model=task_model,
            teacher_model=teacher_model,
            embedding_model=embedding_model,
        )

        self._corpus: list[str]
        self._init_corpus(corpus=corpus)

        self._trainset: list[dspy.Example]
        self._devset: list[dspy.Example]
        self._testset: list[dspy.Example]
        self._init_examples(examples=examples)

        self._metric: Callable[[dspy.Example, dspy.Prediction, Any], float | bool]
        self._init_metric(metric=metric)

        self._optimizer: dspy.MIPROv2
        self._init_optimizer(effort=optimization_effort)

        self._embeddings_id: str
        self._retriever: dspy.retrievers.Embeddings
        self._init_retriever(k=3)

        self._rag_id: str
        self._rag: Rag
        self._init_rag()

    def _init_name(self, name: str) -> None:
        if not re.search(pattern=r"[a-zA-Z0-9]", string=name):
            raise ValueError(
                f"Name must contain letters or numbers, but {name} does not."
            )
        self._name = re.sub(
            pattern=r"[^a-z0-9_]",
            repl="",
            string=re.sub(
                pattern=r"[ -]",
                repl="_",
                string=name.lower(),
            ),
        )

    def _init_models(
        self,
        task_model: str,
        teacher_model: str,
        embedding_model: str,
    ) -> None:
        self._log(msg="Initializing models")

        self._log(msg=f"Task model: {task_model}", sub=True)
        self._task_model = dspy.LM(model=task_model)

        self._log(msg=f"Teacher model: {teacher_model}", sub=True)
        self._teacher_model = dspy.LM(model=teacher_model)

        self._log(msg=f"Embedding model: {embedding_model}", sub=True)
        self._embedding_model = dspy.Embedder(model=embedding_model)

    def _init_corpus(self, corpus: list[str]) -> None:
        self._log(msg="Initializing corpus")

        self._log(msg=f"Total chunks: {len(corpus)}", sub=True)
        self._corpus = corpus

    def _init_examples(self, examples: list[tuple[str, str]]) -> None:
        self._log(msg="Initializing examples")

        n = len(examples)

        if n < 4:
            raise ValueError(
                f"At least 4 examples are required, but only {n} were provided."
            )

        i = n // 2
        j = n * 3 // 4

        self._log(msg=f"Total pairs: {n}", sub=True)
        self._log(msg=f"Train pairs: {i}", sub=True)
        self._log(msg=f"Dev pairs: {j - i}", sub=True)
        self._log(msg=f"Test pairs: {n - j}", sub=True)

        dspy_examples = [
            dspy.Example(question=example[0], answer=example[1]).with_inputs("question")
            for example in examples
        ]

        self._trainset = dspy_examples[:i]
        self._devset = dspy_examples[i:j]
        self._testset = dspy_examples[j:]

    def _init_metric(self, metric: Callable[[str, str], float] | None) -> None:
        self._log(msg="Initializing metric")
        self._log(msg=f"Metric: {'Recall' if not metric else 'Custom'}", sub=True)

        if metric is None:
            recall = Recall(lm=self._teacher_model)

        def dspy_metric(
            example: dspy.Example,
            prediction: dspy.Prediction,
            trace: Any = None,
        ) -> float | bool:
            score: float = (
                metric(example.answer, prediction.answer)
                if metric
                else float(
                    recall(
                        example_answer=example.answer,
                        prediction_answer=prediction.answer,
                    )
                )
            )

            if trace is not None:
                return score >= 0.95
            else:
                return score

        self._metric = dspy_metric

    def _init_optimizer(
        self,
        effort: Literal["light", "medium", "heavy"],
    ) -> None:
        self._log(msg="Initializing optimizer")
        self._log(msg=f"Effort: {effort}", sub=True)

        self._optimizer = dspy.MIPROv2(
            metric=self._metric,
            prompt_model=self._teacher_model,
            task_model=self._task_model,
            auto=effort,
        )

    def _init_retriever(self, k: int) -> None:
        self._log(msg="Initializing retriever")

        if os.path.exists(path=self.name):
            latest_embeddings_id: str | None = utils.get_latest_directory(
                path=self.name,
                prefix="embeddings_",
            )

            path = f"{self.name}/{latest_embeddings_id}"
            self._log(msg=f"Loading from: {path}", sub=True)

            assert latest_embeddings_id is not None

            retriever = dspy.retrievers.Embeddings.from_saved(
                path=f"{self.name}/{latest_embeddings_id}",
                embedder=self._embedding_model,
            )
            self._embeddings_id = latest_embeddings_id
        else:
            new_embeddings_id = f"embeddings_{utils.create_timestamp()}"
            path = f"{self.name}/{new_embeddings_id}"
            self._log(msg=f"Saving to: {path}", sub=True)

            retriever = dspy.retrievers.Embeddings(
                embedder=self._embedding_model,
                corpus=self._corpus,
                k=k,
            )
            os.makedirs(name=self.name)
            retriever.save(path=path)
            self._embeddings_id = new_embeddings_id

        self._retriever = retriever

    def _init_rag(self) -> None:
        self._log(msg="Initializing RAG pipeline")

        assert os.path.exists(path=self.name)

        rag = Rag(lm=self._task_model, retriever=self._retriever)

        latest_rag_id: str | None = utils.get_latest_directory(
            path=self.name,
            prefix="rag_",
        )

        if latest_rag_id is not None:
            path = f"{self.name}/{latest_rag_id}/rag.json"
            self._log(msg=f"Loading from: {path}", sub=True)

            rag.load(path=path)
            self._rag_id = latest_rag_id
        else:
            new_rag_id = f"rag_{utils.create_timestamp()}"
            path = f"{self.name}/{new_rag_id}"
            self._log(msg=f"Saving to: {path}/rag.json", sub=True)

            os.makedirs(name=path)
            rag.save(path=f"{path}/rag.json")
            self._rag_id = new_rag_id

        self._rag = rag

    def _log(self, msg: str, sub: bool = False, indent: int = 2) -> None:
        indent_str = " " * indent
        if not sub:
            text = f"[{self.name}] {msg}"
            sub_indent_str = indent_str
        else:
            text = f"{indent_str}{msg}"
            sub_indent_str = indent_str * 2
        print(textwrap.fill(text=text, subsequent_indent=sub_indent_str))

    @property
    def name(self) -> str:
        """Return the name of the AI assistant."""
        return self._name

    @property
    def embeddings_id(self) -> str:
        """Return the ID of the embeddings."""
        return self._embeddings_id

    @property
    def rag_id(self) -> str:
        """Return the ID of the RAG pipeline."""
        return self._rag_id

    def ask(self, question: str) -> str:
        """Answer the question using the RAG pipeline."""
        self._log(msg="Performing inference")

        pred: dspy.Prediction = self._rag(question=question)

        self._log(msg=f"Question: {question}", sub=True)
        self._log(msg=f"Answer: {pred.answer}", sub=True)

        return str(pred.answer)

    def evaluate(self, use_testset: bool = False) -> float:
        """Evaluate the RAG pipeline based on the devset or testset."""
        self._log(msg="Evaluating RAG pipeline")

        evaluate = dspy.Evaluate(
            devset=self._devset if not use_testset else self._testset,
            metric=self._metric,
        )

        path = f"{self.name}/{self.rag_id}/evaluation_{utils.create_timestamp()}.log"
        self._log(msg=f"Logging to: {path}", sub=True)
        with (
            open(file=path, mode="w") as log,
            contextlib.redirect_stdout(new_target=log),
            contextlib.redirect_stderr(new_target=log),
        ):
            result: dspy.EvaluationResult = evaluate(program=self._rag)

        score = float(result.score)
        self._log(msg=f"Score: {score}", sub=True)

        return score

    def optimize(self) -> None:
        """Optimize the RAG pipeline based on the trainset."""
        self._log(msg="Optimizing RAG pipeline")

        optimized_rag_id = f"rag_{utils.create_timestamp()}"
        path = f"{self.name}/{optimized_rag_id}"

        self._log(msg=f"Logging to: {path}/optimization.log", sub=True)
        os.makedirs(name=path)
        with (
            open(file=f"{path}/optimization.log", mode="w") as log,
            contextlib.redirect_stdout(new_target=log),
            contextlib.redirect_stderr(new_target=log),
        ):
            optimized_rag: Rag = self._optimizer.compile(
                student=self._rag,
                trainset=self._trainset,
                valset=self._devset,
            )

        with open(file=f"{path}/optimization.log", mode="r") as log:
            last_log = log.readlines()[-1].strip()
        self._log(msg=f"Last log: {last_log}", sub=True)

        self._log(msg=f"Saving to: {path}/rag.json", sub=True)
        optimized_rag.save(path=f"{path}/rag.json")

        self._rag_id = optimized_rag_id
        self._rag = optimized_rag


# Verify that DspyAiAssistant conforms to the AiAssistantType protocol
if TYPE_CHECKING:
    _: AiAssistantType = DspyAiAssistant
