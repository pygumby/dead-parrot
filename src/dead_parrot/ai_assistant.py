"""AI Assistant."""

import contextlib
import os
import random
import re
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import dspy
from dspy_temporal import TemporalModule, TemporalTool

from . import utils
from .protocols import (
    AiAssistant,
    AiAssistantClass,
    Document,
    Examples,
    Metric,
    Models,
)


class _AnswerGroundedInContext(dspy.Signature):
    # In DSPy, the signature docstring is used as the instruction for the LM.
    """Answer the question based on retrieved context."""

    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _Rag(dspy.Module):
    def __init__(
        self,
        name: str,
        lm: dspy.LM,
        retriever: Callable[[str], list[str]],
    ) -> None:
        self._name: str = name
        self._lm: dspy.LM = lm
        self._retriever: Callable[[str], list[str]] = retriever
        self._temporal_retriever: TemporalTool = TemporalTool(
            func=retriever,
            name=f"{name}_retriever",
        )
        self._answer_grounded_in_context: dspy.ChainOfThought = dspy.ChainOfThought(
            signature=_AnswerGroundedInContext,
        )

    @property
    def temporal_retriever(self) -> TemporalTool:
        return self._temporal_retriever

    def forward(self, question: str) -> dspy.Prediction:
        context: list[str] = self._retriever(question)
        with dspy.context(lm=self._lm):
            return self._answer_grounded_in_context(
                context=context,
                question=question,
            )

    async def aforward(self, question: str) -> dspy.Prediction:
        context: list[str] = await self._temporal_retriever.run(question)
        return await self._answer_grounded_in_context.acall(
            context=context,
            question=question,
        )


class DspyAiAssistant(AiAssistant):
    """Implementation of the protocol for an AI assistant using dspy."""

    def __init__(
        self,
        name: str,
        models: Models,
        corpus: Document | list[Document],
        dataset: Examples | list[Examples],
        metrics: dict[str, Metric],
    ) -> None:
        """Initialize the AI assistant."""
        self._name: str
        self._init_name(name=name)

        self._task_model: dspy.LM
        self._teacher_model: dspy.LM
        self._embedding_model: dspy.Embedder
        self._init_models(models=models)

        self._embeddings: dspy.retrievers.Embeddings
        self._init_embeddings(corpus=corpus)

        self._trainset: list[dspy.Example]
        self._devset: list[dspy.Example]
        self._testset: list[dspy.Example]
        self._init_dataset(dataset=dataset)

        self._metrics: dict[
            str,
            Callable[[dspy.Example, dspy.Prediction, Any], float | bool],
        ]
        self._init_metrics(metrics=metrics)

        self._rag: _Rag
        self._init_rag()

        self._log(msg="Ready...")

    def _init_name(self, name: str) -> None:
        if not re.search(pattern=r"[a-zA-Z0-9]", string=name):
            raise ValueError(
                f"Name must contain letters or numbers, but '{name}' does not."
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
        models: Models,
    ) -> None:
        self._log(msg="Initializing models")
        self._log(msg=f"Task model: {models.task}", sub=True)
        self._task_model = dspy.LM(model=models.task)
        self._log(msg=f"Teacher model: {models.teacher}", sub=True)
        self._teacher_model = dspy.LM(model=models.teacher)
        self._log(msg=f"Embedding model: {models.embedding}", sub=True)
        self._embedding_model = dspy.Embedder(model=models.embedding)

    def _init_embeddings(self, corpus: Document | list[Document]) -> None:
        self._log(msg="Initializing embeddings")
        corpus = corpus if isinstance(corpus, list) else [corpus]

        def make_chunks(corpus: list[Document]) -> list[str]:
            chunks: list[str] = []
            for doc in corpus:
                self._log(msg=f"Chunking document: {doc.name}", sub=True)
                chunk_overlap: int = doc.chunk_size // 5
                doc_chunks: list[str] = []
                for i, page in enumerate(doc.pages):
                    page_metadata = f"Document: {doc.name}\nPage: {i + 1}\n"
                    for j in range(0, len(page), doc.chunk_size - chunk_overlap):
                        chunk = f"{page_metadata}{page[j : j + doc.chunk_size]}"
                        doc_chunks.append(chunk)
                chunks.extend(doc_chunks)
            return chunks

        if os.path.exists(path=self.name):
            latest_embeddings_dir: str | None = utils.get_latest_subpath(
                path=self.name,
                suffix="_embeddings",
            )
            assert latest_embeddings_dir is not None
            latest_embeddings_dir_path = f"{self.name}/{latest_embeddings_dir}"
            self._log(msg=f"Loading from: {latest_embeddings_dir_path}", sub=True)
            embeddings = dspy.retrievers.Embeddings.from_saved(
                path=latest_embeddings_dir_path,
                embedder=self._embedding_model,
            )
        else:
            chunks: list[str] = make_chunks(corpus=corpus)
            self._log(msg=f"Total chunks to embed: {len(chunks)}", sub=True)
            embeddings = dspy.retrievers.Embeddings(
                embedder=self._embedding_model,
                corpus=chunks,
            )
            new_embeddings_dir = f"{utils.create_timestamp()}_embeddings"
            new_embeddings_dir_path = f"{self.name}/{new_embeddings_dir}"
            self._log(msg=f"Saving to: {new_embeddings_dir_path}", sub=True)
            os.makedirs(name=self.name)
            embeddings.save(path=new_embeddings_dir_path)

        self._embeddings = embeddings

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

    def _init_rag(self) -> None:
        self._log(msg="Initializing RAG pipeline")
        assert os.path.exists(path=self.name)
        rag = _Rag(
            name=self.name,
            lm=self._task_model,
            retriever=lambda query: self._embeddings(query=query).passages,
        )

        latest_rag_file: str | None = utils.get_latest_subpath(
            path=self.name,
            suffix="_rag.json",
        )
        if latest_rag_file is not None:
            latest_rag_file_path = f"{self.name}/{latest_rag_file}"
            self._log(msg=f"Loading from: {latest_rag_file_path}", sub=True)
            rag.load(path=latest_rag_file_path)
        else:
            new_rag_file = f"{utils.create_timestamp()}_rag.json"
            new_rag_file_path = f"{self.name}/{new_rag_file}"
            self._log(msg=f"Saving to: {new_rag_file_path}", sub=True)
            rag.save(path=new_rag_file_path)

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

    def ask(self, question: str) -> dict[str, Any]:
        """Answer the question using the RAG pipeline."""
        self._log(msg="Performing inference")
        self._log(msg=f"Question: {question}", sub=True)
        pred_dict: dict[str, Any] = self._rag(question=question).toDict()
        self._log(msg=f"Answer: {pred_dict['answer']}", sub=True)
        return pred_dict

    def evaluate(self, metric: str, use_testset: bool = False) -> float:
        """Evaluate the RAG pipeline based on the devset or testset."""
        self._log(msg="Evaluating RAG pipeline")
        self._log(msg=f"Metric: {metric}", sub=True)

        if metric not in self._metrics:
            raise ValueError(
                f"Metric '{metric}' not found in metrics:\n{list(self._metrics.keys())}"
            )

        dspy_evaluate = dspy.Evaluate(
            devset=self._devset if not use_testset else self._testset,
            metric=self._metrics[metric],
        )

        evaluation_log_file = f"{utils.create_timestamp()}_evaluation.log"
        evaluation_log_file_path = f"{self.name}/{evaluation_log_file}"
        self._log(msg=f"Logging to: {evaluation_log_file_path}", sub=True)
        with (
            open(file=evaluation_log_file_path, mode="w") as log,
            contextlib.redirect_stdout(new_target=log),
            contextlib.redirect_stderr(new_target=log),
        ):
            result: dspy.EvaluationResult = dspy_evaluate(program=self._rag)

        score = float(result.score)
        self._log(msg=f"Score: {score}", sub=True)

        return score

    def optimize(
        self,
        metric: str,
        effort: Literal["light", "medium", "heavy"],
    ) -> None:
        """Optimize the RAG pipeline based on the trainset."""
        self._log(msg="Optimizing RAG pipeline")
        self._log(msg=f"Metric: {metric}", sub=True)
        self._log(msg=f"Effort: {effort}", sub=True)

        if metric not in self._metrics:
            raise ValueError(
                f"Metric '{metric}' not found in metrics:\n{list(self._metrics.keys())}"
            )

        optimizer = dspy.MIPROv2(
            metric=self._metrics[metric],
            prompt_model=self._teacher_model,
            task_model=self._task_model,
            auto=effort,
        )

        optimization_log_file = f"{utils.create_timestamp()}_optimization.log"
        optimization_log_file_path = f"{self.name}/{optimization_log_file}"
        self._log(msg=f"Logging to: {optimization_log_file_path}", sub=True)
        with (
            open(file=optimization_log_file_path, mode="w") as log,
            contextlib.redirect_stdout(new_target=log),
            contextlib.redirect_stderr(new_target=log),
        ):
            optimized_rag: _Rag = optimizer.compile(
                student=self._rag,
                trainset=self._trainset,
                valset=self._devset,
            )

        with open(file=optimization_log_file_path, mode="r") as log:
            lines = log.readlines()
            last_line = lines.pop().strip() if lines else "Empty log"
        self._log(msg=f"Last log: {last_line}", sub=True)

        optimized_rag_file = f"{utils.create_timestamp()}_rag.json"
        optimized_rag_file_path = f"{self.name}/{optimized_rag_file}"
        self._log(msg=f"Saving to: {optimized_rag_file_path}", sub=True)
        optimized_rag.save(path=optimized_rag_file_path)

        self._rag = optimized_rag

    def to_temporal(self) -> tuple[TemporalModule[dspy.Prediction], TemporalTool]:
        """Wrap the RAG pipeline for use in Temporal workflows."""
        temporal_rag: TemporalModule[dspy.Prediction] = TemporalModule(
            module=self._rag,
            name=f"{self.name}_rag",
            lm=self._task_model,
        )
        return temporal_rag, self._rag.temporal_retriever


# Verify that protocols are correctly implemented
if TYPE_CHECKING:
    _1: AiAssistantClass = DspyAiAssistant
