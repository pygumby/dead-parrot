"""AI Assistant."""

import contextlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import dspy
from dspy_temporal import TemporalModule, TemporalTool

from . import utils
from .agent import Agent
from .protocols import (
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
        retriever: Callable[[str], list[str]],
    ) -> None:
        self._name: str = name
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


class DspyAiAssistant(Agent):
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
        super().__init__(name=name, dataset=dataset, metrics=metrics)

        self._task_model: dspy.LM
        self._teacher_model: dspy.LM
        self._embedding_model: dspy.Embedder
        self._init_models(models=models)

        self._embeddings: dspy.retrievers.Embeddings
        self._init_embeddings(corpus=corpus)

        self._rag: _Rag
        self._init_rag()

        self._log(msg="Ready...")

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
            latest_embeddings_dir: str | None = utils._get_latest_subpath(
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
            new_embeddings_dir = f"{utils._create_timestamp()}_embeddings"
            new_embeddings_dir_path = f"{self.name}/{new_embeddings_dir}"
            self._log(msg=f"Saving to: {new_embeddings_dir_path}", sub=True)
            os.makedirs(name=self.name)
            embeddings.save(path=new_embeddings_dir_path)

        self._embeddings = embeddings

    def _init_rag(self) -> None:
        self._log(msg="Initializing RAG pipeline")
        assert os.path.exists(path=self.name)
        rag = _Rag(
            name=self.name,
            retriever=lambda query: self._embeddings(query=query).passages,
        )

        latest_rag_file: str | None = utils._get_latest_subpath(
            path=self.name,
            suffix="_rag.json",
        )
        if latest_rag_file is not None:
            latest_rag_file_path = f"{self.name}/{latest_rag_file}"
            self._log(msg=f"Loading from: {latest_rag_file_path}", sub=True)
            rag.load(path=latest_rag_file_path)
        else:
            new_rag_file = f"{utils._create_timestamp()}_rag.json"
            new_rag_file_path = f"{self.name}/{new_rag_file}"
            self._log(msg=f"Saving to: {new_rag_file_path}", sub=True)
            rag.save(path=new_rag_file_path)

        self._rag = rag

    def _get_task_model(self) -> dspy.LM:
        """Return the task model of the agent."""
        return self._task_model

    def _get_module(self) -> _Rag:
        """Return the dspy.Module of the agent."""
        return self._rag

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

        optimization_log_file = f"{utils._create_timestamp()}_optimization.log"
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

        optimized_rag_file = f"{utils._create_timestamp()}_rag.json"
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
