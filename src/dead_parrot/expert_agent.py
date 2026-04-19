"""Expert agent."""

import contextlib
import os
from collections.abc import Callable
from typing import Literal

import dspy
from dspy_temporal import TemporalModule, TemporalTool

from . import utils
from .agent import Agent
from .types import Document, Examples, Metric, Models


class _AnswerGroundedInContext(dspy.Signature):
    # In DSPy, the signature docstring is used as the instruction for the LM.
    """Answer the question based on retrieved context."""

    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _Rag(dspy.Module):
    def __init__(self, name: str, retriever: Callable[[str], list[str]]) -> None:
        self._name: str = name
        self._retriever: Callable[[str], list[str]] = retriever
        self._temporal_retriever: TemporalTool = TemporalTool(
            func=retriever,
            name="expert_agent_retriever",
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


class ExpertAgent(Agent):
    """Expert agent."""

    def __init__(
        self,
        name: str,
        models: Models,
        corpus: Document | list[Document],
        dataset: Examples | list[Examples],
        metrics: dict[str, Metric],
    ) -> None:
        """Initialize the expert agent."""
        super().__init__(name=name, dataset=dataset, metrics=metrics)

        self._task_model: dspy.LM
        self._teacher_model: dspy.LM
        self._embedding_model: dspy.Embedder
        self._init_models(models=models)

        self._embeddings: dspy.retrievers.Embeddings
        self._init_embeddings(corpus=corpus)

        self._lm_program: _Rag
        self._init_lm_program()

        self._log(msg="Ready...")

    def _init_models(self, models: Models) -> None:
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

    def _init_lm_program(self) -> None:
        self._log(msg="Initializing LM program")
        assert os.path.exists(path=self.name)
        lm_program = _Rag(
            name=self.name,
            retriever=lambda query: self._embeddings(query=query).passages,
        )

        latest_lm_program_file: str | None = utils._get_latest_subpath(
            path=self.name,
            suffix="_lm_program.json",
        )
        if latest_lm_program_file is not None:
            latest_lm_program_file_path = f"{self.name}/{latest_lm_program_file}"
            self._log(msg=f"Loading from: {latest_lm_program_file_path}", sub=True)
            lm_program.load(path=latest_lm_program_file_path)
        else:
            new_lm_program_file = f"{utils._create_timestamp()}_lm_program.json"
            new_lm_program_file_path = f"{self.name}/{new_lm_program_file}"
            self._log(msg=f"Saving to: {new_lm_program_file_path}", sub=True)
            lm_program.save(path=new_lm_program_file_path)

        self._lm_program = lm_program

    def _get_task_model(self) -> dspy.LM:
        return self._task_model

    def _get_lm_program(self) -> _Rag:
        return self._lm_program

    def optimize(
        self,
        metric: str,
        effort: Literal["light", "medium", "heavy"],
    ) -> None:
        """Optimize the LM program based on the trainset."""
        self._log(msg="Optimizing LM program")
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
            optimized_lm_program: _Rag = optimizer.compile(
                student=self._lm_program,
                trainset=self._trainset,
                valset=self._devset,
            )

        with open(file=optimization_log_file_path, mode="r") as log:
            lines = log.readlines()
            last_line = lines.pop().strip() if lines else "Empty log"
        self._log(msg=f"Last log: {last_line}", sub=True)

        optimized_lm_program_file = f"{utils._create_timestamp()}_lm_program.json"
        optimized_lm_program_file_path = f"{self.name}/{optimized_lm_program_file}"
        self._log(msg=f"Saving to: {optimized_lm_program_file_path}", sub=True)
        optimized_lm_program.save(path=optimized_lm_program_file_path)

        self._lm_program = optimized_lm_program

    def to_temporal(self) -> tuple[TemporalModule[dspy.Prediction], TemporalTool]:
        """Wrap the LM program for use in Temporal workflows."""
        temporal_lm_program: TemporalModule[dspy.Prediction] = TemporalModule(
            module=self._lm_program,
            name="expert_agent",
            lm=self._task_model,
        )
        return temporal_lm_program, self._lm_program.temporal_retriever
