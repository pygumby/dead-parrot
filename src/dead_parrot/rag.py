"""RAG pipeline."""

import dspy


class AnswerGroundedInContext(dspy.Signature):
    # TODO DSPy assigns meaning to signature docstrings -> review
    """Signature for answering a question based on retrieved context."""

    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Rag(dspy.Module):
    """Module for a RAG pipeline."""

    def __init__(self, lm: dspy.LM, retriever: dspy.retrievers.Embeddings) -> None:
        """Initialize the module."""
        self._retriever: dspy.retrievers.Embeddings = retriever
        self._answer_grounded_in_context: dspy.ChainOfThought = dspy.ChainOfThought(
            signature=AnswerGroundedInContext,
        )
        self._answer_grounded_in_context.set_lm(lm=lm)

    def forward(self, question: str) -> dspy.Prediction:
        """Answer the question based on retrieved context."""
        context: list[str] = self._retriever(query=question).passages
        return self._answer_grounded_in_context(context=context, question=question)
