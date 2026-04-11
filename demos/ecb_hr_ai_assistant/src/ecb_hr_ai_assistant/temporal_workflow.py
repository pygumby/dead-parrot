"""AI Assistant Temporal Workflow."""

from typing import Any

import dspy
from dspy_temporal import TemporalModule, TemporalTool
from temporalio import workflow

from .ai_assistant import ai_assistant

rag: TemporalModule[dspy.Prediction]
retriever: TemporalTool
rag, retriever = ai_assistant.to_temporal()


@workflow.defn
class EcbHrAiAssistantWorkflow:
    """Temporal workflow for the AI Assistant."""

    @workflow.run
    async def run(self, question: str) -> dict[str, Any]:
        """Run the workflow to get an answer for the given question."""
        pred: dspy.Prediction = await rag.run(question=question)
        pred_dict: dict[str, Any] = pred.toDict()
        return pred_dict
