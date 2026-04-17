"""AI Agent Temporal Workflow."""

from typing import Any

import dspy
from dspy_temporal import TemporalModule
from temporalio import workflow

from .ai_agent import ai_agent

react: TemporalModule[dspy.Prediction] = ai_agent.to_temporal()


@workflow.defn
class EcbAiAgentWorkflow:
    """Temporal workflow for the AI Agent."""

    @workflow.run
    async def run(self, question: str) -> dict[str, Any]:
        """Run the workflow to get an answer for the given question."""
        pred: dspy.Prediction = await react.run(question=question)
        pred_dict: dict[str, Any] = pred.toDict()
        return pred_dict
