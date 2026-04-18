"""Expert agent Temporal workflow."""

from typing import Any

import dspy
from dspy_temporal import TemporalModule, TemporalTool
from temporalio import workflow

from .expert_agent import expert_agent

temporal_lm_program: TemporalModule[dspy.Prediction]
temporal_retriever: TemporalTool
temporal_lm_program, temporal_retriever = expert_agent.to_temporal()


@workflow.defn
class EcbBsExpertAgentWorkflow:
    """Temporal workflow for the expert agent."""

    @workflow.run
    async def run(self, question: str) -> dict[str, Any]:
        """Run the workflow to get an answer for the given question."""
        pred: dspy.Prediction = await temporal_lm_program.run(question=question)
        pred_dict: dict[str, Any] = pred.toDict()
        return pred_dict
