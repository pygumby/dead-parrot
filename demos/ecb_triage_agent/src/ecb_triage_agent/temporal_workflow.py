"""Triage agent Temporal workflow."""

from typing import Any

import dspy
from dspy_temporal import TemporalModule
from temporalio import workflow

from .triage_agent import triage_agent

temporal_lm_program: TemporalModule[dspy.Prediction] = triage_agent.to_temporal()


@workflow.defn
class EcbTriageAgentWorkflow:
    """Temporal workflow for the triage agent."""

    @workflow.run
    async def run(self, question: str) -> dict[str, Any]:
        """Run the workflow to get an answer for the given question."""
        pred: dspy.Prediction = await temporal_lm_program.run(question=question)
        pred_dict: dict[str, Any] = pred.toDict()
        return pred_dict
