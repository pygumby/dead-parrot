"""Triage agent."""

from collections.abc import Callable
from typing import Any

import dspy
from dspy_temporal import TemporalModule

from .agent import Agent
from .expert_agent_client import ExpertAgentClient
from .types import Examples, Metric


class TriageAgent(Agent):
    """Triage agent."""

    def __init__(
        self,
        name: str,
        task_model: str,
        expert_agent_clients: list[ExpertAgentClient],
        dataset: Examples | list[Examples],
        metrics: dict[str, Metric],
    ) -> None:
        """Initialize the triage agent."""
        super().__init__(
            name=name,
            dataset=dataset,
            metrics=metrics,
        )

        self._task_model: dspy.LM
        self._init_models(task_model=task_model)

        self._tools: list[Callable[[str], dict[str, Any]]]
        self._init_tools(expert_agent_clients)

        self._lm_program: dspy.ReAct
        self._init_lm_program()

        self._log(msg="Ready...")

    def _init_models(self, task_model: str) -> None:
        self._log(msg="Initializing models")
        self._log(msg=f"Task model: {task_model}", sub=True)
        self._task_model = dspy.LM(model=task_model)

    def _init_tools(
        self,
        expert_agent_clients: list[ExpertAgentClient],
    ) -> None:
        self._log(msg="Initializing tools")
        self._tools = []
        for expert_agent_client in expert_agent_clients:
            self._log(msg=f"Tool: {expert_agent_client.name}", sub=True)
            self._tools.append(expert_agent_client.to_tool())

    def _init_lm_program(self) -> None:
        self._log(msg="Initializing LM program")
        self._lm_program = dspy.ReAct(signature="question -> answer", tools=self._tools)

    def _get_task_model(self) -> dspy.LM:
        return self._task_model

    def _get_lm_program(self) -> dspy.ReAct:
        return self._lm_program

    def to_temporal(self) -> TemporalModule[dspy.Prediction]:
        """Wrap the LM program for use in Temporal workflows."""
        return TemporalModule(
            module=self._lm_program,
            name=f"{self.name}_triage_agent",
            lm=self._task_model,
        )
