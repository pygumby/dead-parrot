"""AI Agent."""

from collections.abc import Callable
from typing import Any, Literal

import dspy
import httpx
from dspy_temporal import TemporalModule

from . import utils
from .agent import Agent
from .protocols import (
    Examples,
    Metric,
)


class AiAssistantClient:
    """AI Assistant Client."""

    def __init__(
        self,
        scheme: Literal["http", "https"],
        host: str,
        port: int,
        ask_endpoint: str = "ask",
        card_endpoint: str = "card",
        timeout: int = 60,
    ):
        """Initialize the AI assistant client."""
        self._base_url: str
        self._init_base_url(stripped_host=host, port=port, scheme=scheme)

        self._card_endpoint: str
        self._ask_endpoint: str
        self._init_endpoints(ask_endpoint=ask_endpoint, card_endpoint=card_endpoint)

        self._timeout: float
        self._init_timeout(timeout=timeout)

        self._name: str
        self._description: str
        self._init_card()

    def _init_base_url(self, stripped_host: str, port: int, scheme: str) -> None:
        stripped_host = stripped_host.strip().rstrip("/")
        self._base_url = f"{scheme}://{stripped_host}:{port}"

    def _init_endpoints(self, ask_endpoint: str, card_endpoint: str) -> None:
        self._ask_endpoint = ask_endpoint.strip().strip("/")
        self._card_endpoint = card_endpoint.strip().strip("/")

    def _init_timeout(self, timeout: int) -> None:
        self._timeout = float(timeout)

    def _init_card(self) -> None:
        with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
            response: httpx.Response = client.get(url=f"/{self._card_endpoint}")
            response.raise_for_status()
            card: dict[str, str] = response.json()
            self._name = utils._normalize_name(card["name"])
            self._description = card["description"]

    @property
    def name(self) -> str:
        """Return the name of the connected AI assistant."""
        return self._name

    @property
    def description(self) -> str:
        """Return the description of the connected AI assistant."""
        return self._description

    def ask(self, question: str) -> dict[str, Any]:
        """Answer the question using the connected AI assistant."""
        with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
            response: httpx.Response = client.post(
                url=f"/{self._ask_endpoint}",
                json={"question": question},
            )
            response.raise_for_status()
            response_dict: dict[str, Any] = response.json()
            return response_dict

    def to_tool(self) -> Callable[[str], dict[str, Any]]:
        """Wrap the AI assistant client into a callable tool."""

        def tool(question: str) -> dict[str, Any]:
            return self.ask(question=question)

        tool.__name__ = self.name
        tool.__doc__ = self.description
        return tool


class AiAgent(Agent):
    """AI Agent."""

    def __init__(
        self,
        name: str,
        task_model: str,
        ai_assistant_clients: list[AiAssistantClient],
        dataset: Examples | list[Examples],
        metrics: dict[str, Metric],
    ) -> None:
        """Initialize the AI agent."""
        super().__init__(
            name=name,
            dataset=dataset,
            metrics=metrics,
        )

        self._task_model: dspy.LM
        self._init_task_model(task_model=task_model)

        self._tools: list[Callable[[str], dict[str, Any]]]
        self._init_tools(ai_assistant_clients)

        self._react: dspy.ReAct
        self._init_react()

        self._log(msg="Ready...")

    def _init_task_model(self, task_model: str) -> None:
        self._log(msg="Initializing models")
        self._log(msg=f"Task model: {task_model}", sub=True)
        self._task_model = dspy.LM(model=task_model)

    def _init_tools(
        self,
        ai_assistant_clients: list[AiAssistantClient],
    ) -> None:
        self._log(msg="Initializing tools")
        self._tools = []
        for ai_assistant_client in ai_assistant_clients:
            self._log(msg=f"Tool: {ai_assistant_client.name}", sub=True)
            self._tools.append(ai_assistant_client.to_tool())

    def _init_react(self) -> None:
        self._log(msg="Initializing ReAct pipeline")
        self._react = dspy.ReAct(signature="question -> answer", tools=self._tools)

    def _get_task_model(self) -> dspy.LM:
        """Return the task model for the agent."""
        return self._task_model

    def _get_module(self) -> dspy.ReAct:
        """Return the dspy.Module of the agent."""
        return self._react

    def to_temporal(self) -> TemporalModule[dspy.Prediction]:
        """Wrap the ReAct pipeline for use in Temporal workflows."""
        return TemporalModule(
            module=self._react,
            name=f"{self.name}_react",
            lm=self._task_model,
        )
