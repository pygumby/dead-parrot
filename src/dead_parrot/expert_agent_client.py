"""Expert agent client."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import httpx

from . import utils

if TYPE_CHECKING:
    from .types import ExpertAgentClientClass


class ExpertAgentClient:
    """Expert agent client."""

    def __init__(
        self,
        scheme: Literal["http", "https"],
        host: str,
        port: int,
        ask_endpoint: str = "ask",
        card_endpoint: str = "card",
        timeout: int = 60,
    ):
        """Initialize the expert agent client."""
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
        """Return the name of the connected expert agent."""
        return self._name

    @property
    def description(self) -> str:
        """Return the description of the connected expert agent."""
        return self._description

    def ask(self, question: str) -> dict[str, Any]:
        """Query the connected expert agent with a question."""
        with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
            response: httpx.Response = client.post(
                url=f"/{self._ask_endpoint}",
                json={"question": question},
            )
            response.raise_for_status()
            response_dict: dict[str, Any] = response.json()
            return response_dict

    def to_tool(self) -> Callable[[str], dict[str, Any]]:
        """Wrap the expert agent client into a callable tool."""

        def tool(question: str) -> dict[str, Any]:
            return self.ask(question=question)

        tool.__name__ = self.name
        tool.__doc__ = self.description
        return tool


if TYPE_CHECKING:
    _: ExpertAgentClientClass = ExpertAgentClient
