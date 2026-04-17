"""Agent."""

import textwrap
from abc import ABC

from . import utils


class Agent(ABC):
    """Abstract base class for an agent."""

    def __init__(self, name: str) -> None:
        """Initialize the agent."""
        self._name: str
        self._init_name(name=name)

    def _init_name(self, name: str) -> None:
        self._name = utils._normalize_name(name)

    def _log(self, msg: str, sub: bool = False, indent: int = 2) -> None:
        indent_str = " " * indent
        if not sub:
            text = f"[{self.name}] {msg}"
            sub_indent_str = indent_str
        else:
            text = f"{indent_str}{msg}"
            sub_indent_str = indent_str * 2
        print(textwrap.fill(text=text, subsequent_indent=sub_indent_str))

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self._name
