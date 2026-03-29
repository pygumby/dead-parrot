"""API of dead-parrot."""

from dead_parrot import metrics, utils
from dead_parrot.ai_assistant import DspyAiAssistant
from dead_parrot.protocols import (
    AiAssistant,
    AiAssistantClass,
    Document,
    Examples,
    Metric,
    Models,
)

__all__ = [
    "AiAssistant",
    "AiAssistantClass",
    "DspyAiAssistant",
    "Document",
    "Examples",
    "Metric",
    "Models",
    "metrics",
    "utils",
]
