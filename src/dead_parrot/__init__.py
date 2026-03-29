"""API of dead-parrot."""

from dead_parrot import metrics, utils
from dead_parrot.ai_assistant import DspyAiAssistant
from dead_parrot.protocols import (
    AiAssistant,
    AiAssistantClass,
    Dataset,
    Document,
    Metric,
    Models,
)

__all__ = [
    "AiAssistant",
    "AiAssistantClass",
    "DspyAiAssistant",
    "Document",
    "Dataset",
    "Metric",
    "Models",
    "metrics",
    "utils",
]
