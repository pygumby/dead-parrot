"""API of dead-parrot."""

from dead_parrot import utils
from dead_parrot.ai_assistant import DspyAiAssistant
from dead_parrot.protocols import AiAssistant, AiAssistantType

__all__ = [
    "AiAssistant",
    "AiAssistantType",
    "DspyAiAssistant",
    "utils",
]
