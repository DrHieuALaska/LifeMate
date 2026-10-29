from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AgentInput:
    current_transcript: str
    history: tuple[dict, ...]
    context: dict
    allowed_tool_schemas: tuple[dict, ...]
    previous_tool_results: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ToolDecision:
    name: str
    arguments: dict


@dataclass(frozen=True)
class Finish:
    """Explicit termination; contains no evaluator outcome."""


class Agent(Protocol):
    def next_action(self, request: AgentInput) -> ToolDecision | Finish:
        ...
