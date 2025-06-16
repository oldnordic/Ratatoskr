"""Simple validation stub used by the agent decision loop."""

from typing import Any

from memory.memory import Memory


class Validator:
    """Placeholder validation component."""

    def __init__(self, memory: Memory) -> None:
        # Potential future versions may consult ``memory`` when validating.
        self.memory = memory

    def validate(self, content: Any) -> bool:
        """Always return ``True`` until real validation is implemented."""
        # TODO: implement LLM-based validation
        return True
