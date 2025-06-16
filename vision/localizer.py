from typing import Tuple

from memory.memory import Memory


class Localizer:
    """Stub visual localizer for UI elements."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def locate(self, label: str) -> Tuple[int, int]:
        """Return dummy screen coordinates for ``label``."""
        # TODO: implement VLM-based localization
        return (0, 0)
