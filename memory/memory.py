from typing import Any, Dict, List


class Memory:
    """Simple in-memory store for agent components."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add(self, entry_type: str, content: Any) -> None:
        """Add an entry with the given type and content."""
        self.entries.append({"type": entry_type, "content": content})

    def retrieve(self, entry_type: str | None = None) -> List[Dict[str, Any]]:
        """Retrieve entries, optionally filtered by ``entry_type``."""
        if entry_type is None:
            return self.entries
        return [e for e in self.entries if e["type"] == entry_type]
