"""Helpers for interacting with a locally served LLM via Ollama."""

import logging

# This module only exposes a minimal placeholder as the main program currently
# handles all communication with the LLM in ``main.py``.


def get_llm_client() -> None:
    """Return the LLM client instance if one is configured."""
    logging.info("get_llm_client called (placeholder).")
    return None
