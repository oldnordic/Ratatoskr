import logging

# This module currently provides a placeholder API for obtaining an LLM client.
# The main logic lives in ``main.py`` where the agent is executed in a separate
# worker thread.


def get_llm_client() -> None:
    """Return the LLM client instance if one is configured."""
    logging.info("get_llm_client called (placeholder).")
    return None
