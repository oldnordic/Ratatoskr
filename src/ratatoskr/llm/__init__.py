"""
LLM package for Ratatoskr AI Assistant.

This package contains language model integration and management
for Ollama and other LLM services.

Modules:
- ollama_client: Ollama API client and model management
"""

from .ollama_client import OllamaClient, get_llm_client, test_ollama_connection

__all__ = [
    'OllamaClient',
    'get_llm_client',
    'test_ollama_connection'
]
__version__ = '1.0.0' 