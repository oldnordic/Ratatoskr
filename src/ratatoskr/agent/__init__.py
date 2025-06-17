"""
Agent package for Ratatoskr AI Assistant.

This package contains the core agent components including policy decision making,
execution orchestration, and agent management.

Modules:
- execute: High-level agent orchestration and execution
- policy: Decision-making logic and policy management
"""

from .execute import AgentEngine
from .policy import Policy

__all__ = ['AgentEngine', 'Policy']
__version__ = '1.0.0' 