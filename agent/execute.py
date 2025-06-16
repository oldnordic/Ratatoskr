from typing import Any, Dict, List

from langchain.tools import Tool

from memory.memory import Memory
from agent.policy import Policy
from vision.localizer import Localizer
from validator.checker import Validator


class AgentEngine:
    """High level orchestrator for the agent."""

    def __init__(self, model_name: str, tools: List[Tool]) -> None:
        self.memory = Memory()
        self.policy = Policy(model_name, tools, self.memory)
        self.localizer = Localizer(self.memory)
        self.validator = Validator(self.memory)

    def run(self, user_input: str, history: List[Dict[str, Any]]) -> str:
        decision = self.policy.next_step(user_input, history)
        if decision.startswith("Action: Click"):
            label = decision.split(":", 1)[1].strip()
            x, y = self.localizer.locate(label)
            result = f"Clicked {label} at ({x},{y})"
            self.memory.add("action_result", result)
            if not self.validator.validate(result):
                return self.run(user_input, history)
            return result
        if self.validator.validate(decision):
            return decision
        return self.run(user_input, history)
