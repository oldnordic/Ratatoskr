# memory/memory.py
from typing import List, Dict, Any

class Memory:
    """
    Central memory buffer for agent modules.
    Stores entries as dicts with 'type' and 'content'.
    """
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(self, entry_type: str, content: Any) -> None:
        """Add an entry of a given type to memory."""
        self.entries.append({"type": entry_type, "content": content})

    def retrieve(self, entry_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve entries; if type specified, filter by entry_type."""
        if entry_type is None:
            return self.entries
        return [e for e in self.entries if e["type"] == entry_type]

# agent/policy.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from memory.memory import Memory
from typing import Any, Dict, List

class Policy:
    def __init__(self, model_name: str, tools: List[Tool], memory: Memory):
        self.llm = ChatOllama(model=model_name, temperature=0.7)
        prompt_template = """
You are the policy module. Decide the next action or answer.
Memory: {memory}
Input: {input}
Tools: {tools}
Decision:"""
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.tools = tools
        self.memory = memory
        self.agent = create_react_agent(self.llm, tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=tools, verbose=False)

    def next_step(self, user_input: str, history: List[Dict[str,Any]]) -> str:
        # update memory
        self.memory.add('user_input', user_input)
        chat_history = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        response = self.executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        decision = response.get('output', '')
        self.memory.add('policy_decision', decision)
        return decision

# vision/localizer.py
from typing import Tuple
from memory.memory import Memory

class Localizer:
    def __init__(self, memory: Memory):
        self.memory = memory

    def locate(self, label: str) -> Tuple[int, int]:
        """
        Stub: locate UI element by label. Return (x,y) coordinates.
        Replace with VLM call (e.g. Holo1) using screenshots from memory.
        """
        # TODO: implement VLM-based localization
        return (0, 0)

# validator/checker.py
from memory.memory import Memory

class Validator:
    def __init__(self, memory: Memory):
        self.memory = memory

    def validate(self, content: Any) -> bool:
        """
        Stub: validate an action or answer. Return True if OK, False otherwise.
        Replace with lightweight LLM validation prompt.
        """
        # TODO: implement LLM-based validation
        return True

# agent/execute.py
from memory.memory import Memory
from agent.policy import Policy
from vision.localizer import Localizer
from validator.checker import Validator
from typing import Dict, Any

class AgentEngine:
    def __init__(self, model_name: str, tools: list[Tool]):
        self.memory = Memory()
        self.policy = Policy(model_name, tools, self.memory)
        self.localizer = Localizer(self.memory)
        self.validator = Validator(self.memory)

    def run(self, user_input: str, history: list[Dict[str,Any]]) -> str:
        # Step 1: policy decides
        decision = self.policy.next_step(user_input, history)
        # Step 2: if action (e.g. click label)
        if decision.startswith('Action: Click'):
            label = decision.split(':',1)[1].strip()
            x,y = self.localizer.locate(label)
            # perform click event here (e.g. via JS or PyQt)
            result = f'Clicked {label} at ({x},{y})'
            self.memory.add('action_result', result)
            if not self.validator.validate(result):
                return self.run(user_input, history)
            return result
        # Step 3: else final answer
        if self.validator.validate(decision):
            return decision
        # If validation failed, re-decide
        return self.run(user_input, history)
