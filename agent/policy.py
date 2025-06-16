from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

from memory.memory import Memory


class Policy:
    """Decides the next action for the agent."""

    def __init__(self, model_name: str, tools: List[Tool], memory: Memory) -> None:
        self.llm = ChatOllama(model=model_name, temperature=0.7)
        prompt_template = (
            "You are the policy module. Decide the next action or answer.\n"
            "Memory: {memory}\nInput: {input}\nTools: {tools}\nDecision:"
        )
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.tools = tools
        self.memory = memory
        self.agent = create_react_agent(self.llm, tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=tools, verbose=False)

    def next_step(self, user_input: str, history: List[Dict[str, Any]]) -> str:
        self.memory.add("user_input", user_input)
        chat_history = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        response = self.executor.invoke({"input": user_input, "chat_history": chat_history})
        decision = response.get("output", "")
        self.memory.add("policy_decision", decision)
        return decision
