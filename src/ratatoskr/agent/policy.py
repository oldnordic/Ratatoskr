"""
Policy logic for agent decision making and action selection.

This module implements the core decision-making logic for the Ratatoskr agent.
It uses LangChain's ReAct agent framework to determine the next action based
on user input, available tools, and conversation history.

Key Features:
- LangChain ReAct agent integration
- Tool-based decision making
- Conversation history management
- Configurable model parameters
- Error handling and logging
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

from memory.memory import Memory

# Configuration constants
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_EXECUTION_TIME = 300  # seconds
DEFAULT_VERBOSE = False


class Policy:
    """
    Decision-making component for the Ratatoskr agent.
    
    This class manages the agent's decision-making process using LangChain's
    ReAct framework. It combines language model reasoning with tool execution
    to determine appropriate responses to user queries.
    
    Attributes:
        llm: The language model used for decision making
        tools: Available tools for the agent to use
        memory: Memory system for storing context
        agent: The ReAct agent instance
        executor: The agent executor for running the agent
    """
    
    def __init__(
        self, 
        model_name: str, 
        tools: List[Tool], 
        memory: Memory,
        temperature: float = DEFAULT_TEMPERATURE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_execution_time: int = DEFAULT_MAX_EXECUTION_TIME,
        verbose: bool = DEFAULT_VERBOSE
    ) -> None:
        """
        Initialize the policy component.
        
        Args:
            model_name: Name of the Ollama model to use
            tools: List of available tools for the agent
            memory: Memory system for context storage
            temperature: Model temperature for response generation
            max_iterations: Maximum iterations for agent reasoning
            max_execution_time: Maximum execution time in seconds
            verbose: Whether to enable verbose logging
        """
        self.memory = memory
        self.tools = tools
        
        # Initialize language model
        logging.info(f"Initializing language model: {model_name}")
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        
        # Create prompt template for agent reasoning
        self.prompt = self._create_prompt_template()
        
        # Create ReAct agent
        logging.info("Creating ReAct agent...")
        self.agent = create_react_agent(self.llm, tools, self.prompt)
        
        # Create agent executor with safety limits
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time
        )
        
        logging.info("Policy component initialized successfully")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create the prompt template for agent reasoning.
        
        Returns:
            PromptTemplate: Configured prompt template
        """
        prompt_template = """
You are the policy module for Ratatoskr, an AI assistant. Your role is to decide
the next action or provide a direct answer based on the user's input.

Available Context:
- Memory: {memory}
- User Input: {input}
- Available Tools: {tools}

Instructions:
1. Analyze the user's request carefully
2. Consider relevant information from memory
3. Decide whether to use a tool or provide a direct answer
4. If using a tool, specify which tool and the input
5. If providing a direct answer, be helpful and informative

Decision Format:
- For tool use: "Action: [tool_name] | Input: [tool_input]"
- For direct answer: "Answer: [your response]"

Begin your analysis:
"""
        return PromptTemplate.from_template(prompt_template)
    
    def next_step(self, user_input: str, history: List[Dict[str, Any]]) -> str:
        """
        Determine the next action or answer based on user input and history.
        
        Args:
            user_input: The user's current input
            history: Previous conversation messages
            
        Returns:
            str: The agent's decision (action or answer)
        """
        try:
            # Store user input in memory
            self.memory.add("user_input", user_input)
            
            # Format conversation history
            chat_history = self._format_history(history)
            
            # Prepare input for agent
            agent_input = {
                "input": user_input,
                "chat_history": chat_history,
                "memory": self._get_memory_context(),
                "tools": self._format_tools()
            }
            
            logging.info(f"Processing user input: '{user_input[:100]}...'")
            
            # Execute agent reasoning
            response = self.executor.invoke(agent_input)
            decision = response.get("output", "")
            
            # Store decision in memory
            self.memory.add("policy_decision", decision)
            
            logging.info(f"Policy decision: '{decision[:100]}...'")
            return decision
            
        except Exception as e:
            logging.error(f"Error in policy decision making: {e}", exc_info=True)
            error_msg = f"Policy error: {e}"
            self.memory.add("policy_error", error_msg)
            return error_msg
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Format conversation history for the agent.
        
        Args:
            history: List of conversation messages
            
        Returns:
            str: Formatted conversation history
        """
        if not history:
            return "No previous conversation."
        
        formatted_messages = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")
        
        return "\n".join(formatted_messages)
    
    def _get_memory_context(self) -> str:
        """
        Get relevant context from memory.
        
        Returns:
            str: Formatted memory context
        """
        try:
            # Get recent memory entries
            recent_entries = self.memory.retrieve()
            
            if not recent_entries:
                return "No recent memory entries."
            
            # Format recent entries (limit to last 5)
            recent_entries = recent_entries[-5:]
            formatted_entries = []
            
            for entry in recent_entries:
                entry_type = entry.get("type", "unknown")
                content = entry.get("content", "")
                timestamp = entry.get("timestamp", "")
                formatted_entries.append(f"[{entry_type}] {content}")
            
            return "\n".join(formatted_entries)
            
        except Exception as e:
            logging.error(f"Error getting memory context: {e}")
            return "Memory context unavailable."
    
    def _format_tools(self) -> str:
        """
        Format available tools for the agent.
        
        Returns:
            str: Formatted tool descriptions
        """
        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools:
            name = getattr(tool, 'name', 'Unknown')
            description = getattr(tool, 'description', 'No description')
            tool_descriptions.append(f"- {name}: {description}")
        
        return "\n".join(tool_descriptions)
    
    def get_policy_info(self) -> Dict[str, Any]:
        """
        Get information about the current policy configuration.
        
        Returns:
            dict: Policy configuration and status information
        """
        try:
            return {
                "model_name": self.llm.model if hasattr(self.llm, 'model') else "Unknown",
                "temperature": self.llm.temperature if hasattr(self.llm, 'temperature') else DEFAULT_TEMPERATURE,
                "max_iterations": self.executor.max_iterations,
                "max_execution_time": self.executor.max_execution_time,
                "verbose": self.executor.verbose,
                "tool_count": len(self.tools),
                "memory_entries": self.memory.count()
            }
        except Exception as e:
            logging.error(f"Error getting policy info: {e}")
            return {"error": str(e)}
    
    def update_tools(self, new_tools: List[Tool]) -> None:
        """
        Update the available tools for the agent.
        
        Args:
            new_tools: New list of tools to use
        """
        try:
            logging.info(f"Updating tools: {len(new_tools)} new tools")
            
            self.tools = new_tools
            
            # Recreate agent with new tools
            self.agent = create_react_agent(self.llm, new_tools, self.prompt)
            self.executor = AgentExecutor(
                agent=self.agent,
                tools=new_tools,
                verbose=self.executor.verbose,
                handle_parsing_errors=True,
                max_iterations=self.executor.max_iterations,
                max_execution_time=self.executor.max_execution_time
            )
            
            logging.info("Tools updated successfully")
            
        except Exception as e:
            logging.error(f"Error updating tools: {e}")
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        try:
            self.memory.clear()
            logging.info("Agent memory cleared")
        except Exception as e:
            logging.error(f"Error clearing memory: {e}")
