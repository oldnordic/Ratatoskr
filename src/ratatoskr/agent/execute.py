"""
High-level orchestration of agent components for decision execution.

This module provides the main execution engine for the Ratatoskr agent,
coordinating between policy decisions, memory management, vision processing,
and validation. It implements a decision loop that processes user input
and executes appropriate actions.

Key Features:
- Agent component orchestration
- Decision loop management
- Action execution and validation
- Error handling and recovery
- Component integration
"""

import logging
from typing import Any, Dict, List

from langchain.tools import Tool

from memory.memory import Memory
from agent.policy import Policy
from vision.localizer import Localizer
from validator.checker import Validator

# Configuration constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds


class AgentEngine:
    """
    High-level orchestrator for the Ratatoskr agent.
    
    This class coordinates all major subsystems of the agent, including
    policy decision making, memory management, vision processing, and
    validation. It implements a robust decision loop that can handle
    various types of actions and responses.
    
    Attributes:
        memory: Memory system for context storage
        policy: Policy component for decision making
        localizer: Vision component for UI element localization
        validator: Validation component for result verification
        max_retries: Maximum number of retry attempts for failed actions
    """
    
    def __init__(
        self, 
        model_name: str, 
        tools: List[Tool],
        max_retries: int = DEFAULT_MAX_RETRIES
    ) -> None:
        """
        Initialize the agent engine with all required components.
        
        Args:
            model_name: Name of the language model to use
            tools: List of available tools for the agent
            max_retries: Maximum number of retry attempts for failed actions
        """
        logging.info("Initializing AgentEngine...")
        
        # Initialize core components
        self.memory = Memory()
        self.policy = Policy(model_name, tools, self.memory)
        self.localizer = Localizer(self.memory)
        self.validator = Validator(self.memory)
        self.max_retries = max_retries
        
        logging.info("AgentEngine initialized successfully")
    
    def run(self, user_input: str, history: List[Dict[str, Any]]) -> str:
        """
        Execute the main agent decision loop.
        
        This method processes user input through the complete agent pipeline:
        1. Policy decision making
        2. Action execution (if needed)
        3. Result validation
        4. Memory storage
        
        Args:
            user_input: The user's input text
            history: Previous conversation messages
            
        Returns:
            str: The final response or result
        """
        try:
            logging.info(f"Starting agent execution for input: '{user_input[:100]}...'")
            
            # Store initial context
            self.memory.add("execution_start", user_input)
            
            # Get policy decision
            decision = self.policy.next_step(user_input, history)
            
            # Execute decision with retry logic
            result = self._execute_decision(decision, user_input, history)
            
            # Store final result
            self.memory.add("execution_result", result)
            
            logging.info(f"Agent execution completed: '{result[:100]}...'")
            return result
            
        except Exception as e:
            logging.error(f"Error in agent execution: {e}", exc_info=True)
            error_msg = f"Agent execution error: {e}"
            self.memory.add("execution_error", error_msg)
            return error_msg
    
    def _execute_decision(
        self, 
        decision: str, 
        user_input: str, 
        history: List[Dict[str, Any]]
    ) -> str:
        """
        Execute a policy decision with retry logic.
        
        Args:
            decision: The policy decision to execute
            user_input: Original user input for context
            history: Conversation history for context
            
        Returns:
            str: Execution result
        """
        for attempt in range(self.max_retries):
            try:
                logging.debug(f"Executing decision (attempt {attempt + 1}): '{decision[:100]}...'")
                
                # Handle different types of decisions
                if decision.startswith("Action: Click"):
                    return self._execute_click_action(decision)
                elif decision.startswith("Action:"):
                    return self._execute_tool_action(decision)
                else:
                    # Direct answer or response
                    return self._validate_and_return(decision)
                    
            except Exception as e:
                logging.warning(f"Execution attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Try again with a fresh decision
                    decision = self.policy.next_step(user_input, history)
                else:
                    # Final attempt failed
                    raise
        
        return "Maximum retry attempts exceeded"
    
    def _execute_click_action(self, decision: str) -> str:
        """
        Execute a click action using the vision localizer.
        
        Args:
            decision: Decision string containing click action
            
        Returns:
            str: Result of the click action
        """
        try:
            # Extract label from decision
            if ":" in decision:
                label = decision.split(":", 1)[1].strip()
            else:
                label = "unknown"
            
            logging.info(f"Executing click action for label: {label}")
            
            # Locate element using vision system
            x, y = self.localizer.locate(label)
            
            # Execute click (simulated for now)
            result = f"Clicked {label} at coordinates ({x}, {y})"
            
            # Store action result
            self.memory.add("action_result", result)
            
            # Validate result
            if not self.validator.validate(result):
                logging.warning(f"Click action validation failed: {result}")
                return f"Click action failed validation: {result}"
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing click action: {e}")
            return f"Click action error: {e}"
    
    def _execute_tool_action(self, decision: str) -> str:
        """
        Execute a tool action.
        
        Args:
            decision: Decision string containing tool action
            
        Returns:
            str: Result of the tool action
        """
        try:
            logging.info(f"Executing tool action: {decision}")
            
            # Parse tool action
            tool_name, tool_input = self._parse_tool_action(decision)
            
            if not tool_name:
                return f"Invalid tool action format: {decision}"
            
            # Find and execute the tool
            tool_result = self._execute_tool(tool_name, tool_input)
            
            # Store action result
            self.memory.add("tool_result", f"{tool_name}: {tool_result}")
            
            return tool_result
            
        except Exception as e:
            logging.error(f"Error executing tool action: {e}")
            return f"Tool action error: {e}"
    
    def _parse_tool_action(self, decision: str) -> tuple[str, str]:
        """
        Parse tool action from decision string.
        
        Args:
            decision: Decision string containing tool action
            
        Returns:
            tuple[str, str]: (tool_name, tool_input)
        """
        try:
            # Expected format: "Action: ToolName\nAction Input: tool_input"
            lines = decision.strip().split('\n')
            
            tool_name = ""
            tool_input = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Action:"):
                    tool_name = line[7:].strip()
                elif line.startswith("Action Input:"):
                    tool_input = line[13:].strip()
            
            # If no Action Input found, try to extract from the same line as Action
            if not tool_input and ":" in tool_name:
                parts = tool_name.split(":", 1)
                if len(parts) == 2:
                    tool_name = parts[0].strip()
                    tool_input = parts[1].strip()
            
            return tool_name, tool_input
            
        except Exception as e:
            logging.error(f"Error parsing tool action: {e}")
            return "", ""
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a specific tool.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input for the tool
            
        Returns:
            str: Result of tool execution
        """
        try:
            # Map tool names to actual tool functions
            tool_mapping = {
                "Web Search": self._execute_web_search,
                "Browse Web": self._execute_browse_web,
                "Long-Term Memory Search": self._execute_memory_search,
                "Save to Memory": self._execute_save_memory
            }
            
            if tool_name in tool_mapping:
                return tool_mapping[tool_name](tool_input)
            else:
                return f"Unknown tool: {tool_name}"
                
        except Exception as e:
            logging.error(f"Error executing tool {tool_name}: {e}")
            return f"Tool execution error: {e}"
    
    def _execute_web_search(self, query: str) -> str:
        """Execute web search tool."""
        try:
            from tools.web_search import perform_web_search
            return perform_web_search(query)
        except Exception as e:
            return f"Web search error: {e}"
    
    def _execute_browse_web(self, query: str) -> str:
        """Execute browse web tool."""
        try:
            from tools.browser_tool import browse_search
            return browse_search(query, None)
        except Exception as e:
            return f"Browse web error: {e}"
    
    def _execute_memory_search(self, query: str) -> str:
        """Execute memory search tool."""
        try:
            # Use conversation manager for memory search since Memory class doesn't have search
            from memory.conversation_manager import ConversationManager
            conversation_manager = ConversationManager()
            return conversation_manager.get_relevant_memories(query)
        except Exception as e:
            return f"Memory search error: {e}"
    
    def _execute_save_memory(self, content: str) -> str:
        """Execute save to memory tool."""
        try:
            self.memory.add("important_info", content)
            return f"Saved to memory: {content[:50]}..."
        except Exception as e:
            return f"Save memory error: {e}"
    
    def _validate_and_return(self, response: str) -> str:
        """
        Validate a response and return it.
        
        Args:
            response: The response to validate
            
        Returns:
            str: Validated response
        """
        try:
            if self.validator.validate(response):
                return response
            else:
                logging.warning(f"Response validation failed: {response}")
                return f"Response failed validation: {response}"
                
        except Exception as e:
            logging.error(f"Error validating response: {e}")
            return f"Validation error: {e}"
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the agent engine status.
        
        Returns:
            dict: Engine status and configuration information
        """
        try:
            return {
                "max_retries": self.max_retries,
                "memory_entries": self.memory.count(),
                "policy_info": self.policy.get_policy_info(),
                "localizer_available": hasattr(self.localizer, 'locate'),
                "validator_available": hasattr(self.validator, 'validate')
            }
        except Exception as e:
            logging.error(f"Error getting engine info: {e}")
            return {"error": str(e)}
    
    def reset(self) -> None:
        """Reset the agent engine state."""
        try:
            logging.info("Resetting agent engine...")
            
            # Clear memory
            self.memory.clear()
            
            # Reset policy memory
            self.policy.clear_memory()
            
            logging.info("Agent engine reset completed")
            
        except Exception as e:
            logging.error(f"Error resetting agent engine: {e}")
    
    def update_tools(self, new_tools: List[Tool]) -> None:
        """
        Update the tools available to the agent.
        
        Args:
            new_tools: New list of tools to use
        """
        try:
            logging.info(f"Updating agent tools: {len(new_tools)} tools")
            self.policy.update_tools(new_tools)
            logging.info("Agent tools updated successfully")
            
        except Exception as e:
            logging.error(f"Error updating agent tools: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the agent's memory.
        
        Returns:
            dict: Memory summary information
        """
        try:
            return self.memory.get_summary()
        except Exception as e:
            logging.error(f"Error getting memory summary: {e}")
            return {"error": str(e)}
