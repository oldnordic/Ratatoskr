"""
Comprehensive tests for the agent system components.

This module contains unit tests for the agent functionality including
policy decision making, execution orchestration, and agent management.

Test Coverage:
- Policy decision making
- Agent execution flow
- Tool integration
- Error handling and recovery
- Performance benchmarks
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os

# Import the modules to test
from agent.policy import Policy
from agent.execute import AgentEngine


class TestPolicy(unittest.TestCase):
    """Test suite for policy decision making."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.policy = Policy()
        self.mock_llm = MagicMock()
        self.mock_memory = MagicMock()
        self.mock_tools = [MagicMock(), MagicMock()]
    
    def test_analyze_input_basic(self):
        """Test basic input analysis."""
        user_input = "What is the weather today?"
        
        with patch.object(self.policy, '_get_llm_response') as mock_llm_response:
            mock_llm_response.return_value = {
                'action': 'web_search',
                'reasoning': 'Need to search for current weather information',
                'confidence': 0.8
            }
            
            result = self.policy.analyze_input(user_input, self.mock_llm, self.mock_memory)
            
            self.assertIsInstance(result, dict)
            self.assertIn('action', result)
            self.assertIn('reasoning', result)
            self.assertIn('confidence', result)
            self.assertEqual(result['action'], 'web_search')
    
    def test_analyze_input_with_context(self):
        """Test input analysis with memory context."""
        user_input = "What did we discuss earlier?"
        
        # Mock memory retrieval
        self.mock_memory.retrieve_relevant.return_value = [
            {'content': 'We discussed AI assistants', 'timestamp': '2024-01-01'}
        ]
        
        with patch.object(self.policy, '_get_llm_response') as mock_llm_response:
            mock_llm_response.return_value = {
                'action': 'memory_retrieval',
                'reasoning': 'User is asking about previous conversation',
                'confidence': 0.9
            }
            
            result = self.policy.analyze_input(user_input, self.mock_llm, self.mock_memory)
            
            # Verify memory was consulted
            self.mock_memory.retrieve_relevant.assert_called_once()
            self.assertEqual(result['action'], 'memory_retrieval')
    
    def test_analyze_input_empty(self):
        """Test input analysis with empty input."""
        result = self.policy.analyze_input("", self.mock_llm, self.mock_memory)
        self.assertEqual(result['action'], 'clarification')
        self.assertLess(result['confidence'], 0.5)
    
    def test_analyze_input_none(self):
        """Test input analysis with None input."""
        result = self.policy.analyze_input(None, self.mock_llm, self.mock_memory)
        self.assertEqual(result['action'], 'clarification')
        self.assertLess(result['confidence'], 0.5)
    
    def test_select_tool_basic(self):
        """Test basic tool selection."""
        action = 'web_search'
        available_tools = [
            {'name': 'web_search', 'description': 'Search the web'},
            {'name': 'browser_tool', 'description': 'Browse websites'}
        ]
        
        result = self.policy.select_tool(action, available_tools)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['name'], 'web_search')
    
    def test_select_tool_no_match(self):
        """Test tool selection when no matching tool is found."""
        action = 'nonexistent_action'
        available_tools = [
            {'name': 'web_search', 'description': 'Search the web'}
        ]
        
        result = self.policy.select_tool(action, available_tools)
        
        self.assertIsNone(result)
    
    def test_select_tool_empty_list(self):
        """Test tool selection with empty tool list."""
        action = 'web_search'
        available_tools = []
        
        result = self.policy.select_tool(action, available_tools)
        
        self.assertIsNone(result)
    
    def test_get_llm_response_success(self):
        """Test successful LLM response generation."""
        prompt = "Analyze this input: Hello world"
        
        self.mock_llm.generate.return_value = """
        {
            "action": "greeting",
            "reasoning": "User said hello",
            "confidence": 0.9
        }
        """
        
        result = self.policy._get_llm_response(prompt, self.mock_llm)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['action'], 'greeting')
        self.assertEqual(result['confidence'], 0.9)
    
    def test_get_llm_response_invalid_json(self):
        """Test LLM response with invalid JSON."""
        prompt = "Analyze this input: Hello world"
        
        self.mock_llm.generate.return_value = "Invalid JSON response"
        
        result = self.policy._get_llm_response(prompt, self.mock_llm)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['action'], 'clarification')
        self.assertLess(result['confidence'], 0.5)
    
    def test_get_llm_response_none(self):
        """Test LLM response when LLM returns None."""
        prompt = "Analyze this input: Hello world"
        
        self.mock_llm.generate.return_value = None
        
        result = self.policy._get_llm_response(prompt, self.mock_llm)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['action'], 'clarification')
        self.assertLess(result['confidence'], 0.5)


class TestAgentEngine(unittest.TestCase):
    """Test suite for agent execution engine."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.engine = AgentEngine()
        self.mock_policy = MagicMock()
        self.mock_memory = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_tools = [MagicMock(), MagicMock()]
    
    def test_execute_basic_flow(self):
        """Test basic execution flow."""
        user_input = "Search for Python tutorials"
        
        # Mock policy analysis
        self.mock_policy.analyze_input.return_value = {
            'action': 'web_search',
            'reasoning': 'User wants to search for tutorials',
            'confidence': 0.8
        }
        
        # Mock tool selection
        self.mock_policy.select_tool.return_value = {
            'name': 'web_search',
            'description': 'Search the web',
            'function': MagicMock(return_value='Search results')
        }
        
        # Mock tool execution
        mock_tool = MagicMock()
        mock_tool.run.return_value = 'Search results for Python tutorials'
        
        with patch.object(self.engine, '_get_tool_by_name') as mock_get_tool:
            mock_get_tool.return_value = mock_tool
            
            result = self.engine.execute(
                user_input,
                self.mock_policy,
                self.mock_memory,
                self.mock_llm,
                self.mock_tools
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('response', result)
            self.assertTrue(result['success'])
            self.assertIn('Search results', result['response'])
    
    def test_execute_with_memory_storage(self):
        """Test execution with memory storage."""
        user_input = "Remember that I like Python"
        
        # Mock policy analysis
        self.mock_policy.analyze_input.return_value = {
            'action': 'memory_store',
            'reasoning': 'User wants to store information',
            'confidence': 0.9
        }
        
        # Mock tool selection
        self.mock_policy.select_tool.return_value = {
            'name': 'memory_store',
            'description': 'Store information in memory',
            'function': MagicMock(return_value=True)
        }
        
        # Mock tool execution
        mock_tool = MagicMock()
        mock_tool.run.return_value = 'Information stored successfully'
        
        with patch.object(self.engine, '_get_tool_by_name') as mock_get_tool:
            mock_get_tool.return_value = mock_tool
            
            result = self.engine.execute(
                user_input,
                self.mock_policy,
                self.mock_memory,
                self.mock_llm,
                self.mock_tools
            )
            
            # Verify memory was updated
            self.mock_memory.add.assert_called_once()
            self.assertTrue(result['success'])
    
    def test_execute_tool_not_found(self):
        """Test execution when tool is not found."""
        user_input = "Perform unknown action"
        
        # Mock policy analysis
        self.mock_policy.analyze_input.return_value = {
            'action': 'unknown_action',
            'reasoning': 'Unknown action requested',
            'confidence': 0.3
        }
        
        # Mock tool selection returning None
        self.mock_policy.select_tool.return_value = None
        
        result = self.engine.execute(
            user_input,
            self.mock_policy,
            self.mock_memory,
            self.mock_llm,
            self.mock_tools
        )
        
        self.assertFalse(result['success'])
        self.assertIn('Tool not found', result['response'])
    
    def test_execute_tool_execution_error(self):
        """Test execution when tool execution fails."""
        user_input = "Search for something"
        
        # Mock policy analysis
        self.mock_policy.analyze_input.return_value = {
            'action': 'web_search',
            'reasoning': 'User wants to search',
            'confidence': 0.8
        }
        
        # Mock tool selection
        self.mock_policy.select_tool.return_value = {
            'name': 'web_search',
            'description': 'Search the web',
            'function': MagicMock()
        }
        
        # Mock tool execution failure
        mock_tool = MagicMock()
        mock_tool.run.side_effect = Exception("Tool execution failed")
        
        with patch.object(self.engine, '_get_tool_by_name') as mock_get_tool:
            mock_get_tool.return_value = mock_tool
            
            result = self.engine.execute(
                user_input,
                self.mock_policy,
                self.mock_memory,
                self.mock_llm,
                self.mock_tools
            )
            
            self.assertFalse(result['success'])
            self.assertIn('error', result['response'])
    
    def test_execute_with_retry(self):
        """Test execution with retry mechanism."""
        user_input = "Search for information"
        
        # Mock policy analysis
        self.mock_policy.analyze_input.return_value = {
            'action': 'web_search',
            'reasoning': 'User wants to search',
            'confidence': 0.8
        }
        
        # Mock tool selection
        self.mock_policy.select_tool.return_value = {
            'name': 'web_search',
            'description': 'Search the web',
            'function': MagicMock()
        }
        
        # Mock tool execution that fails first, then succeeds
        mock_tool = MagicMock()
        mock_tool.run.side_effect = [Exception("First failure"), "Success on retry"]
        
        with patch.object(self.engine, '_get_tool_by_name') as mock_get_tool:
            mock_get_tool.return_value = mock_tool
            
            result = self.engine.execute(
                user_input,
                self.mock_policy,
                self.mock_memory,
                self.mock_llm,
                self.mock_tools
            )
            
            # Should succeed on retry
            self.assertTrue(result['success'])
            self.assertEqual(result['response'], 'Success on retry')
            # Should have been called twice (initial + retry)
            self.assertEqual(mock_tool.run.call_count, 2)
    
    def test_get_tool_by_name_success(self):
        """Test successful tool retrieval by name."""
        tool_name = 'web_search'
        
        # Create mock tools with names
        mock_tool1 = MagicMock()
        mock_tool1.name = 'web_search'
        mock_tool2 = MagicMock()
        mock_tool2.name = 'browser_tool'
        
        tools = [mock_tool1, mock_tool2]
        
        result = self.engine._get_tool_by_name(tool_name, tools)
        
        self.assertEqual(result, mock_tool1)
    
    def test_get_tool_by_name_not_found(self):
        """Test tool retrieval when tool is not found."""
        tool_name = 'nonexistent_tool'
        
        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = 'web_search'
        mock_tool2 = MagicMock()
        mock_tool2.name = 'browser_tool'
        
        tools = [mock_tool1, mock_tool2]
        
        result = self.engine._get_tool_by_name(tool_name, tools)
        
        self.assertIsNone(result)
    
    def test_get_tool_by_name_empty_list(self):
        """Test tool retrieval with empty tool list."""
        tool_name = 'web_search'
        tools = []
        
        result = self.engine._get_tool_by_name(tool_name, tools)
        
        self.assertIsNone(result)


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agent components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.policy = Policy()
        self.engine = AgentEngine()
        self.mock_memory = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_tools = [MagicMock(), MagicMock()]
    
    def test_full_agent_workflow(self):
        """Test complete agent workflow from input to response."""
        user_input = "What is the current time?"
        
        # Mock LLM responses for policy analysis
        self.mock_llm.generate.return_value = """
        {
            "action": "system_time",
            "reasoning": "User is asking for current time",
            "confidence": 0.9
        }
        """
        
        # Mock tool
        mock_tool = MagicMock()
        mock_tool.name = 'system_time'
        mock_tool.run.return_value = 'Current time is 12:00 PM'
        
        with patch.object(self.engine, '_get_tool_by_name') as mock_get_tool:
            mock_get_tool.return_value = mock_tool
            
            # Execute full workflow
            result = self.engine.execute(
                user_input,
                self.policy,
                self.mock_memory,
                self.mock_llm,
                [mock_tool]
            )
            
            # Verify complete workflow
            self.assertTrue(result['success'])
            self.assertIn('Current time', result['response'])
            
            # Verify memory was consulted and updated
            self.mock_memory.retrieve_relevant.assert_called()
            self.mock_memory.add.assert_called()


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 