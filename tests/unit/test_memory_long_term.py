"""
Comprehensive tests for the long-term memory system.

This module contains unit tests for the long-term memory functionality
including vector storage, retrieval, and management operations.

Test Coverage:
- Memory addition and storage
- Memory retrieval and similarity search
- Memory statistics and management
- Error handling and edge cases
- Performance benchmarks
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the functions to test
from memory.long_term import (
    add_memory,
    retrieve_relevant_memories,
    clear_memory,
    get_memory_stats,
    search_memories
)


class TestLongTermMemory(unittest.TestCase):
    """Test suite for long-term memory functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.memory_dir = os.path.join(self.test_dir, "test_memory")
        
        # Mock configuration
        self.config_patcher = patch('memory.long_term.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.memory.memory_dir = self.memory_dir
        self.mock_config.memory.max_retrieval = 5
        self.mock_config.memory.max_memory_size = 1000
    
    def tearDown(self):
        """Clean up test environment after each test."""
        # Stop the config patch
        self.config_patcher.stop()
        
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_add_memory_success(self):
        """Test successful memory addition."""
        # Test data
        content = "This is a test memory entry"
        metadata = {"source": "test", "timestamp": "2024-01-01"}
        
        # Mock ChromaDB operations
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Execute function
            result = add_memory(content, metadata)
            
            # Verify results
            self.assertTrue(result)
            mock_collection.add.assert_called_once()
            
            # Verify the call arguments
            call_args = mock_collection.add.call_args
            self.assertIn('documents', call_args[1])
            self.assertIn('metadatas', call_args[1])
            self.assertEqual(call_args[1]['documents'][0], content)
            self.assertEqual(call_args[1]['metadatas'][0], metadata)
    
    def test_add_memory_with_empty_content(self):
        """Test memory addition with empty content."""
        result = add_memory("", {"source": "test"})
        self.assertFalse(result)
    
    def test_add_memory_with_none_content(self):
        """Test memory addition with None content."""
        result = add_memory(None, {"source": "test"})
        self.assertFalse(result)
    
    def test_retrieve_relevant_memories_success(self):
        """Test successful memory retrieval."""
        query = "test query"
        
        # Mock ChromaDB operations
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock query results
            mock_results = {
                'documents': [['Test memory 1', 'Test memory 2']],
                'metadatas': [[{'source': 'test1'}, {'source': 'test2'}]],
                'distances': [[0.1, 0.3]]
            }
            mock_collection.query.return_value = mock_results
            
            # Execute function
            result = retrieve_relevant_memories(query)
            
            # Verify results
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['content'], 'Test memory 1')
            self.assertEqual(result[0]['metadata']['source'], 'test1')
            self.assertEqual(result[1]['content'], 'Test memory 2')
            self.assertEqual(result[1]['metadata']['source'], 'test2')
    
    def test_retrieve_relevant_memories_empty_query(self):
        """Test memory retrieval with empty query."""
        result = retrieve_relevant_memories("")
        self.assertEqual(result, [])
    
    def test_retrieve_relevant_memories_no_results(self):
        """Test memory retrieval when no results are found."""
        query = "nonexistent query"
        
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock empty results
            mock_collection.query.return_value = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            
            result = retrieve_relevant_memories(query)
            self.assertEqual(result, [])
    
    def test_clear_memory_success(self):
        """Test successful memory clearing."""
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            result = clear_memory()
            self.assertTrue(result)
            mock_collection.delete.assert_called_once()
    
    def test_get_memory_stats_success(self):
        """Test successful memory statistics retrieval."""
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock collection count
            mock_collection.count.return_value = 42
            
            result = get_memory_stats()
            
            self.assertIsInstance(result, dict)
            self.assertIn('total_memories', result)
            self.assertEqual(result['total_memories'], 42)
            self.assertIn('memory_directory', result)
            self.assertEqual(result['memory_directory'], self.memory_dir)
    
    def test_search_memories_success(self):
        """Test successful memory search."""
        query = "search query"
        max_results = 3
        
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock search results
            mock_results = {
                'documents': [['Result 1', 'Result 2', 'Result 3']],
                'metadatas': [[{'source': 'src1'}, {'source': 'src2'}, {'source': 'src3'}]],
                'distances': [[0.1, 0.2, 0.3]]
            }
            mock_collection.query.return_value = mock_results
            
            result = search_memories(query, max_results)
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]['content'], 'Result 1')
            self.assertEqual(result[2]['content'], 'Result 3')
    
    def test_chromadb_connection_error(self):
        """Test handling of ChromaDB connection errors."""
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_chroma.PersistentClient.side_effect = Exception("Connection failed")
            
            # Test add_memory with connection error
            result = add_memory("test content", {"source": "test"})
            self.assertFalse(result)
            
            # Test retrieve_relevant_memories with connection error
            result = retrieve_relevant_memories("test query")
            self.assertEqual(result, [])
            
            # Test clear_memory with connection error
            result = clear_memory()
            self.assertFalse(result)
            
            # Test get_memory_stats with connection error
            result = get_memory_stats()
            self.assertIsInstance(result, dict)
            self.assertEqual(result['total_memories'], 0)
    
    def test_memory_directory_creation(self):
        """Test automatic creation of memory directory."""
        # Ensure directory doesn't exist initially
        if os.path.exists(self.memory_dir):
            shutil.rmtree(self.memory_dir)
        
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # This should create the directory
            add_memory("test content", {"source": "test"})
            
            # Verify directory was created
            self.assertTrue(os.path.exists(self.memory_dir))


class TestMemoryPerformance(unittest.TestCase):
    """Performance tests for memory operations."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.memory_dir = os.path.join(self.test_dir, "perf_memory")
        
        self.config_patcher = patch('memory.long_term.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.memory.memory_dir = self.memory_dir
        self.mock_config.memory.max_retrieval = 10
        self.mock_config.memory.max_memory_size = 10000
    
    def tearDown(self):
        """Clean up performance test environment."""
        self.config_patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_bulk_memory_addition_performance(self):
        """Test performance of adding multiple memories."""
        import time
        
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Test with 100 memories
            start_time = time.time()
            for i in range(100):
                add_memory(f"Memory {i}", {"index": i, "timestamp": "2024-01-01"})
            end_time = time.time()
            
            execution_time = end_time - start_time
            self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
            
            # Verify all calls were made
            self.assertEqual(mock_collection.add.call_count, 100)
    
    def test_large_query_performance(self):
        """Test performance of querying with large result set."""
        import time
        
        with patch('memory.long_term.chromadb') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock large result set
            large_documents = [f"Document {i}" for i in range(1000)]
            large_metadatas = [{"index": i} for i in range(1000)]
            large_distances = [[0.1] * 1000]
            
            mock_collection.query.return_value = {
                'documents': [large_documents],
                'metadatas': [large_metadatas],
                'distances': [large_distances]
            }
            
            start_time = time.time()
            result = retrieve_relevant_memories("test query")
            end_time = time.time()
            
            execution_time = end_time - start_time
            self.assertLess(execution_time, 2.0)  # Should complete within 2 seconds
            self.assertEqual(len(result), 1000)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
