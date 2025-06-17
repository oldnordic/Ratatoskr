"""
Comprehensive tests for the tools package.

This module contains unit tests for external tool integrations including
web search, browser automation, and other utility functions.

Test Coverage:
- Web search functionality
- Browser automation
- Text extraction
- Error handling
- Performance benchmarks
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os
import json

# Import the modules to test
from tools.web_search import perform_web_search
from tools.browser_tool import browse_search, extract_text_from_html


class TestWebSearch(unittest.TestCase):
    """Test suite for web search functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.test_query = "Python programming tutorials"
        self.expected_results = [
            {
                "title": "Python Tutorial - Learn Python Programming",
                "link": "https://example.com/python-tutorial",
                "snippet": "Learn Python programming with our comprehensive tutorial..."
            },
            {
                "title": "Python for Beginners",
                "link": "https://example.com/python-beginners",
                "snippet": "Start your Python journey with this beginner-friendly guide..."
            }
        ]
    
    @patch('tools.web_search.requests.get')
    def test_perform_web_search_success(self, mock_get):
        """Test successful web search."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractURL": "https://example.com",
            "Abstract": "Search results for Python tutorials",
            "Results": self.expected_results
        }
        mock_get.return_value = mock_response
        
        result = perform_web_search(self.test_query)
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        self.assertIn('summary', result)
        self.assertEqual(len(result['results']), 2)
        self.assertEqual(result['results'][0]['title'], self.expected_results[0]['title'])
    
    @patch('tools.web_search.requests.get')
    def test_perform_web_search_no_results(self, mock_get):
        """Test web search with no results."""
        # Mock response with no results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractURL": "",
            "Abstract": "",
            "Results": []
        }
        mock_get.return_value = mock_response
        
        result = perform_web_search(self.test_query)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['results']), 0)
        self.assertIn('No results found', result['summary'])
    
    @patch('tools.web_search.requests.get')
    def test_perform_web_search_http_error(self, mock_get):
        """Test web search with HTTP error."""
        # Mock HTTP error
        mock_get.side_effect = Exception("HTTP request failed")
        
        result = perform_web_search(self.test_query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result['summary'])
        self.assertEqual(len(result['results']), 0)
    
    @patch('tools.web_search.requests.get')
    def test_perform_web_search_timeout(self, mock_get):
        """Test web search with timeout."""
        # Mock timeout
        mock_get.side_effect = Exception("Request timeout")
        
        result = perform_web_search(self.test_query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('timeout', result['summary'].lower())
        self.assertEqual(len(result['results']), 0)
    
    def test_perform_web_search_empty_query(self):
        """Test web search with empty query."""
        result = perform_web_search("")
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result['summary'])
        self.assertEqual(len(result['results']), 0)
    
    def test_perform_web_search_none_query(self):
        """Test web search with None query."""
        result = perform_web_search(None)
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result['summary'])
        self.assertEqual(len(result['results']), 0)
    
    @patch('tools.web_search.requests.get')
    def test_perform_web_search_with_max_results(self, mock_get):
        """Test web search with max results limit."""
        # Mock response with many results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractURL": "https://example.com",
            "Abstract": "Search results",
            "Results": self.expected_results * 5  # 10 results total
        }
        mock_get.return_value = mock_response
        
        result = perform_web_search(self.test_query, max_results=3)
        
        self.assertEqual(len(result['results']), 3)
    
    @patch('tools.web_search.requests.get')
    def test_perform_web_search_response_format(self, mock_get):
        """Test web search response format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractURL": "https://example.com",
            "Abstract": "Search results for Python tutorials",
            "Results": self.expected_results
        }
        mock_get.return_value = mock_response
        
        result = perform_web_search(self.test_query)
        
        # Verify response structure
        self.assertIn('query', result)
        self.assertIn('results', result)
        self.assertIn('summary', result)
        self.assertIn('timestamp', result)
        
        # Verify result structure
        for res in result['results']:
            self.assertIn('title', res)
            self.assertIn('link', res)
            self.assertIn('snippet', res)


class TestBrowserTool(unittest.TestCase):
    """Test suite for browser automation functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.test_url = "https://example.com"
        self.test_html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Welcome to Test Page</h1>
                <p>This is a test paragraph with some content.</p>
                <div class="content">
                    <p>More content here.</p>
                </div>
            </body>
        </html>
        """
    
    @patch('tools.browser_tool.requests.get')
    def test_browse_search_success(self, mock_get):
        """Test successful web browsing."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self.test_html
        mock_response.headers = {'content-type': 'text/html'}
        mock_get.return_value = mock_response
        
        result = browse_search(self.test_url)
        
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertIn('title', result)
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'success')
        self.assertIn('Test Page', result['title'])
        self.assertIn('Welcome to Test Page', result['content'])
    
    @patch('tools.browser_tool.requests.get')
    def test_browse_search_http_error(self, mock_get):
        """Test web browsing with HTTP error."""
        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Page not found"
        mock_get.return_value = mock_response
        
        result = browse_search(self.test_url)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'error')
        self.assertIn('404', result['content'])
    
    @patch('tools.browser_tool.requests.get')
    def test_browse_search_connection_error(self, mock_get):
        """Test web browsing with connection error."""
        # Mock connection error
        mock_get.side_effect = Exception("Connection failed")
        
        result = browse_search(self.test_url)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'error')
        self.assertIn('Connection failed', result['content'])
    
    def test_browse_search_invalid_url(self):
        """Test web browsing with invalid URL."""
        result = browse_search("invalid-url")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'error')
        self.assertIn('Invalid URL', result['content'])
    
    def test_browse_search_empty_url(self):
        """Test web browsing with empty URL."""
        result = browse_search("")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'error')
        self.assertIn('URL cannot be empty', result['content'])
    
    def test_extract_text_from_html_success(self):
        """Test successful text extraction from HTML."""
        result = extract_text_from_html(self.test_html)
        
        self.assertIsInstance(result, str)
        self.assertIn('Welcome to Test Page', result)
        self.assertIn('This is a test paragraph', result)
        self.assertIn('More content here', result)
        # Should not contain HTML tags
        self.assertNotIn('<html>', result)
        self.assertNotIn('<body>', result)
        self.assertNotIn('<p>', result)
    
    def test_extract_text_from_html_empty(self):
        """Test text extraction from empty HTML."""
        result = extract_text_from_html("")
        
        self.assertEqual(result, "")
    
    def test_extract_text_from_html_none(self):
        """Test text extraction from None HTML."""
        result = extract_text_from_html(None)
        
        self.assertEqual(result, "")
    
    def test_extract_text_from_html_no_text(self):
        """Test text extraction from HTML with no text content."""
        html_no_text = "<html><head></head><body><div></div></body></html>"
        result = extract_text_from_html(html_no_text)
        
        self.assertEqual(result.strip(), "")
    
    def test_extract_text_from_html_complex(self):
        """Test text extraction from complex HTML."""
        complex_html = """
        <html>
            <head>
                <title>Complex Page</title>
                <script>var x = 1;</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <header>
                    <nav>
                        <a href="#">Home</a>
                        <a href="#">About</a>
                    </nav>
                </header>
                <main>
                    <article>
                        <h1>Main Article</h1>
                        <p>This is the main content.</p>
                        <blockquote>This is a quote.</blockquote>
                    </article>
                    <aside>
                        <h2>Sidebar</h2>
                        <ul>
                            <li>Item 1</li>
                            <li>Item 2</li>
                        </ul>
                    </aside>
                </main>
                <footer>
                    <p>Copyright 2024</p>
                </footer>
            </body>
        </html>
        """
        
        result = extract_text_from_html(complex_html)
        
        # Should extract text content
        self.assertIn('Complex Page', result)
        self.assertIn('Main Article', result)
        self.assertIn('This is the main content', result)
        self.assertIn('This is a quote', result)
        self.assertIn('Sidebar', result)
        self.assertIn('Item 1', result)
        self.assertIn('Item 2', result)
        self.assertIn('Copyright 2024', result)
        
        # Should not contain script or style content
        self.assertNotIn('var x = 1', result)
        self.assertNotIn('color: red', result)
        
        # Should not contain HTML tags
        self.assertNotIn('<html>', result)
        self.assertNotIn('<body>', result)
        self.assertNotIn('<p>', result)


class TestToolsIntegration(unittest.TestCase):
    """Integration tests for tools package."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_query = "Python programming"
        self.test_url = "https://example.com"
    
    @patch('tools.web_search.requests.get')
    @patch('tools.browser_tool.requests.get')
    def test_search_and_browse_workflow(self, mock_browser_get, mock_search_get):
        """Test complete search and browse workflow."""
        # Mock web search response
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "AbstractURL": "https://example.com",
            "Abstract": "Search results for Python programming",
            "Results": [{
                "title": "Python Programming Guide",
                "link": "https://example.com/python-guide",
                "snippet": "Learn Python programming..."
            }]
        }
        mock_search_get.return_value = search_response
        
        # Mock browser response
        browser_response = MagicMock()
        browser_response.status_code = 200
        browser_response.text = "<html><title>Python Guide</title><body>Content here</body></html>"
        browser_response.headers = {'content-type': 'text/html'}
        mock_browser_get.return_value = browser_response
        
        # Perform search
        search_result = perform_web_search(self.test_query)
        
        # Browse first result
        if search_result['results']:
            first_result_url = search_result['results'][0]['link']
            browse_result = browse_search(first_result_url)
            
            # Verify integration
            self.assertTrue(search_result['results'])
            self.assertEqual(browse_result['status'], 'success')
            self.assertIn('Python Guide', browse_result['title'])
    
    @patch('tools.web_search.requests.get')
    def test_search_with_text_extraction(self, mock_get):
        """Test search with text extraction from results."""
        # Mock response with HTML content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractURL": "https://example.com",
            "Abstract": "Search results",
            "Results": [{
                "title": "Test Page",
                "link": "https://example.com",
                "snippet": "<p>This is a test snippet</p>"
            }]
        }
        mock_get.return_value = mock_response
        
        result = perform_web_search(self.test_query)
        
        # Extract text from snippet
        if result['results']:
            snippet = result['results'][0]['snippet']
            extracted_text = extract_text_from_html(snippet)
            
            # Verify text extraction
            self.assertIn('This is a test snippet', extracted_text)
            self.assertNotIn('<p>', extracted_text)


class TestToolsPerformance(unittest.TestCase):
    """Performance tests for tools package."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.test_query = "performance test query"
        self.test_url = "https://example.com"
    
    @patch('tools.web_search.requests.get')
    def test_web_search_performance(self, mock_get):
        """Test web search performance."""
        import time
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "AbstractURL": "https://example.com",
            "Abstract": "Search results",
            "Results": [{"title": f"Result {i}", "link": f"https://example{i}.com", "snippet": f"Snippet {i}"} for i in range(100)]
        }
        mock_get.return_value = mock_response
        
        start_time = time.time()
        result = perform_web_search(self.test_query)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)  # Should complete within 1 second
        self.assertEqual(len(result['results']), 100)
    
    @patch('tools.browser_tool.requests.get')
    def test_browser_tool_performance(self, mock_get):
        """Test browser tool performance."""
        import time
        
        # Mock response with large HTML
        large_html = "<html><body>" + "<p>Content paragraph</p>" * 1000 + "</body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = large_html
        mock_response.headers = {'content-type': 'text/html'}
        mock_get.return_value = mock_response
        
        start_time = time.time()
        result = browse_search(self.test_url)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 2.0)  # Should complete within 2 seconds
        self.assertEqual(result['status'], 'success')
    
    def test_text_extraction_performance(self):
        """Test text extraction performance."""
        import time
        
        # Create large HTML content
        large_html = "<html><body>"
        for i in range(1000):
            large_html += f"<p>Paragraph {i} with some content</p>"
        large_html += "</body></html>"
        
        start_time = time.time()
        result = extract_text_from_html(large_html)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)  # Should complete within 1 second
        self.assertIn('Paragraph 0', result)
        self.assertIn('Paragraph 999', result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 