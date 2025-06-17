"""
Web browsing utilities for scraping web pages.

This module provides functionality to search and scrape web content using
DuckDuckGo search results. It includes HTML parsing and text extraction
capabilities for web content retrieval.
"""

import logging
import time
from typing import Optional
import httpx
from bs4 import BeautifulSoup

# Configuration constants
SEARCH_URL = "https://duckduckgo.com/html/"
REQUEST_TIMEOUT = 10  # seconds
PAGE_LOAD_TIMEOUT = 15  # seconds
MAX_TEXT_LENGTH = 1000  # characters
POLL_INTERVAL = 0.1  # seconds


def browse_search(query: str, app_ref: Optional[object] = None) -> str:
    """
    Perform a web search and return content from the first result page.
    
    This function searches DuckDuckGo for the given query and extracts
    text content from the first search result. It handles both direct
    HTTP requests and browser-based navigation if an app reference is provided.
    
    Args:
        query: The search query to perform
        app_ref: Optional application reference for browser-based navigation
        
    Returns:
        str: Extracted text content from the first search result, or error message
        
    Note:
        When app_ref is None, falls back to direct HTTP requests.
        When app_ref is provided, uses browser-based navigation for better
        JavaScript rendering support.
    """
    logging.info(f"Performing browse search for: '{query}'")
    
    # Use browser-based navigation if app reference is available
    if app_ref and hasattr(app_ref, 'browser_bridge'):
        return _browse_with_browser(query, app_ref)
    else:
        return _browse_with_http(query)


def _browse_with_http(query: str) -> str:
    """
    Perform web search using direct HTTP requests.
    
    Args:
        query: The search query to perform
        
    Returns:
        str: Extracted text content or error message
    """
    try:
        # Construct search URL
        search_url = f"{SEARCH_URL}?q={query.replace(' ', '+')}"
        
        # Perform initial search request
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(search_url)
            response.raise_for_status()
        
        # Parse search results
        soup = BeautifulSoup(response.text, "html.parser")
        first_result = soup.select_one(".result__a")
        
        if not first_result or not first_result.get("href"):
            return "❌ No search results found."
        
        # Extract link from first result
        result_link = first_result["href"]
        logging.info(f"Fetching content from: {result_link}")
        
        # Fetch the actual result page
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            page_response = client.get(result_link)
            page_response.raise_for_status()
        
        # Extract and clean text content
        page_soup = BeautifulSoup(page_response.text, "html.parser")
        text_content = page_soup.get_text(separator="\n")
        
        # Truncate and format result
        truncated_text = text_content[:MAX_TEXT_LENGTH]
        return f"{truncated_text}\n\n[...]"
        
    except httpx.HTTPError as e:
        logging.error(f"HTTP request failed: {e}")
        return f"❌ Failed to fetch web content: {e}"
    except Exception as e:
        logging.error(f"Unexpected error in HTTP browsing: {e}")
        return f"❌ Browse search error: {e}"


def _browse_with_browser(query: str, app_ref: object) -> str:
    """
    Perform web search using browser-based navigation.
    
    Args:
        query: The search query to perform
        app_ref: Application reference with browser bridge
        
    Returns:
        str: Extracted text content or error message
    """
    url = f"{SEARCH_URL}?q={query.replace(' ', '+')}"
    html_holder = {"content": None}

    def on_html(html: str) -> None:
        """Callback to capture page HTML once loaded."""
        html_holder["content"] = html

    try:
        # Connect to browser bridge and navigate to search page
        app_ref.browser_bridge.page_loaded.connect(on_html)
        app_ref.browser_bridge.navigate(url)

        # Wait for page to load
        timeout = time.time() + PAGE_LOAD_TIMEOUT
        while html_holder["content"] is None and time.time() < timeout:
            time.sleep(POLL_INTERVAL)

        app_ref.browser_bridge.page_loaded.disconnect(on_html)

        if not html_holder["content"]:
            return "❌ Failed to load search page."

        # Parse search results and get first link
        soup = BeautifulSoup(html_holder["content"], "html.parser")
        first_result = soup.select_one(".result__a")
        
        if not first_result or not first_result.get("href"):
            return "❌ No search results found."

        # Navigate to the first result page
        result_link = first_result["href"]
        html_holder["content"] = None
        
        app_ref.browser_bridge.page_loaded.connect(on_html)
        app_ref.browser_bridge.navigate(result_link)
        
        timeout = time.time() + PAGE_LOAD_TIMEOUT
        while html_holder["content"] is None and time.time() < timeout:
            time.sleep(POLL_INTERVAL)
            
        app_ref.browser_bridge.page_loaded.disconnect(on_html)

        if not html_holder["content"]:
            return "❌ Failed to load result page."

        # Extract and format text content
        text_content = BeautifulSoup(html_holder["content"], "html.parser").get_text(separator="\n")
        truncated_text = text_content[:MAX_TEXT_LENGTH]
        
        return f"{truncated_text}\n\n[...]"
        
    except Exception as e:
        logging.error(f"Browser-based browsing failed: {e}")
        return f"❌ Browser search error: {e}"


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text content from HTML.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        str: Cleaned text content
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Extract and clean text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        return "\n".join(chunk for chunk in chunks if chunk)
        
    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        return ""
