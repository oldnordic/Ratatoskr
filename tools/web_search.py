import logging
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

def _extract_text_from_html(html_content: str) -> str:
    """Uses BeautifulSoup to extract clean text from HTML content."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        return ""

def perform_web_search(query: str) -> str:
    """
    Performs a refined web search, fetches content, and returns cleaned text.
    """
    logging.info(f"Performing web search for: '{query}'")
    
    # --- CHANGE: Create a more focused search query ---
    # We add keywords and site restrictions for better results.
    refined_query = f"top news headlines site:dw.com"
    logging.info(f"Refined search query to: '{refined_query}'")
    
    try:
        with DDGS() as ddgs:
            # --- CHANGE: Added timelimit='d' to get results from the last day ---
            results = [r for r in ddgs.text(refined_query, max_results=3, timelimit='d')]
            
            if not results:
                logging.warning("Web search returned no recent results.")
                return "I couldn't find any news from the last day on DW.com."

            all_content = []
            for i, result in enumerate(results):
                url = result.get('href')
                if not url: continue
                
                logging.info(f"Fetching content from URL: {url}")
                try:
                    response = requests.get(url, headers={'User-Agent': 'RatatoskrBot/1.0'}, timeout=8)
                    response.raise_for_status()
                    page_text = _extract_text_from_html(response.text)
                    if page_text:
                        # Prepend the article title for better context
                        title = result.get('title', 'Source')
                        all_content.append(f"From {title}: {page_text[:1500]}")
                except requests.RequestException as e:
                    logging.warning(f"Could not fetch or read URL {url}: {e}")

            if not all_content:
                return "Could not retrieve readable content from today's news."

            return "\n\n".join(all_content)

    except Exception as e:
        logging.error(f"An unexpected error occurred during web search: {e}")
        return "An error occurred while performing the web search."