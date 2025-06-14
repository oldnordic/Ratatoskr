# tools/browser_tool.py

from typing import Any
import re
from bs4 import BeautifulSoup

def browse_search(query: str, app_ref) -> str:
    """
    Loads DuckDuckGo (or any search URL) in the embedded browser
    then scrapes and returns the visible text of the first result page.
    """
    # build the search URL
    url = "https://duckduckgo.com/html/?q=" + query.replace(" ", "+")
    html_holder = {"content": None}

    def on_html(html: str):
        html_holder["content"] = html

    # connect to the bridge
    app_ref.browser_bridge.page_loaded.connect(on_html)
    app_ref.browser_bridge.navigate(url)

    # spin until page_loaded fires (in real code, use QEventLoop or callback)
    import time
    timeout = time.time() + 15
    while html_holder["content"] is None and time.time() < timeout:
        time.sleep(0.1)

    app_ref.browser_bridge.page_loaded.disconnect(on_html)

    if not html_holder["content"]:
        return "❌ Failed to load page."

    # simple text extraction
    soup = BeautifulSoup(html_holder["content"], "html.parser")
    # get the first result block
    first = soup.select_one(".result__a")
    if first and first.get("href"):
        # navigate to actual first link
        link = first["href"]
        html_holder["content"] = None
        app_ref.browser_bridge.page_loaded.connect(on_html)
        app_ref.browser_bridge.navigate(link)
        # wait again...
        timeout = time.time() + 15
        while html_holder["content"] is None and time.time() < timeout:
            time.sleep(0.1)
        app_ref.browser_bridge.page_loaded.disconnect(on_html)

        if not html_holder["content"]:
            return "❌ Failed to load result page."

        text = BeautifulSoup(html_holder["content"], "html.parser").get_text(separator="\n")
        # trim to first 1000 chars
        return text[:1000] + "\n\n[…]"
    else:
        return "❌ No results found."
