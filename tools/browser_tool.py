"""Utilities for scraping the first result page in an embedded browser."""

from bs4 import BeautifulSoup


def browse_search(query: str, app_ref) -> str:
    """Load a result page and return its visible text."""
    url = "https://duckduckgo.com/html/?q=" + query.replace(" ", "+")
    html_holder = {"content": None}

    def on_html(html: str) -> None:
        html_holder["content"] = html

    app_ref.browser_bridge.page_loaded.connect(on_html)
    app_ref.browser_bridge.navigate(url)

    import time

    timeout = time.time() + 15
    while html_holder["content"] is None and time.time() < timeout:
        time.sleep(0.1)

    app_ref.browser_bridge.page_loaded.disconnect(on_html)

    if not html_holder["content"]:
        return "❌ Failed to load page."

    soup = BeautifulSoup(html_holder["content"], "html.parser")
    first = soup.select_one(".result__a")
    if first and first.get("href"):
        link = first["href"]
        html_holder["content"] = None
        app_ref.browser_bridge.page_loaded.connect(on_html)
        app_ref.browser_bridge.navigate(link)
        timeout = time.time() + 15
        while html_holder["content"] is None and time.time() < timeout:
            time.sleep(0.1)
        app_ref.browser_bridge.page_loaded.disconnect(on_html)

        if not html_holder["content"]:
            return "❌ Failed to load result page."

        text = BeautifulSoup(html_holder["content"], "html.parser").get_text(separator="\n")
        return text[:1000] + "\n\n[…]"
    return "❌ No results found."
