from __future__ import annotations
from rich import print
from src.tools.tools import search_web, scrape_url


result = search_web.invoke("What is the latest news on AI research?")
# get one of the search results and scrape it
out1 = result[0]["URL"] if isinstance(result, list) and len(result) > 0 else None

if out1:
    print(f"Scraping content from: {out1}")
    print(scrape_url.invoke(out1))
else:
    print("No valid search results to scrape.")
