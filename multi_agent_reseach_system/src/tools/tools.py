# Web search tool

import os
from langchain.tools import tool
import requests
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    """
    Search the web for a given query using Tavily API.

    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    try:
        response = tavily.search(query, max_results=2, search_depth="basic")
        results = response.get("results", [])
        if not results:
            return "No results found."
        out = []
        for result in results:
            out.append(f"Title: {result.get('title', 'N/A')}\nURL: {result.get('url', 'N/A')}\nContent: {result.get('content', 'N/A')}\n")
        return "\n".join(out)
    except Exception as e:
        return f"An error occurred while searching the web: {str(e)}"