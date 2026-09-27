"""Search a running openalex-semantic-search server.

Usage:
    export OPENALEX_SEARCH_API_KEYS=...   # the key the server was started with
    python examples/search.py "graph neural networks for protein folding"
"""

import json
import os
import sys
import urllib.request

URL = os.environ.get("OPENALEX_SEARCH_URL", "http://127.0.0.1:8100")
KEY = os.environ["OPENALEX_SEARCH_API_KEYS"].split(",")[0]


def search(query: str, **filters) -> dict:
    body = json.dumps({"query": query, "limit": 10, **filters}).encode()
    request = urllib.request.Request(
        f"{URL}/search",
        data=body,
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "attention is all you need"
    for result in search(query)["results"]:
        print(f"{result.get('publication_year', '')}  {result.get('title', '')}")
