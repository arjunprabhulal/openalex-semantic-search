#!/usr/bin/env bash
# Filtered and sorted searches against a running server.
# Usage: OPENALEX_SEARCH_API_KEYS=... ./examples/filters.sh
set -euo pipefail
URL="${OPENALEX_SEARCH_URL:-http://127.0.0.1:8100}"
KEY="${OPENALEX_SEARCH_API_KEYS%%,*}"

search() {
  curl -fsS "$URL/search" -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d "$1"
  echo
}

# Recent, well-cited work on a topic
search '{"query": "retrieval augmented generation", "year_min": 2022, "min_citations": 50, "limit": 5}'

# Newest open-access papers
search '{"query": "coastal wetlands blue carbon", "sort": "newest", "open_access_only": true, "limit": 5}'
