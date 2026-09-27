"""Build a searchable index from your own documents instead of OpenAlex.

Input is JSON Lines, one document per line:

    {"id": "doc-1", "title": "...", "body": "...", "year": 2024, "popularity": 12, "category": "..."}

Only "id" and "title" are required. Then serve it like any other index:

    python examples/custom_corpus.py docs.jsonl indexes/my-docs
    openalex-semantic-search serve --generation indexes/my-docs --port 8100
"""

import json
import sys
from pathlib import Path

from openalex_semantic_search.builder import build_generation
from openalex_semantic_search.config import Stage
from openalex_semantic_search.embeddings import SentenceTransformerEmbedder
from openalex_semantic_search.records import Paper


def to_record(doc: dict) -> Paper:
    body = doc.get("body", "")
    return Paper(
        openalex_id=str(doc["id"]),                      # stable unique id
        title=doc["title"],
        embedding_text=f"{doc['title']}. {body[:2000]}",  # what gets embedded
        snippet=body[:1500],                             # shown in results
        authors=tuple(doc.get("authors", ())),
        publication_year=int(doc.get("year", 0)),        # year filters and sorts
        cited_by_count=int(doc.get("popularity", 0)),    # popularity boost and sort
        topic=doc.get("category", ""),                   # topic filter
        field="", doi="", is_oa=True, oa_url="",
    )


def main(source: Path, output: Path) -> None:
    with source.open(encoding="utf-8") as lines:
        records = [to_record(json.loads(line)) for line in lines if line.strip()]
    # IVF lists grow with corpus size; about 4 * sqrt(n) is a reasonable start.
    lists = max(16, min(131_072, int(4 * len(records) ** 0.5)))
    stage = Stage("custom", len(records), lists, max(8, lists // 8))
    manifest = build_generation(
        records, stage=stage, output=output, embedder=SentenceTransformerEmbedder()
    )
    print(f"Built {manifest['records']:,} records into {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
