from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Sequence

from .records import Paper

FTS_TOKEN = re.compile(r"[\w-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Filters:
    year_min: int | None = None
    year_max: int | None = None
    min_citations: int | None = None
    open_access_only: bool = False
    topic: str | None = None
    field: str | None = None

    @property
    def active(self) -> bool:
        return any(
            value is not None and value is not False
            for value in (
                self.year_min,
                self.year_max,
                self.min_citations,
                self.open_access_only,
                self.topic,
                self.field,
            )
        )


def normalize_title(value: str) -> str:
    return " ".join(FTS_TOKEN.findall(value.casefold()))


class MetadataStore:
    """SQLite metadata for small generations.

    A sqlite3 connection must not run statements from two threads at once, and
    the API serves searches from a thread pool, so each thread gets its own
    connection to the same file.
    """

    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = path
        self.read_only = read_only
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self.connection  # open this thread's connection now so errors surface early

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        else:
            connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
        if not self.read_only:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
            with self._connections_lock:
                self._connections.append(connection)
        return connection

    def close(self) -> None:
        with self._connections_lock:
            connections, self._connections = self._connections, []
        for connection in connections:
            connection.close()
        self._local = threading.local()

    def create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                row_id INTEGER PRIMARY KEY,
                openalex_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                snippet TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                publication_year INTEGER NOT NULL,
                doi TEXT NOT NULL,
                cited_by_count INTEGER NOT NULL,
                topic TEXT NOT NULL,
                field_name TEXT NOT NULL,
                is_oa INTEGER NOT NULL CHECK (is_oa IN (0, 1)),
                oa_url TEXT NOT NULL,
                work_type TEXT NOT NULL DEFAULT '',
                venue TEXT NOT NULL DEFAULT '',
                publication_date TEXT NOT NULL DEFAULT '',
                landing_url TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS papers_year_idx ON papers(publication_year);
            CREATE INDEX IF NOT EXISTS papers_citations_idx ON papers(cited_by_count);
            CREATE INDEX IF NOT EXISTS papers_oa_idx ON papers(is_oa);
            CREATE INDEX IF NOT EXISTS papers_topic_idx ON papers(topic);
            CREATE INDEX IF NOT EXISTS papers_field_idx ON papers(field_name);
            CREATE INDEX IF NOT EXISTS papers_normalized_title_idx ON papers(normalized_title);
            CREATE VIRTUAL TABLE IF NOT EXISTS paper_titles USING fts5(
                title,
                content='papers',
                content_rowid='row_id',
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        self.connection.commit()

    def insert_paper_batch(self, papers: Sequence[Paper], start_row_id: int) -> None:
        """Streaming insert for large assemblies: rows only, no FTS work.
        Call build_fts() once after the last batch."""
        rows = [
            (
                start_row_id + offset,
                paper.openalex_id,
                paper.title,
                normalize_title(paper.title),
                paper.snippet,
                json.dumps(paper.authors, ensure_ascii=False, separators=(",", ":")),
                paper.publication_year,
                paper.doi,
                paper.cited_by_count,
                paper.topic,
                paper.field,
                int(paper.is_oa),
                paper.oa_url,
                paper.work_type,
                paper.venue,
                paper.publication_date,
                paper.landing_url,
            )
            for offset, paper in enumerate(papers)
        ]
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO papers (
                    row_id, openalex_id, title, normalized_title, snippet,
                    authors_json, publication_year, doi, cited_by_count,
                    topic, field_name, is_oa, oa_url,
                    work_type, venue, publication_date, landing_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def build_fts(self) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO paper_titles(paper_titles) VALUES('rebuild')")
            self.connection.execute("INSERT INTO paper_titles(paper_titles) VALUES('optimize')")

    def insert_papers(self, papers: Sequence[Paper]) -> None:
        rows = [
            (
                row_id,
                paper.openalex_id,
                paper.title,
                normalize_title(paper.title),
                paper.snippet,
                json.dumps(paper.authors, ensure_ascii=False, separators=(",", ":")),
                paper.publication_year,
                paper.doi,
                paper.cited_by_count,
                paper.topic,
                paper.field,
                int(paper.is_oa),
                paper.oa_url,
                paper.work_type,
                paper.venue,
                paper.publication_date,
                paper.landing_url,
            )
            for row_id, paper in enumerate(papers)
        ]
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO papers (
                    row_id, openalex_id, title, normalized_title, snippet,
                    authors_json, publication_year, doi, cited_by_count,
                    topic, field_name, is_oa, oa_url,
                    work_type, venue, publication_date, landing_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.connection.execute("INSERT INTO paper_titles(paper_titles) VALUES('rebuild')")
            self.connection.execute("INSERT INTO paper_titles(paper_titles) VALUES('optimize')")

    @staticmethod
    def _filter_sql(filters: Filters, *, prefix: str = "") -> tuple[list[str], list[object]]:
        column = lambda name: f"{prefix}{name}"
        clauses: list[str] = []
        values: list[object] = []
        if filters.year_min is not None or filters.year_max is not None:
            # Year 0 means "unknown"; a year filter must not admit it.
            clauses.append(f"{column('publication_year')} > 0")
        if filters.year_min is not None:
            clauses.append(f"{column('publication_year')} >= ?")
            values.append(filters.year_min)
        if filters.year_max is not None:
            clauses.append(f"{column('publication_year')} <= ?")
            values.append(filters.year_max)
        if filters.min_citations is not None:
            clauses.append(f"{column('cited_by_count')} >= ?")
            values.append(filters.min_citations)
        if filters.open_access_only:
            clauses.append(f"{column('is_oa')} = 1")
        if filters.topic:
            clauses.append(f"{column('topic')} = ?")
            values.append(filters.topic)
        if filters.field:
            clauses.append(f"{column('field_name')} = ?")
            values.append(filters.field)
        return clauses, values

    def count(self, filters: Filters = Filters()) -> int:
        clauses, values = self._filter_sql(filters)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.connection.execute(f"SELECT COUNT(*) FROM papers{where}", values).fetchone()
        return int(row[0])

    def eligible_ids(self, filters: Filters) -> list[int]:
        clauses, values = self._filter_sql(filters)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT row_id FROM papers{where} ORDER BY row_id", values
        )
        return [int(row[0]) for row in rows]

    def filter_candidate_ids(self, ids: Sequence[int], filters: Filters) -> list[int]:
        if not ids:
            return []
        if not filters.active:
            return [int(value) for value in ids]
        accepted: list[int] = []
        clauses, filter_values = self._filter_sql(filters)
        for offset in range(0, len(ids), 800):
            chunk = [int(value) for value in ids[offset : offset + 800]]
            placeholders = ",".join("?" for _ in chunk)
            where = [f"row_id IN ({placeholders})", *clauses]
            rows = self.connection.execute(
                f"SELECT row_id FROM papers WHERE {' AND '.join(where)}",
                [*chunk, *filter_values],
            )
            accepted.extend(int(row[0]) for row in rows)
        accepted_set = set(accepted)
        return [int(value) for value in ids if int(value) in accepted_set]

    def title_search(self, query: str, filters: Filters, limit: int = 20) -> list[int]:
        tokens = FTS_TOKEN.findall(query.casefold())
        if not tokens:
            return []
        fts_query = " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        clauses, values = self._filter_sql(filters, prefix="p.")
        where = ["paper_titles MATCH ?", *clauses]
        rows = self.connection.execute(
            f"""
            SELECT p.row_id
            FROM paper_titles
            JOIN papers p ON p.row_id = paper_titles.rowid
            WHERE {' AND '.join(where)}
            ORDER BY bm25(paper_titles), p.cited_by_count DESC
            LIMIT ?
            """,
            [fts_query, *values, limit],
        )
        return [int(row[0]) for row in rows]

    def title_key_exists(self, normalized: str) -> bool:
        """Whether any row's stored normalized title is exactly ``normalized``."""
        if not normalized:
            return False
        row = self.connection.execute(
            "SELECT 1 FROM papers WHERE normalized_title = ? LIMIT 1", (normalized,)
        ).fetchone()
        return row is not None

    def rarest_year(self) -> tuple[int, int] | None:
        row = self.connection.execute(
            """
            SELECT publication_year, COUNT(*) AS count
            FROM papers
            WHERE publication_year > 0
            GROUP BY publication_year
            ORDER BY count ASC, publication_year ASC
            LIMIT 1
            """
        ).fetchone()
        return (int(row[0]), int(row[1])) if row is not None else None

    def row_id_for_openalex_id(self, openalex_id: str) -> int | None:
        row = self.connection.execute(
            "SELECT row_id FROM papers WHERE openalex_id = ?",
            (openalex_id,),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def fetch(self, ids: Sequence[int]) -> dict[int, dict]:
        if not ids:
            return {}
        output: dict[int, dict] = {}
        for offset in range(0, len(ids), 800):
            chunk = [int(value) for value in ids[offset : offset + 800]]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT * FROM papers WHERE row_id IN ({placeholders})", chunk
            )
            for row in rows:
                # Older generations predate the richer card columns; default
                # them so mixed-generation serving keeps working.
                keys = row.keys()
                extra = lambda name: row[name] if name in keys else ""
                output[int(row["row_id"])] = {
                    "row_id": int(row["row_id"]),
                    "openalex_id": row["openalex_id"],
                    "title": row["title"],
                    "normalized_title": row["normalized_title"],
                    "snippet": row["snippet"],
                    "authors": json.loads(row["authors_json"]),
                    "publication_year": int(row["publication_year"]),
                    "doi": row["doi"],
                    "cited_by_count": int(row["cited_by_count"]),
                    "topic": row["topic"],
                    "field": row["field_name"],
                    "is_oa": bool(row["is_oa"]),
                    "oa_url": row["oa_url"],
                    "work_type": extra("work_type"),
                    "venue": extra("venue"),
                    "publication_date": extra("publication_date"),
                    "landing_url": extra("landing_url"),
                }
        return output
