"""Storage experiments against the immutable 10k-v4 generation (read-only).

Answers, per the 10K stage review:
1. SQLite size by component (tables, indexes, FTS)
2. Per-field payload bytes
3. Compressed abstracts (and a full rich-card sidecar) outside SQLite
4. A retrieval/filter-only index variant
5. Feasibility notes for on-demand card metadata by OpenAlex ID
6. A measured production-corpus denominator from the snapshot manifest
7. Projections for each representation vs the 300GB budget
"""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

GEN = Path("/var/lib/openalex-semantic-search/indexes/10k-v4")
TMP = Path("/var/lib/openalex-semantic-search/tmp/storage-exp")
TMP.mkdir(parents=True, exist_ok=True)
RECORDS = 10_000
VECTOR_BYTES_PER_RECORD = None  # measured below from real artifacts

src = sqlite3.connect(f"file:{GEN}/metadata.sqlite3?mode=ro", uri=True)
src.row_factory = sqlite3.Row


def vacuum_size(setup_sql: list[str], name: str) -> int:
    """Copy the DB, apply DDL, VACUUM into a fresh file, return its size."""
    work = TMP / f"{name}.work.sqlite3"
    out = TMP / f"{name}.sqlite3"
    for p in (work, out):
        p.unlink(missing_ok=True)
    subprocess.run(
        ["cp", str(GEN / "metadata.sqlite3"), str(work)], check=True
    )
    c = sqlite3.connect(work)
    for stmt in setup_sql:
        c.executescript(stmt)
    c.commit()
    c.execute(f"VACUUM INTO '{out}'")
    c.close()
    work.unlink()
    return out.stat().st_size


print("=== 1. SQLite by component (VACUUMed sizes) ===")
full = vacuum_size([], "full")
no_fts = vacuum_size(["DROP TABLE paper_titles;"], "no-fts")
no_idx = vacuum_size(
    [
        "DROP INDEX papers_year_idx; DROP INDEX papers_citations_idx;"
        "DROP INDEX papers_oa_idx; DROP INDEX papers_topic_idx;"
        "DROP INDEX papers_field_idx; DROP INDEX papers_normalized_title_idx;"
    ],
    "no-indexes",
)
fts_cost = full - no_fts
idx_cost = full - no_idx
table_cost = full - fts_cost - idx_cost
for label, size in [
    ("full (defragmented)", full),
    ("papers table+payload", table_cost),
    ("title FTS", fts_cost),
    ("filter/title indexes", idx_cost),
]:
    print(f"  {label:24s} {size/1e6:8.2f} MB  {size/RECORDS:7.0f} B/record")

print("\n=== 2. Per-field payload bytes (SUM(length)) ===")
fields = [
    "openalex_id", "title", "normalized_title", "snippet", "authors_json",
    "doi", "oa_url", "landing_url", "venue", "topic", "field_name",
    "work_type", "publication_date",
]
field_bytes = {}
for f in fields:
    field_bytes[f] = src.execute(f"SELECT SUM(length({f})) FROM papers").fetchone()[0] or 0
for f, b in sorted(field_bytes.items(), key=lambda kv: -kv[1]):
    print(f"  {f:18s} {b/1e6:7.2f} MB  {b/RECORDS:6.0f} B/record")

print("\n=== 3. Compression outside SQLite (zstd -15) ===")


def zstd_bytes(payload: bytes, name: str) -> int:
    raw = TMP / f"{name}.raw"
    raw.write_bytes(payload)
    out = TMP / f"{name}.zst"
    out.unlink(missing_ok=True)
    subprocess.run(["zstd", "-15", "-q", "-f", str(raw), "-o", str(out)], check=True)
    size = out.stat().st_size
    raw.unlink()
    return size


abstracts = "\n".join(
    r[0] for r in src.execute("SELECT snippet FROM papers ORDER BY row_id")
).encode()
abs_z = zstd_bytes(abstracts, "abstracts")
print(f"  abstracts: {len(abstracts)/1e6:.2f} MB raw -> {abs_z/1e6:.2f} MB "
      f"({abs_z/RECORDS:.0f} B/record, ratio {len(abstracts)/max(abs_z,1):.1f}x)")

rich_rows = []
for r in src.execute(
    "SELECT snippet, authors_json, doi, oa_url, landing_url, venue,"
    " work_type, publication_date FROM papers ORDER BY row_id"
):
    rich_rows.append(json.dumps(dict(r), ensure_ascii=False, separators=(",", ":")))
rich = "\n".join(rich_rows).encode()
rich_z = zstd_bytes(rich, "rich-sidecar")
print(f"  full card sidecar (abstract+authors+urls+venue+type+date):")
print(f"    {len(rich)/1e6:.2f} MB raw -> {rich_z/1e6:.2f} MB "
      f"({rich_z/RECORDS:.0f} B/record, ratio {len(rich)/max(rich_z,1):.1f}x)")
print("    (block storage for random access adds ~5-10% overhead)")

print("\n=== 4. Retrieval/filter-only SQLite variant ===")
minimal = TMP / "minimal.sqlite3"
minimal.unlink(missing_ok=True)
m = sqlite3.connect(minimal)
m.executescript(
    """
    CREATE TABLE papers (
        row_id INTEGER PRIMARY KEY,
        openalex_key INTEGER NOT NULL,
        title TEXT NOT NULL,
        publication_year INTEGER NOT NULL,
        cited_by_count INTEGER NOT NULL,
        is_oa INTEGER NOT NULL,
        topic_id INTEGER NOT NULL,
        field_id INTEGER NOT NULL
    );
    CREATE TABLE topics (topic_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
    CREATE TABLE fields (field_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
    CREATE INDEX p_year ON papers(publication_year);
    CREATE INDEX p_cited ON papers(cited_by_count);
    CREATE INDEX p_topic ON papers(topic_id);
    CREATE INDEX p_field ON papers(field_id);
    CREATE VIRTUAL TABLE paper_titles USING fts5(
        title, content='papers', content_rowid='row_id',
        tokenize='unicode61 remove_diacritics 2'
    );
    """
)
topics, fields_map = {}, {}
rows = []
for r in src.execute(
    "SELECT row_id, openalex_id, title, publication_year, cited_by_count,"
    " is_oa, topic, field_name FROM papers ORDER BY row_id"
):
    t = topics.setdefault(r["topic"], len(topics))
    f = fields_map.setdefault(r["field_name"], len(fields_map))
    key = int(r["openalex_id"].rsplit("W", 1)[1])  # numeric OpenAlex work id
    rows.append((r["row_id"], key, r["title"], r["publication_year"],
                 r["cited_by_count"], r["is_oa"], t, f))
m.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?)", rows)
m.executemany("INSERT INTO topics VALUES (?,?)", [(v, k) for k, v in topics.items()])
m.executemany("INSERT INTO fields VALUES (?,?)", [(v, k) for k, v in fields_map.items()])
m.execute("INSERT INTO paper_titles(paper_titles) VALUES('rebuild')")
m.execute("INSERT INTO paper_titles(paper_titles) VALUES('optimize')")
m.commit()
m.execute(f"VACUUM INTO '{TMP / 'minimal-v.sqlite3'}'")
m.close()
minimal_size = (TMP / "minimal-v.sqlite3").stat().st_size
print(f"  minimal index (id, title+FTS, year, citations, oa, topic/field as ints):")
print(f"    {minimal_size/1e6:.2f} MB  {minimal_size/RECORDS:.0f} B/record")

print("\n=== 6. Measured denominator from the snapshot manifest ===")
import boto3
from botocore import UNSIGNED
from botocore.client import Config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openalex_semantic_search.records import parse_work, _iter_json_lines  # noqa: E402
import gzip
client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
resp = client.get_object(Bucket="openalex", Key="data/jsonl/works/manifest.json")
manifest = json.loads(resp["Body"].read())
total_snapshot = sum(
    (f.get("meta") or {}).get("record_count", 0)
    for f in manifest["files"] if isinstance(f, dict)
)
sample_files = [manifest["files"][len(manifest["files"]) // 3],
                manifest["files"][2 * len(manifest["files"]) // 3]]
raw_n = usable_n = 0
for item in sample_files:
    key = item["url"].split("s3://openalex/", 1)[1]
    obj = client.get_object(Bucket="openalex", Key=key)
    with gzip.GzipFile(fileobj=obj["Body"]) as arch:
        for i, work in enumerate(_iter_json_lines(arch)):
            if i >= 3_000:
                break
            raw_n += 1
            if parse_work(work) is not None:
                usable_n += 1
    obj["Body"].close()
ratio = usable_n / raw_n
denominator = int(total_snapshot * ratio)
print(f"  snapshot manifest total: {total_snapshot:,} records")
print(f"  usable-title ratio (sample {raw_n:,} from 2 files): {ratio:.3f}")
print(f"  measured denominator: {denominator:,} usable works")

print("\n=== 7. Projections (vectors measured from artifacts) ===")
int8 = (GEN / "int8-vectors.bin").stat().st_size + (GEN / "int8-scales.npy").stat().st_size
faiss_idx = (GEN / "ivfpq.faiss").stat().st_size
vec = (int8 + faiss_idx) / RECORDS
print(f"  vector layers: {vec:.0f} B/record (INT8 {int8/RECORDS:.0f} + IVF-PQ {faiss_idx/RECORDS:.0f})")
budget = 300e9
reps = {
    "A. current rich SQLite (v4 baseline)": vec + full / RECORDS,
    "B. minimal index + zstd rich sidecar (+8% block overhead)": vec + minimal_size / RECORDS + rich_z / RECORDS * 1.08,
    "C. minimal index + on-demand OpenAlex API for cards": vec + minimal_size / RECORDS,
    "D. minimal index + zstd abstracts only, rest on demand": vec + minimal_size / RECORDS + abs_z / RECORDS * 1.08,
}
print(f"  budget: 300GB; allowance at measured denominator: {budget/denominator:.0f} B/record")
for name, bpr in reps.items():
    proj = bpr * denominator / 1e9
    verdict = "PASS" if proj <= 300 else "FAIL"
    print(f"  [{verdict}] {name}: {bpr:.0f} B/record -> {proj:.1f} GB")
