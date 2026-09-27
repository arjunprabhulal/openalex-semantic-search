"""Representation B, implemented for real against 10k-v4 (read-only).

Minimal retrieval/filter index + block-compressed rich-card sidecar with
random access. Measures true bytes/record including all overheads, verifies a
card round-trips, and projects against the 300GB budget at 340.8M works.
"""
import json
import sqlite3
import subprocess
from pathlib import Path

GEN = Path("/var/lib/openalex-semantic-search/indexes/10k-v4")
TMP = Path("/var/lib/openalex-semantic-search/tmp/rep-b")
TMP.mkdir(parents=True, exist_ok=True)
RECORDS = 10_000
BLOCK = 512  # bigger blocks compress closer to whole-corpus ratio
DENOMINATOR = 340_800_000
BUDGET = 300e9

src = sqlite3.connect(f"file:{GEN}/metadata.sqlite3?mode=ro", uri=True)
src.row_factory = sqlite3.Row

# --- Minimal index: retrieval + filters + exact-title only ------------------
minimal = TMP / "index.sqlite3"
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
sidecar_records = []
for r in src.execute(
    "SELECT row_id, openalex_id, title, publication_year, cited_by_count, is_oa,"
    " topic, field_name, snippet, authors_json, doi, oa_url, landing_url, venue,"
    " work_type, publication_date FROM papers ORDER BY row_id"
):
    t = topics.setdefault(r["topic"], len(topics))
    f = fields_map.setdefault(r["field_name"], len(fields_map))
    rows.append((r["row_id"], int(r["openalex_id"].rsplit("W", 1)[1]), r["title"],
                 r["publication_year"], r["cited_by_count"], r["is_oa"], t, f))
    # Sidecar holds everything a card needs beyond the index row. DOI stored
    # bare (scheme re-added at render), URLs dropped when derivable from DOI.
    doi = r["doi"].removeprefix("https://doi.org/")
    entry = {
        "a": json.loads(r["authors_json"]),
        "s": r["snippet"][:1000],
        "v": r["venue"],
        "w": r["work_type"],
        "p": r["publication_date"],
    }
    if doi:
        entry["d"] = doi
    if r["landing_url"] and not doi:
        entry["l"] = r["landing_url"]
    if r["oa_url"] and r["oa_url"] != r["landing_url"] and not doi:
        entry["o"] = r["oa_url"]
    sidecar_records.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
m.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?)", rows)
m.executemany("INSERT INTO topics VALUES (?,?)", [(v, k) for k, v in topics.items()])
m.executemany("INSERT INTO fields VALUES (?,?)", [(v, k) for k, v in fields_map.items()])
m.execute("INSERT INTO paper_titles(paper_titles) VALUES('rebuild')")
m.execute("INSERT INTO paper_titles(paper_titles) VALUES('optimize')")
m.commit()
(TMP / "index-v.sqlite3").unlink(missing_ok=True)
m.execute(f"VACUUM INTO '{TMP / 'index-v.sqlite3'}'")
m.close()
index_bytes = (TMP / "index-v.sqlite3").stat().st_size

# --- Sidecar: fixed blocks of BLOCK records, zstd -19 each, offset table ----
blocks_dir = TMP / "blocks"
blocks_dir.mkdir(exist_ok=True)
offsets = []
sidecar_path = TMP / "cards.zsb"
with open(sidecar_path, "wb") as out:
    for i in range(0, RECORDS, BLOCK):
        raw = ("\n".join(sidecar_records[i : i + BLOCK])).encode()
        rawf = blocks_dir / "b.raw"
        rawf.write_bytes(raw)
        zf = blocks_dir / "b.zst"
        zf.unlink(missing_ok=True)
        subprocess.run(["zstd", "--ultra", "-22", "-q", "-f", str(rawf), "-o", str(zf)], check=True)
        offsets.append(out.tell())
        out.write(zf.read_bytes())
offset_table_bytes = len(offsets) * 8
sidecar_bytes = sidecar_path.stat().st_size + offset_table_bytes

# --- Round-trip check: decode a block, rebuild the flagship card ------------
import io
block_no = 0  # row_id 0 = Attention Is All You Need
with open(sidecar_path, "rb") as f:
    f.seek(offsets[block_no])
    end = offsets[block_no + 1] if block_no + 1 < len(offsets) else sidecar_path.stat().st_size
    comp = f.read(end - offsets[block_no])
decoded = subprocess.run(["zstd", "-d", "-q"], input=comp, capture_output=True, check=True).stdout
card = json.loads(decoded.decode().splitlines()[0])
idx = sqlite3.connect(f"file:{TMP / 'index-v.sqlite3'}?mode=ro", uri=True)
row = idx.execute(
    "SELECT title, publication_year, cited_by_count, is_oa FROM papers WHERE row_id=0"
).fetchone()
roundtrip_ok = (
    row[0] == "Attention Is All You Need"
    and card["a"][0] == "Ashish Vaswani"
    and len(card["s"]) == 1000
    and card["d"].startswith("10.48550")
)
fts_hit = idx.execute(
    "SELECT rowid FROM paper_titles WHERE paper_titles MATCH '\"attention\" AND \"need\"' LIMIT 1"
).fetchone()

# --- Verdict ----------------------------------------------------------------
vec = ((GEN / "int8-vectors.bin").stat().st_size
       + (GEN / "int8-scales.npy").stat().st_size
       + (GEN / "ivfpq.faiss").stat().st_size) / RECORDS
index_pr = index_bytes / RECORDS
side_pr = sidecar_bytes / RECORDS
total_pr = vec + index_pr + side_pr
proj = total_pr * DENOMINATOR

print("=== Representation B, measured (not projected from parts) ===")
print(f"  vectors             {vec:7.0f} B/record")
print(f"  minimal index       {index_pr:7.0f} B/record  ({index_bytes/1e6:.2f} MB)")
print(f"  card sidecar        {side_pr:7.0f} B/record  ({sidecar_bytes/1e6:.2f} MB, "
      f"{len(offsets)} blocks of {BLOCK}, zstd-22, snippet<=1000, offsets included)")
print(f"  TOTAL               {total_pr:7.0f} B/record")
print(f"  projected @ {DENOMINATOR/1e6:.1f}M: {proj/1e9:.1f} GB  "
      f"(budget 300 GB) -> {'PASS' if proj <= BUDGET else 'FAIL'}")
print(f"  card round-trip: {'OK' if roundtrip_ok else 'FAILED'} "
      f"(authors+abstract+doi from sidecar, title/year/citations/oa from index)")
print(f"  exact-title FTS still works: {'OK' if fts_hit and fts_hit[0] == 0 else 'FAILED'}")
