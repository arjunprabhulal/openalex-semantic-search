"""Score the labelled queries against a live /search endpoint.

Read-only: it only sends POST /search requests, spaced apart. The key comes
from the environment and is never printed or passed on a command line.

    export OPENALEX_SEARCH_API_KEY=...          # one application key
    python benchmark/run_labelled.py \
        --url http://127.0.0.1:8109/search \
        --baseline benchmark_table.md \
        --out after.json

Prints recall@1/5/20 per query type, the behaviour checks, and, with
--baseline, each labelled query's rank before and after. Standard library only,
so it runs on the server without installing anything.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent


def norm(text: str) -> str:
    return " ".join(re.findall(r"[\w-]+", (text or "").casefold()))


def matches(result: dict, expect: dict) -> bool:
    if result.get("work_id") in expect.get("work_ids", ()):
        return True
    titles = expect.get("title_contains")
    authors = expect.get("author_contains")
    if not titles and not authors:
        return False
    title_ok = not titles or any(norm(t) in norm(result.get("title", "")) for t in titles)
    author_ok = not authors or any(
        a.lower() in name.lower() for a in authors for name in result.get("authors") or ()
    )
    return title_ok and author_ok


def post(url: str, key: str, body: dict, timeout: float) -> tuple[int, dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            return response.status, payload, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, {"detail": error.read().decode("utf-8", "replace")}, (
            time.perf_counter() - started
        ) * 1000


def suspect_ids() -> set[str]:
    """Work ids benchmark/suspect-citation-records.json flags as unreliable."""
    path = HERE / "suspect-citation-records.json"
    if not path.exists():
        return set()
    works = json.loads(path.read_text(encoding="utf-8")).get("works", [])
    return {work["work_id"] for work in works if work.get("flag_citations")}


def run_check(check: str, body: dict) -> tuple[bool, str]:
    results = body.get("results") or []
    name, _, argument = check.partition(":")
    if name in ("sorted_desc", "sorted_asc"):
        values = [r.get(argument) for r in results]
        reverse = name == "sorted_desc"
        # The sort is exact over rows that clear the relevance floor; weaker
        # rows may follow in their own sorted run. Report the leading run.
        run = 1
        while run < len(values) and (
            values[run] <= values[run - 1] if reverse else values[run] >= values[run - 1]
        ):
            run += 1
        return run >= min(len(values), 10), f"leading sorted run {run}/{len(values)}"
    if name == "first_is":
        first = results[0]["work_id"] if results else None
        return first == argument, f"first={first}"
    if name == "no_duplicate_titles":
        seen: dict[str, str] = {}
        for r in results:
            key = norm(r.get("title", ""))
            author = (r.get("authors") or [""])[0].split()[-1:] or [""]
            signature = key + "|" + author[0].lower()
            if signature in seen:
                return False, f"{r['work_id']} duplicates {seen[signature]}"
            seen[signature] = r["work_id"]
        return True, f"{len(results)} results, no same-title same-author pair"
    if name == "years_within":
        low, high = (int(v) for v in argument.split(":"))
        bad = [r["publication_year"] for r in results if not low <= r["publication_year"] <= high]
        return bool(results) and not bad, f"{len(results)} results, out of range {bad[:5]}"
    if name == "first_not_flagged":
        # Records whose citation count belongs to another work must not lead a
        # most_cited list: neither by the served flag nor by the suspect list.
        first = results[0] if results else {}
        flagged = bool(first.get("citation_count_unreliable")) or first.get("work_id") in suspect_ids()
        return bool(results) and not flagged, f"first={first.get('work_id')} cited={first.get('cited_by_count')}"
    if name == "non_empty":
        return bool(results), f"{len(results)} results"
    if name == "page_consistent":
        has_more = body.get("has_more")
        next_offset = body.get("next_offset")
        ok = bool(results) and (
            (has_more and next_offset == body["offset"] + len(results))
            or (not has_more and next_offset is None)
        )
        return ok, f"offset={body.get('offset')} n={len(results)} has_more={has_more} next={next_offset} total={body.get('total_matches')}"
    if name == "past_pool":
        ok = not results and body.get("has_more") is False and body.get("offset") == body.get(
            "total_matches"
        )
        return ok, f"offset={body.get('offset')} total={body.get('total_matches')} has_more={body.get('has_more')}"
    return False, f"unknown check {check}"


def read_baseline(path: Path) -> dict[str, str]:
    ranks: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and re.fullmatch(r"[A-Z]\d\d", cells[0]):
            ranks[cells[0]] = cells[4]
    return ranks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.environ.get("OPENALEX_SEARCH_URL", "http://127.0.0.1:8109/search"))
    parser.add_argument("--queries", type=Path, default=HERE / "labelled-queries.json")
    parser.add_argument("--baseline", type=Path, help="benchmark_table.md from the audit")
    parser.add_argument("--out", type=Path, help="write raw responses and scores as JSON")
    parser.add_argument("--delay", type=float, default=1.2, help="seconds between requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--only", help="comma-separated query ids")
    parser.add_argument("--include-heavy", action="store_true")
    args = parser.parse_args(argv)

    key = os.environ.get("OPENALEX_SEARCH_API_KEY", "")
    if not key:
        print("Set OPENALEX_SEARCH_API_KEY to one application key.", file=sys.stderr)
        return 2
    queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    only = set(args.only.split(",")) if args.only else None
    baseline = read_baseline(args.baseline) if args.baseline else {}

    rows, checks, raw = [], [], {}
    for index, entry in enumerate(queries):
        if only and entry["id"] not in only:
            continue
        if entry.get("heavy") and not args.include_heavy:
            continue
        if index:
            time.sleep(args.delay)
        status, body, elapsed = post(args.url, key, entry["request"], args.timeout)
        raw[entry["id"]] = {"status": status, "elapsed_ms": round(elapsed, 1), "body": body}
        if status != 200:
            print(f"{entry['id']}: HTTP {status} {str(body)[:200]}")
            continue
        if "check" in entry:
            ok, detail = run_check(entry["check"], body)
            checks.append((entry["id"], entry["type"], entry["check"], ok, detail, elapsed))
            continue
        rank = next(
            (i + 1 for i, result in enumerate(body["results"]) if matches(result, entry["expect"])),
            None,
        )
        rows.append((entry["id"], entry["type"], entry["request"]["query"], rank, elapsed, entry.get("note")))

    print("| id | type | query | rank | before | ms |")
    print("|---|---|---|---|---|---|")
    changes = {"better": 0, "worse": 0, "same": 0}
    for qid, qtype, query, rank, elapsed, note in rows:
        before = baseline.get(qid, "")
        now = str(rank) if rank else ">20"
        if before:
            b = 21 if before.startswith(">") else int(before)
            n = 21 if rank is None else rank
            changes["better" if n < b else "worse" if n > b else "same"] += 1
        flag = " (known gap)" if note and "gap" in note.lower() else ""
        print(f"| {qid} | {qtype} | `{query[:60]}` | {now}{flag} | {before} | {elapsed:.0f} |")
    print()
    print("| type | n | recall@1 | recall@5 | recall@20 |")
    print("|---|---|---|---|---|")
    by_type: dict[str, list[int | None]] = {}
    for _, qtype, _, rank, _, _ in rows:
        by_type.setdefault(qtype, []).append(rank)
    total = [0, 0, 0, 0]
    for qtype, ranks in by_type.items():
        hits = [sum(1 for r in ranks if r and r <= k) for k in (1, 5, 20)]
        total = [total[0] + len(ranks), *(t + h for t, h in zip(total[1:], hits))]
        print(f"| {qtype} | {len(ranks)} | {hits[0]}/{len(ranks)} | {hits[1]}/{len(ranks)} | {hits[2]}/{len(ranks)} |")
    if total[0]:
        n = total[0]
        print(f"| **all** | {n} | {total[1]}/{n} ({total[1]/n:.0%}) | {total[2]}/{n} ({total[2]/n:.0%}) | {total[3]}/{n} ({total[3]/n:.0%}) |")
    if baseline:
        print(f"\nAgainst baseline: {changes['better']} better, {changes['worse']} worse, {changes['same']} same")
    print("\n| id | type | check | pass | detail | ms |")
    print("|---|---|---|---|---|---|")
    for qid, qtype, check, ok, detail, elapsed in checks:
        print(f"| {qid} | {qtype} | {check} | {'Y' if ok else 'N'} | {detail} | {elapsed:.0f} |")
    if args.out:
        args.out.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    failed_checks = sum(1 for check in checks if not check[3])
    return 1 if failed_checks or changes["worse"] else 0


if __name__ == "__main__":
    sys.exit(main())
