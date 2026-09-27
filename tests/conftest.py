"""A production-shaped generation for query-time tests.

The older engine tests set candidate_count equal to the corpus, so the whole
corpus is the candidate pool and sort, pinning, dedup and filters are never
exercised the way they run on 315M rows. This builds 5,000 rows served with a
50-row candidate pool, on the compact metadata backend production uses, with
the title-prefix and work-id lookup indexes loaded beside it.
"""

from __future__ import annotations

from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from openalex_semantic_search.builder import build_generation
from openalex_semantic_search.compact_store import (
    CompactMetadataStore,
    build_compact_metadata,
    stable_text_hash,
)
from openalex_semantic_search.config import Stage
from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.records import Paper
from openalex_semantic_search.title_prefix import TitlePrefixIndex, build_title_prefix_index
from openalex_semantic_search.work_index import WorkIdIndex, normalize_work_id

CORPUS_ROWS = 5_000
CANDIDATE_POOL = 50


def work(
    key: str,
    title: str,
    *,
    abstract: str = "",
    year: int = 2010,
    citations: int = 0,
    author: str = "",
    doi: str = "",
    topic: str = "Topic special",
    field: str = "Computer Science",
) -> Paper:
    return Paper(
        openalex_id=f"https://openalex.org/{key}",
        title=title,
        embedding_text=f"{title}. {abstract}" if abstract else title,
        snippet=abstract,
        authors=(author,) if author else (),
        publication_year=year,
        doi=doi,
        cited_by_count=citations,
        topic=topic,
        field=field,
        is_oa=False,
        oa_url="",
    )


def special_works() -> list[Paper]:
    works = [
        work(
            "W3098425262",
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            abstract="Parametric and non-parametric memory for language generation",
            year=2020,
            citations=18,
            author="Patrick Lewis",
        ),
        # Stored with U+2010, the way publisher metadata often spells it.
        work(
            "W2963341956",
            "BERT: Pre‐training of Deep Bidirectional Transformers for Language Understanding",
            abstract="Bidirectional encoder representations from transformers",
            year=2019,
            citations=90_000,
            author="Jacob Devlin",
        ),
        work(
            "WPRETRAIN",
            "Pretraining Language Models at Scale",
            abstract="Scaling laws for language model pretraining",
            year=2021,
            citations=50,
            author="Ada Author",
        ),
        # Two OpenAlex records of one paper, and a different paper that
        # happens to share the title.
        work(
            "W2194775991",
            "Deep Residual Learning for Image Recognition",
            abstract="Residual networks make deep image models easier to optimize",
            year=2016,
            citations=229_638,
            author="Kaiming He",
            doi="https://doi.org/10.1109/cvpr.2016.90",
        ),
        work(
            "W2949650786",
            "Deep Residual Learning for Image Recognition",
            abstract="Residual networks make deep image models easier to optimize",
            year=2015,
            citations=4_772,
            author="Kaiming He",
            doi="https://doi.org/10.48550/arxiv.1512.03385",
        ),
        work(
            "WRESNETOTHER",
            "Deep Residual Learning for Image Recognition",
            abstract="A course report reproducing residual networks",
            year=2022,
            citations=10,
            author="Someone Else",
        ),
        # Sidecar year 2025 is wrong; the correction says 2017.
        work(
            "W2626778328",
            "Attention Is All You Need",
            abstract="Transformer attention architecture for sequence modeling",
            year=2025,
            citations=7_679,
            author="Ashish Vaswani",
        ),
        work(
            "WEFFORT",
            "Attention and Effort",
            abstract="Attention and effort in cognition",
            year=1973,
            citations=500_000,
            author="Daniel Kahneman",
        ),
        work(
            "WCRISPRWHEAT",
            "CRISPR–Cas9 genome editing in wheat",
            abstract="Genome editing of wheat with CRISPR Cas9",
            year=2017,
            citations=358,
            author="Dong Kim",
        ),
        work(
            "WCRISPRSTEM",
            "CRISPR/Cas9 genome editing in human stem cells",
            abstract="Genome editing of stem cells with CRISPR Cas9",
            year=2018,
            citations=327,
            author="Rasmus Bak",
        ),
    ]
    works += [
        work(
            f"WCRISPREXACT{index}",
            "CRISPR/Cas9 genome editing",
            abstract="Genome editing with CRISPR Cas9",
            year=year,
            citations=citations,
            author=f"Editor {index}",
        )
        for index, (year, citations) in enumerate(((2014, 5_000), (2019, 20), (2023, 3)))
    ]
    works += [
        work(
            f"WCRISPRTOPIC{index}",
            f"Genome editing with CRISPR Cas9 in species {index}",
            abstract=f"CRISPR Cas9 genome editing study {index}",
            year=2000 + index % 26,
            citations=index,
            author=f"Biologist {index}",
        )
        for index in range(30)
    ]
    # 150 titles sharing a common opening, stored least cited first: the old
    # prefix path kept the first 100 rows it found and lost the landmark.
    works += [
        work(
            f"WSTUDY{index}",
            f"A study of the effect of treatment {index} on outcomes",
            abstract=f"Outcomes after treatment {index}",
            year=2000 + index % 20,
            citations=index,
            author=f"Clinician {index}",
        )
        for index in range(149)
    ]
    works.append(
        work(
            "WSTUDYLANDMARK",
            "A study of the effect of aspirin on outcomes",
            abstract="Outcomes after aspirin",
            year=1990,
            citations=90_000,
            author="Clinician Landmark",
        )
    )
    # Semantic decoys: closer to the typed words than the paper itself, so the
    # 50-row candidate pool cannot rescue a title the lookup missed.
    works += [
        work(
            f"WRAGDECOY{index}",
            f"Retrieval augmented generation for knowledge intensive NLP tasks, report {index}",
            abstract="retrieval augmented generation knowledge intensive nlp tasks",
            year=2024,
            citations=index,
            author=f"Decoy {index}",
        )
        for index in range(60)
    ]
    works += [
        work(
            f"WBERTDECOY{index}",
            f"Pre-training of deep bidirectional transformers for language understanding {index}",
            abstract="bert pre-training deep bidirectional transformers language understanding",
            year=2023,
            citations=index,
            author=f"Decoy {index}",
        )
        for index in range(60)
    ]
    works += [
        work(f"WUNDATED{index}", f"Undated research paper {index}", year=0, citations=index)
        for index in range(5)
    ]
    # Title-only duplicate records with no author: same title, adjacent years.
    works += [
        work("WNOAUTHORA", "Proceedings of the tiny workshop", year=2011, citations=3),
        work("WNOAUTHORB", "Proceedings of the tiny workshop", year=2012, citations=9),
    ]
    return works


def filler(index: int) -> Paper:
    return work(
        f"W{9_000_000 + index}",
        f"Measured research paper {index}",
        abstract=f"A controlled study of topic {index % 17} and method {index % 31}",
        year=1990 + index % 35,
        citations=(index * 7_919) % 5_000,
        author=f"Researcher {index}",
        doi=f"https://doi.org/10.1000/{index}",
        topic=f"Topic {index % 17}",
        field="Medicine" if index % 3 == 0 else "Computer Science",
    )


def corpus() -> list[Paper]:
    specials = special_works()
    return [*(filler(index) for index in range(CORPUS_ROWS - len(specials))), *specials]


def build_production_shaped(
    root: Path,
    papers: list[Paper],
    *,
    candidate_count: int = CANDIDATE_POOL,
    backend: str = "numpy",
) -> Path:
    """Build a generation, then serve it from compact metadata like production."""
    generation = root / "generation"
    build_generation(
        iter(papers),
        stage=Stage("test", len(papers), 64, 4, candidate_count=candidate_count),
        output=generation,
        embedder=HashingEmbedder(),
        backend=backend,
        batch_size=512,
    )
    staging = root / "shards"
    staging.mkdir()
    shard = staging / "shard-00000.int8.npy"
    np.save(shard, np.zeros((len(papers), 2), dtype=np.int8))
    with gzip.open(staging / "shard-00000.meta.jsonl.gz", "wt", encoding="utf-8") as lines:
        for paper in papers:
            record = asdict(paper)
            record.pop("embedding_text")
            lines.write(json.dumps(record) + "\n")
    build_compact_metadata(
        [shard], generation, shard_rows=(len(papers),), total=len(papers), workers=1, block_rows=64
    )
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata_backend"] = "compact"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    build_title_prefix_index(
        generation,
        root / "prefixes",
        hasher=stable_text_hash,
        store_factory=CompactMetadataStore,
        batch_rows=1_000,
    )
    ids = [normalize_work_id(paper.openalex_id) for paper in papers]
    hashes = np.asarray([stable_text_hash(value) for value in ids], dtype=np.uint64)
    order = np.argsort(hashes, kind="stable")
    work_ids = root / "work-ids"
    work_ids.mkdir()
    np.save(work_ids / "work-id-hashes.npy", hashes[order])
    np.save(work_ids / "work-id-rows.npy", np.arange(len(ids), dtype=np.uint32)[order])
    (work_ids / "work-id-index.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "generation": generation.name,
                "records": len(papers),
                "ids": len(papers),
            }
        ),
        encoding="utf-8",
    )
    return generation


@pytest.fixture(scope="session")
def production_shaped(tmp_path_factory):
    """(generation, prefix index, work id index) for the 5,000-row corpus."""
    root = tmp_path_factory.mktemp("production-shaped")
    generation = build_production_shaped(root, corpus())
    return (
        generation,
        TitlePrefixIndex.load(root / "prefixes"),
        WorkIdIndex.load(root / "work-ids"),
    )
