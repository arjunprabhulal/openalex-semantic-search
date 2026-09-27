import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi import Response

from openalex_semantic_search.api import create_app


def test_index_progress_endpoint_is_not_cached(tmp_path):
    (tmp_path / "shard-00000.done").write_text(
        json.dumps(
            {
                "records": 1_000_000,
                "records_done": 1_000_000,
                "file_index": 4,
            }
        ),
        encoding="utf-8",
    )
    engine = SimpleNamespace(
        record_count=10_000,
        manifest={"stage": "test", "backend": "numpy"},
    )
    app = create_app(
        api_keys=frozenset({"test-key"}),
        engine=engine,
        progress_artifacts=tmp_path,
    )

    # Exercise the route function without FastAPI's optional httpx2 test client.
    app.state.engine = engine
    route = next(route for route in app.routes if route.path == "/index-progress")
    response = Response()
    body = route.endpoint(response)

    assert response.headers["cache-control"] == "no-store"
    assert body["indexed_records"] == 10_000
    assert body["verified_embedding_records"] == 1_000_000
