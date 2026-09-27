import sys
import types

import pytest

from openalex_semantic_search.embeddings import SentenceTransformerEmbedder


def test_sentence_transformer_uses_explicit_writable_cache(monkeypatch, tmp_path):
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured.update(kwargs)

        @staticmethod
        def get_sentence_embedding_dimension():
            return 384

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    cache = tmp_path / "models"

    SentenceTransformerEmbedder(cache_folder=cache)

    assert cache.is_dir()
    assert captured["cache_folder"] == str(cache)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workers": 0}, "workers must be positive"),
        ({"tokenizer_processes": -1}, "tokenizer_processes must not be negative"),
        ({"tokenizer_threads": 0}, "tokenizer_threads must be positive"),
        (
            {"workers": 2, "tokenizer_processes": 2},
            "embedding workers and tokenizer processes cannot be enabled together",
        ),
    ],
)
def test_sentence_transformer_rejects_invalid_parallelism(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SentenceTransformerEmbedder(**kwargs)
