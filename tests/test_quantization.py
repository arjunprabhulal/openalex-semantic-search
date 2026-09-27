import numpy as np

from openalex_semantic_search.embeddings import normalize_rows
from openalex_semantic_search.quantization import fit_int8_scales, int8_scores, quantize_normalized


def test_int8_refinement_preserves_nearest_neighbors():
    rng = np.random.default_rng(42)
    vectors = normalize_rows(rng.normal(size=(2_000, 384)).astype(np.float32))
    query = normalize_rows(rng.normal(size=(1, 384)).astype(np.float32))[0]
    exact = np.argsort(vectors @ query)[::-1][:20]
    scales = fit_int8_scales(vectors)
    approximate_scores = int8_scores(
        query,
        quantize_normalized(vectors, scales),
        scales,
    )
    approximate = np.argsort(approximate_scores)[::-1][:20]
    assert len(set(exact).intersection(approximate)) / 20 >= 0.95


def test_quantized_vectors_are_signed_bytes():
    values = np.array([[-1.2, -0.5, 0.0, 0.5, 1.2]], dtype=np.float32)
    scales = np.full(values.shape[1], 1 / 127, dtype=np.float32)
    codes = quantize_normalized(values, scales)
    assert codes.dtype == np.int8
    assert codes.tolist() == [[-127, -64, 0, 64, 127]]
