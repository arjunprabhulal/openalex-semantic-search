from __future__ import annotations

import numpy as np

INT8_LEVELS = 127.0


def fit_int8_scales(vectors: np.ndarray, *, quantile: float = 1.0) -> np.ndarray:
    """Fit one symmetric scale per dimension from a representative calibration set."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("vectors must have shape (records, dimensions)")
    if not 0.9 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0.9 and 1.0")
    maxima = np.quantile(np.abs(values), quantile, axis=0).astype(np.float32)
    maxima[maxima < np.finfo(np.float32).eps] = 1.0
    return maxima / INT8_LEVELS


def quantize_normalized(vectors: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Symmetrically quantize normalized vectors with calibrated dimension scales."""
    values = np.asarray(vectors, dtype=np.float32)
    scale_values = np.asarray(scales, dtype=np.float32)
    return np.rint(np.clip(values / scale_values, -INT8_LEVELS, INT8_LEVELS)).astype(np.int8)


def int8_scores(query: np.ndarray, codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Approximate cosine scores after dequantizing only the candidate vectors."""
    query_vector = np.asarray(query, dtype=np.float32)
    candidate_codes = np.asarray(codes, dtype=np.int8)
    scale_values = np.asarray(scales, dtype=np.float32)
    dequantized = candidate_codes.astype(np.float32)
    dequantized *= scale_values
    numerators = dequantized @ query_vector
    denominators = np.linalg.norm(dequantized, axis=1) * max(
        float(np.linalg.norm(query_vector)),
        np.finfo(np.float32).eps,
    )
    return np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=denominators > 0,
    )
