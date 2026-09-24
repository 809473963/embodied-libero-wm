#!/usr/bin/env python3
"""Synthetic checks for local counterfactual metric construction."""

from __future__ import annotations

import numpy as np

from scripts.audit_local_candidate_counterfactual import (
    correlation,
    cosine_distance,
    pairwise,
    safe_ratio,
    summary,
)


def test_pairwise_cosine_and_l2_shapes() -> None:
    values = np.zeros((2, 3, 8, 4), dtype=np.float32)
    values[:, 0, :, 0] = 1.0
    values[:, 1, :, 1] = 1.0
    values[:, 2, :, 0] = 1.0
    cosine = pairwise(values, "cosine")
    l2 = pairwise(values, "l2")
    assert cosine.shape == (2, 3, 8)
    assert l2.shape == (2, 3, 8)
    assert np.allclose(cosine[:, 0], 1.0)
    assert np.allclose(cosine[:, 1], 0.0)
    assert np.allclose(l2[:, 0], np.sqrt(2.0))
    assert np.isfinite(cosine).all()


def test_candidate_specific_error_and_near_zero_ratio_rule() -> None:
    target = np.zeros((1, 3, 8, 4), dtype=np.float32)
    prediction = np.zeros_like(target)
    target[..., 0] = 1.0
    prediction[..., 0] = 1.0
    error = cosine_distance(prediction, target)
    assert error.shape == (1, 3, 8)
    assert np.allclose(error, 0.0)

    blocked = safe_ratio(np.full(8, 2e-6), np.full(8, 9e-6), 1e-5)
    assert blocked["eligible"] is False
    assert blocked["ratio_of_means"] is None
    assert np.isclose(blocked["predicted_minus_real"], -7e-6)

    eligible = safe_ratio(np.full(8, 2e-5), np.full(8, 4e-5), 1e-5)
    assert eligible["eligible"] is True
    assert np.isclose(eligible["ratio_of_means"], 0.5)


def test_summary_and_correlation_preserve_grain() -> None:
    x = np.arange(1, 25, dtype=np.float64).reshape(3, 8)
    y = 2.0 * x + 1.0
    stats = summary(x)
    relation = correlation(x, y)
    assert stats["n"] == 24
    assert stats["mean"] == 12.5
    assert relation["n"] == 24
    assert np.isclose(relation["pearson"], 1.0)
    assert np.isclose(relation["spearman"], 1.0)

