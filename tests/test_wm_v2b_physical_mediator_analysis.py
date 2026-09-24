#!/usr/bin/env python3
"""Synthetic metric checks for the physical-mediator audit."""

from __future__ import annotations

import numpy as np

from scripts.audit_wm_v2b_physical_mediator import (
    absolute_from_smooth_l1,
    bootstrap_metrics,
    differential_stats,
    pairwise_delta,
    rank_contexts,
    sign_fidelity,
    smooth_l1_from_absolute,
)


def test_smooth_l1_inversion_and_pairwise_delta() -> None:
    absolute = np.asarray([0.0, 0.25, 0.999, 1.0, 2.5], dtype=np.float64)
    recovered = absolute_from_smooth_l1(smooth_l1_from_absolute(absolute))
    assert np.allclose(recovered, absolute, atol=1e-12, rtol=0)

    values = np.zeros((2, 3, 8, 12), dtype=np.float64)
    values[:, 1] = 1.0
    values[:, 2] = 3.0
    delta = pairwise_delta(values)
    assert delta.shape == (2, 3, 8, 12)
    assert np.all(delta[:, 0] == -1.0)
    assert np.all(delta[:, 1] == -3.0)
    assert np.all(delta[:, 2] == -2.0)


def test_differential_stats_distinguish_perfect_from_zero() -> None:
    real = np.ones((4, 3, 8, 12), dtype=np.float64)
    perfect = differential_stats(real, real, 1e-4)
    assert np.isclose(perfect["direction_cosine"]["mean"], 1.0)
    assert np.isclose(perfect["magnitude_ratio"]["ratio_of_means"], 1.0)
    assert np.isclose(perfect["differential_rmse"], 0.0)
    assert np.isclose(perfect["relative_improvement_over_zero"], 1.0)

    zero = differential_stats(np.zeros_like(real), real, 1e-4)
    assert np.isclose(zero["direction_cosine"]["mean"], 0.0)
    assert np.isclose(zero["magnitude_ratio"]["ratio_of_means"], 0.0)
    assert np.isclose(zero["differential_rmse"], zero["zero_difference_rmse"])
    assert np.isclose(zero["relative_improvement_over_zero"], 0.0)


def test_ranking_sign_and_block_bootstrap() -> None:
    branches = 5
    real_norm = np.zeros((branches, 3, 8), dtype=np.float64)
    real_norm[:, 0] = 1.0
    real_norm[:, 1] = 2.0
    real_norm[:, 2] = 3.0
    predicted_norm = real_norm * 1.5
    within, ordering = rank_contexts(predicted_norm, real_norm, 1e-8)
    assert np.allclose(within, 1.0)
    assert np.allclose(ordering, 1.0)

    real = np.ones((branches, 3, 8, 12), dtype=np.float64)
    predicted = 1.5 * real
    task = np.arange(branches, dtype=np.int64)
    sign = sign_fidelity(predicted, real, 1e-3, task)
    assert sign["overall"]["accuracy"] == 1.0
    assert all(row["accuracy"] == 1.0 for row in sign["dimensions"].values())

    boot = bootstrap_metrics(
        predicted,
        real,
        within,
        ordering,
        replicates=200,
        seed=7,
        minimum_real_norm=1e-4,
    )
    assert boot["direction_mean_ci95"] == [1.0, 1.0]
    assert np.allclose(boot["within_context_spearman_ci95"], [1.0, 1.0])
    assert boot["pairwise_ordering_accuracy_ci95"] == [1.0, 1.0]
    assert boot["relative_improvement_ci95"][0] > 0
