#!/usr/bin/env python3
"""Synthetic checks for closed-form WM-v2B.1 calibration."""

from __future__ import annotations

import numpy as np

from scripts.analyze_wm_v2b_1_differential_calibration import block_bootstrap
from scripts.fit_wm_v2b_1_differential_calibration import (
    GROUPS,
    apply_form,
    fit_all,
    fit_scalar,
)


def test_closed_form_scalar_recovers_known_scale_and_keeps_negative_alpha() -> None:
    predicted = np.arange(1, 49, dtype=np.float64).reshape(2, 2, 3, 4)
    assert np.isclose(fit_scalar(predicted, 0.5 * predicted), 0.5)
    assert np.isclose(fit_scalar(predicted, -2.0 * predicted), -2.0)


def test_all_preregistered_forms_fit_expected_axes() -> None:
    rng = np.random.default_rng(7)
    predicted = rng.normal(size=(5, 3, 8, 12))
    horizon_scale = np.linspace(0.4, 1.1, 8)
    real_horizon = predicted * horizon_scale[None, None, :, None]
    fitted_horizon = fit_all(predicted, real_horizon)["per_horizon_scalar"]
    assert np.allclose(fitted_horizon, horizon_scale, atol=1e-12, rtol=0)
    assert np.allclose(
        apply_form(predicted, "per_horizon_scalar", fitted_horizon),
        real_horizon,
        atol=1e-12,
        rtol=0,
    )

    group_scale = {"ee_pos": 0.5, "gripper": 1.5, "joints": 0.8}
    real_group = predicted.copy()
    for group, indices in GROUPS.items():
        real_group[..., indices] *= group_scale[group]
    fitted_group = fit_all(predicted, real_group)["groupwise_scalar"]
    assert all(
        np.isclose(fitted_group[group], group_scale[group], atol=1e-12)
        for group in GROUPS
    )
    assert np.allclose(
        apply_form(predicted, "groupwise_scalar", fitted_group),
        real_group,
        atol=1e-12,
        rtol=0,
    )


def test_branch_block_bootstrap_rewards_held_out_calibration() -> None:
    rng = np.random.default_rng(11)
    real = rng.normal(size=(12, 3, 8, 12))
    raw = 2.0 * real
    calibrated = real.copy()
    within = np.ones((12, 8), dtype=np.float64)
    ordering = np.ones((12, 8, 3), dtype=np.float64)
    result = block_bootstrap(
        calibrated,
        raw,
        real,
        within,
        ordering,
        sign_threshold=1e-3,
        replicates=300,
        seed=3,
        minimum_real_norm=1e-4,
    )
    assert result["improvement_zero_ci95"][0] > 0.99
    assert result["improvement_raw_ci95"][0] > 0.99
    assert result["direction_mean_ci95"][0] > 0.99
    assert result["within_context_spearman_ci95"] == [1.0, 1.0]
    assert result["ordering_accuracy_ci95"] == [1.0, 1.0]
    assert result["sign_accuracy_ci95"] == [1.0, 1.0]
