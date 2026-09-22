#!/usr/bin/env python3
"""D20 grid metrics and deterministic selection-rule tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.select_mismatch_threshold import choose_grid_row, evaluate_pair


def main() -> int:
    mismatch = np.arange(100, dtype=np.float64) / 100.0
    uncertainty = mismatch[::-1].copy()
    metrics = evaluate_pair(mismatch, uncertainty, 0.94, 0.94)
    assert metrics["check_count"] == 100
    assert metrics["trigger_count"] == 10
    assert abs(metrics["nominal_or_trigger_rate"] - 0.10) < 1e-12
    rows = [
        {"mismatch_quantile": 0.99, "uncertainty_quantile": 0.99, "nominal_or_trigger_rate": 0.009},
        {"mismatch_quantile": 0.995, "uncertainty_quantile": 0.995, "nominal_or_trigger_rate": 0.008},
        {"mismatch_quantile": 0.999, "uncertainty_quantile": 0.999, "nominal_or_trigger_rate": 0.001},
        {"mismatch_quantile": 0.95, "uncertainty_quantile": 0.95, "nominal_or_trigger_rate": 0.05},
    ]
    selected = choose_grid_row(rows, target_rate=0.01)
    assert selected == rows[0]
    print({"metrics": metrics, "selected": selected})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
