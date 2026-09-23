#!/usr/bin/env python3
"""Synthetic checks for C1-v2 trace aggregation and partition guards."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_c1_v2_development import summarize_interface, validate_protocol


def event(offset: float, selected_outside: int) -> dict:
    return {
        "interface": {
            "actor_normalized_range": [-2.0 + offset, 2.0 + offset],
            "actor_denormalized_preclip": {
                "elements": 56,
                "outside_elements": 2,
                "clipped_element_ratio": 2 / 56,
                "min": -1.2,
                "max": 1.1,
            },
            "candidate_preclip": {
                "elements": 168,
                "outside_elements": 5,
                "clipped_element_ratio": 5 / 168,
                "min": -1.3,
                "max": 1.2,
            },
            "raw_candidate_range": [-1.0, 1.0],
            "wm_normalized_candidate_range": [-3.0 + offset, 2.5 + offset],
        },
        "selected_candidate_preclip": {
            "elements": 56,
            "outside_elements": selected_outside,
            "clipped_element_ratio": selected_outside / 56,
            "min": -1.1,
            "max": 1.05,
        },
    }


def main() -> None:
    protocol, partition = validate_protocol(ROOT / "configs/c1_v2_rebase_protocol.json")
    assert partition["init_state_indices"] == list(range(30, 40))
    assert set(partition["init_state_indices"]).isdisjoint(range(40, 50))
    summary = summarize_interface([event(0.0, 1), event(0.5, 3)])
    assert summary["actor_normalized_range"] == [-2.0, 2.5]
    assert summary["actor_denormalized_preclip"]["elements"] == 112
    assert summary["actor_denormalized_preclip"]["outside_elements"] == 4
    assert summary["all_candidate_preclip"]["elements"] == 336
    assert summary["selected_candidate_preclip"]["outside_elements"] == 4
    assert summary["selected_candidate_preclip"]["clipped_element_ratio"] == 4 / 112
    print(
        {
            "status": "PASS",
            "protocol": protocol["protocol_version"],
            "selected_clipping_ratio": summary["selected_candidate_preclip"][
                "clipped_element_ratio"
            ],
        }
    )


if __name__ == "__main__":
    main()
