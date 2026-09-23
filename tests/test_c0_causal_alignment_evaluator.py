#!/usr/bin/env python3
"""Unit checks for causal-alignment evaluator guards and summaries."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_c0_causal_alignment import (
    PROJECT_ROOT,
    build_result,
    require_committed_holdout_go,
    sha256,
)


def main() -> None:
    protocol_path = PROJECT_ROOT / "configs/c0_v2_causal_alignment_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checkpoint = PROJECT_ROOT / protocol["frozen_inputs"]["c0_v1_checkpoint"]
    episodes = [
        {
            "episode_index": 0,
            "seed": 30000,
            "init_state_index": 30,
            "success": True,
            "steps": 10,
            "action_clipping": {
                "planned_chunks": 2,
                "planned_action_elements": 112,
                "elements_below_minus_one": 1,
                "elements_above_one": 1,
                "elements_outside_minus_one_to_one": 2,
                "clipped_element_ratio": 2 / 112,
                "raw_min": -1.1,
                "raw_max": 1.1,
            },
        }
    ]
    task = {
        "task_id": 0,
        "task_name": "synthetic",
        "task_language": "synthetic",
        "episodes": 1,
        "successes": 1,
        "success_rate": 1.0,
        "mean_steps": 10.0,
        "episode_results": episodes,
        "action_clipping": {},
    }
    result = build_result(
        protocol_path,
        protocol,
        "controller_development",
        "c0_v1",
        checkpoint,
        sha256(checkpoint),
        1,
        [0],
        [task],
        True,
    )
    assert result["successes"] == 1
    assert result["mean_steps"] == 10.0
    assert result["nonzero_tasks"] == 1
    assert result["action_clipping"]["clipped_element_ratio"] == 2 / 112

    require_committed_holdout_go("controller_development", None, "p", "c")
    try:
        require_committed_holdout_go("confirmatory_holdout", None, "p", "c")
    except ValueError as error:
        assert "--gate-decision" in str(error)
    else:
        raise AssertionError("holdout ran without a committed gate")

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        untracked = Path(directory) / "decision.json"
        untracked.write_text(json.dumps({"decision": "GO_CONFIRMATORY_HOLDOUT"}))
        try:
            require_committed_holdout_go("confirmatory_holdout", untracked, "p", "c")
        except Exception:
            pass
        else:
            raise AssertionError("holdout accepted an uncommitted gate")
    print({"status": "PASS", "holdout_locked": True, "raw_episode_summary": True})


if __name__ == "__main__":
    main()
