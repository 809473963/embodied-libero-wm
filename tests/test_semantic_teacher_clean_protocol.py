import json
from pathlib import Path

from src.probes.semantic_teacher_clean import compare_teacher, task_grounding_contract


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_boundary_and_gates():
    protocol = json.loads((ROOT / "configs/semantic_teacher_clean_protocol.json").read_text())
    assert protocol["status"] == "PREREGISTERED_BEFORE_ANY_REAL_DEMO_ARRAY_ACCESS"
    assert protocol["partitions"]["calibration"] == [31, 32, 33, 34]
    assert protocol["partitions"]["frozen_development_evaluation"] == [36, 37, 38, 39]
    assert protocol["partitions"]["forbidden_including_metadata"] == list(range(40, 50))
    assert protocol["gates"]["local_non_tie_min"] == 0.30
    assert protocol["gates"]["language_change_min"] == 0.20
    assert protocol["gates"]["severe_proprio_accuracy"] == 0.90


def test_all_source_contracts_reconstruct_exactly():
    saved = json.loads((ROOT / "configs/task_grounding_contract.json").read_text())
    assert len(saved["tasks"]) == 10
    for expected in saved["tasks"]:
        actual = task_grounding_contract(ROOT / expected["bddl_path"], expected["task_id"])
        assert actual["source_atom"] == expected["source_atom"]
        assert actual["exact_final_goal"] == ["On", "akita_black_bowl_1", "plate_1"]
        assert actual["bddl_sha256"] == expected["bddl_sha256"]


def test_t1_is_lexicographic_and_allows_ties_and_regression():
    kwargs = dict(family="T1", epsilon_goal=0.05, epsilon_reach=0.05)
    assert compare_teacher(False, -2.0, False, -1.0, True, -10.0, False, -10.0, **kwargs) == 1
    assert compare_teacher(False, -2.0, False, -1.0, False, -1.0, False, -1.0, **kwargs) == 1
    assert compare_teacher(False, -1.0, False, -1.0, False, -2.0, False, -1.0, **kwargs) == -1
    assert compare_teacher(False, -1.0, False, -1.0, False, -0.97, False, -1.0, **kwargs) == 0


def test_t2_stage_order_is_fixed_without_weighted_sum():
    kwargs = dict(family="T2", epsilon_goal=0.05, epsilon_reach=0.05)
    assert compare_teacher(False, -1.0, False, -0.1, False, -9.0, True, -9.0, **kwargs) == 1
    assert compare_teacher(False, -1.0, True, -9.0, False, -0.5, True, -9.0, **kwargs) == 1
    assert compare_teacher(False, -1.0, False, -2.0, False, -1.0, False, -1.0, **kwargs) == 1


def test_runner_contains_no_hdf5_discovery_api():
    source = (ROOT / "scripts/run_semantic_teacher_clean.py").read_text()
    for forbidden in (".keys()", ".visit(", ".visititems(", "list(hdf5.keys())"):
        assert forbidden not in source
    assert "AllowedDemoGuard" in source

