"""Hard-boundary tests for exact-path demonstration access."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.allowed_demo_guard import AllowedDemoGuard, DemoAccessViolation


def synthetic_store(path: Path) -> None:
    with h5py.File(path, "w") as store:
        for demo_id in (31, 36, 40):
            group = store.create_group(f"data/demo_{demo_id}")
            group.create_dataset("states", data=np.full((2, 3), demo_id, dtype=np.float64))


def guard(path: Path, partition: str = "calibration") -> AllowedDemoGuard:
    return AllowedDemoGuard(partition, {task: path for task in range(10)}, ["states"])


def test_allowed_exact_group_and_dataset_access_passes(tmp_path: Path) -> None:
    path = tmp_path / "source.hdf5"
    synthetic_store(path)
    access = guard(path)
    assert access.exact_group_exists(0, 31)
    assert access.dataset_metadata(0, 31, "states") == {"shape": [2, 3], "dtype": "float64"}
    assert np.array_equal(access.read_dataset(0, 31, "states"), np.full((2, 3), 31.0))
    access.assert_clean()
    assert access.ledger.as_dict()["forbidden_access_count"] == 0


def test_forbidden_numeric_access_fails_before_file_open(tmp_path: Path) -> None:
    access = guard(tmp_path / "does-not-exist.hdf5")
    with pytest.raises(DemoAccessViolation):
        access.read_dataset(0, 40, "states")


def test_forbidden_metadata_access_fails_before_file_open(tmp_path: Path) -> None:
    access = guard(tmp_path / "does-not-exist.hdf5")
    with pytest.raises(DemoAccessViolation):
        access.dataset_metadata(0, 40, "states")


def test_forbidden_existence_probe_fails_before_file_open(tmp_path: Path) -> None:
    access = guard(tmp_path / "does-not-exist.hdf5")
    with pytest.raises(DemoAccessViolation):
        access.exact_group_exists(0, 40)


@pytest.mark.parametrize("method", ["list_demo_groups", "keys", "visit", "visititems"])
def test_group_enumeration_is_never_available(tmp_path: Path, method: str) -> None:
    access = guard(tmp_path / "does-not-exist.hdf5")
    with pytest.raises(DemoAccessViolation):
        getattr(access, method)()


def test_partition_allowlists_are_disjoint(tmp_path: Path) -> None:
    calibration = guard(tmp_path / "none.hdf5", "calibration")
    evaluation = guard(tmp_path / "none.hdf5", "evaluation")
    assert calibration.allowed_demo_ids == (31, 32, 33, 34)
    assert evaluation.allowed_demo_ids == (36, 37, 38, 39)
    assert set(calibration.allowed_demo_ids).isdisjoint(evaluation.allowed_demo_ids)

