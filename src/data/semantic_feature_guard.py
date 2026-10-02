"""Exact-path guard for the frozen D13 natural-feature store.

This is the feature-store analogue of :mod:`allowed_demo_guard`.  It never
enumerates HDF5 groups and exposes only the global frozen language matrix and
the ``vision`` dataset for one explicitly admitted task/demo pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data.allowed_demo_guard import AccessLedger, DemoAccessViolation


class AllowedDemoFeatureGuard:
    """Read only allowlisted natural features from one frozen D13 store."""

    _PARTITIONS = {
        "calibration": (31, 32, 33, 34),
        "evaluation": (36, 37, 38, 39),
    }
    _LANGUAGE_PATH = "task_metadata/language_features"

    def __init__(self, partition: str, store_path: str | Path) -> None:
        if partition not in self._PARTITIONS:
            raise ValueError("partition must be calibration or evaluation")
        self.partition = partition
        self.allowed_demo_ids = self._PARTITIONS[partition]
        self._store_path = Path(store_path)
        self.ledger = AccessLedger(partition, self.allowed_demo_ids)

    @staticmethod
    def exact_group(task_id: int, demo_id: int) -> str:
        return f"tasks/task_{int(task_id)}/demos/demo_{int(demo_id)}"

    def _deny(
        self, operation: str, task_id: int | None = None, demo_id: int | None = None
    ) -> None:
        self.ledger.record_forbidden(operation, task_id, demo_id)
        raise DemoAccessViolation(
            f"{operation} is forbidden for task={task_id}, demo={demo_id}"
        )

    def _validate(self, task_id: int, demo_id: int, operation: str) -> tuple[int, int]:
        task_id, demo_id = int(task_id), int(demo_id)
        if task_id not in range(10) or demo_id not in self.allowed_demo_ids:
            self._deny(operation, task_id, demo_id)
        return task_id, demo_id

    def read_vision(self, task_id: int, demo_id: int) -> np.ndarray:
        task_id, demo_id = self._validate(task_id, demo_id, "vision_access")
        group = self.exact_group(task_id, demo_id)
        exact_path = f"{group}/vision"
        with h5py.File(self._store_path, "r") as source:
            value = source[exact_path][:]
        self.ledger.opened_task_ids.add(task_id)
        self.ledger.opened_demo_ids.add(demo_id)
        self.ledger.opened_exact_groups.append(f"{self._store_path}:{group}")
        self.ledger.numeric_datasets_accessed.append(
            f"{self._store_path}:{exact_path}"
        )
        return np.asarray(value)

    def read_language(self) -> np.ndarray:
        with h5py.File(self._store_path, "r") as source:
            value = source[self._LANGUAGE_PATH][:]
        self.ledger.numeric_datasets_accessed.append(
            f"{self._store_path}:{self._LANGUAGE_PATH}"
        )
        return np.asarray(value)

    def dataset_metadata(
        self, task_id: int, demo_id: int, dataset: str = "vision"
    ) -> dict[str, Any]:
        task_id, demo_id = self._validate(task_id, demo_id, "metadata_access")
        if dataset != "vision":
            self._deny(f"dataset_not_allowlisted:{dataset}", task_id, demo_id)
        exact_path = f"{self.exact_group(task_id, demo_id)}/vision"
        with h5py.File(self._store_path, "r") as source:
            node = source[exact_path]
            result = {"shape": list(node.shape), "dtype": str(node.dtype)}
        self.ledger.metadata_operations.append(
            f"dataset_metadata:{self._store_path}:{exact_path}"
        )
        return result

    def list_demo_groups(self, *_: Any, **__: Any) -> None:
        self._deny("group_enumeration")

    def keys(self, *_: Any, **__: Any) -> None:
        self._deny("keys_enumeration")

    def visit(self, *_: Any, **__: Any) -> None:
        self._deny("recursive_visit")

    def visititems(self, *_: Any, **__: Any) -> None:
        self._deny("recursive_visititems")

    def assert_clean(self) -> None:
        if self.ledger.forbidden_attempts:
            raise DemoAccessViolation(
                f"access ledger contains {len(self.ledger.forbidden_attempts)} forbidden attempts"
            )
