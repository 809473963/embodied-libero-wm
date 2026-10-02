"""Exact-path HDF5 access guard for isolated demonstration partitions.

The public API never exposes an h5py file or group and offers no enumeration.
Every operation validates the requested demo ID before the file is opened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


class DemoAccessViolation(RuntimeError):
    """A caller attempted an operation outside the exact allowlist contract."""


@dataclass
class AccessLedger:
    partition: str
    allowed_demo_ids: tuple[int, ...]
    opened_task_ids: set[int] = field(default_factory=set)
    opened_demo_ids: set[int] = field(default_factory=set)
    opened_exact_groups: list[str] = field(default_factory=list)
    numeric_datasets_accessed: list[str] = field(default_factory=list)
    metadata_operations: list[str] = field(default_factory=list)
    forbidden_attempts: list[dict[str, Any]] = field(default_factory=list)

    def record_forbidden(self, operation: str, task_id: int | None, demo_id: int | None) -> None:
        self.forbidden_attempts.append(
            {"operation": operation, "task_id": task_id, "demo_id": demo_id}
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "allowed_demo_ids": list(self.allowed_demo_ids),
            "opened_task_ids": sorted(self.opened_task_ids),
            "opened_demo_ids": sorted(self.opened_demo_ids),
            "opened_exact_hdf5_groups": list(dict.fromkeys(self.opened_exact_groups)),
            "numeric_datasets_accessed": list(dict.fromkeys(self.numeric_datasets_accessed)),
            "metadata_operations": list(dict.fromkeys(self.metadata_operations)),
            "forbidden_attempts": self.forbidden_attempts,
            "forbidden_access_count": len(self.forbidden_attempts),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n")


class AllowedDemoGuard:
    """Read exact datasets for explicitly allowed demos without group discovery."""

    _PARTITIONS = {
        "calibration": (31, 32, 33, 34),
        "evaluation": (36, 37, 38, 39),
    }

    def __init__(
        self,
        partition: str,
        task_files: dict[int, Path],
        allowed_datasets: Iterable[str],
    ) -> None:
        if partition not in self._PARTITIONS:
            raise ValueError("partition must be calibration or evaluation")
        if set(task_files) != set(range(10)):
            raise ValueError("task_files must map exactly task0--9")
        self.partition = partition
        self.allowed_demo_ids = self._PARTITIONS[partition]
        self._task_files = {int(key): Path(value) for key, value in task_files.items()}
        self._allowed_datasets = frozenset(str(value).strip("/") for value in allowed_datasets)
        self.ledger = AccessLedger(partition, self.allowed_demo_ids)

    @staticmethod
    def exact_group(demo_id: int) -> str:
        return f"data/demo_{int(demo_id)}"

    def _deny(self, operation: str, task_id: int | None = None, demo_id: int | None = None) -> None:
        self.ledger.record_forbidden(operation, task_id, demo_id)
        raise DemoAccessViolation(
            f"{operation} is forbidden for task={task_id}, demo={demo_id}"
        )

    def _validate(self, task_id: int, demo_id: int, operation: str) -> Path:
        task_id, demo_id = int(task_id), int(demo_id)
        if task_id not in self._task_files:
            self._deny(operation, task_id, demo_id)
        if demo_id not in self.allowed_demo_ids:
            self._deny(operation, task_id, demo_id)
        return self._task_files[task_id]

    def _dataset_path(self, task_id: int, demo_id: int, dataset: str) -> tuple[Path, str]:
        file_path = self._validate(task_id, demo_id, "dataset_access")
        relative = str(dataset).strip("/")
        if relative not in self._allowed_datasets:
            self._deny(f"dataset_not_allowlisted:{relative}", task_id, demo_id)
        return file_path, f"{self.exact_group(demo_id)}/{relative}"

    def read_dataset(self, task_id: int, demo_id: int, dataset: str) -> np.ndarray:
        file_path, exact_path = self._dataset_path(task_id, demo_id, dataset)
        with h5py.File(file_path, "r") as source:
            value = source[exact_path][:]
        self.ledger.opened_task_ids.add(int(task_id))
        self.ledger.opened_demo_ids.add(int(demo_id))
        self.ledger.opened_exact_groups.append(f"{file_path}:{self.exact_group(demo_id)}")
        self.ledger.numeric_datasets_accessed.append(f"{file_path}:{exact_path}")
        return np.asarray(value)

    def dataset_metadata(self, task_id: int, demo_id: int, dataset: str) -> dict[str, Any]:
        file_path, exact_path = self._dataset_path(task_id, demo_id, dataset)
        with h5py.File(file_path, "r") as source:
            node = source[exact_path]
            result = {"shape": list(node.shape), "dtype": str(node.dtype)}
        self.ledger.opened_task_ids.add(int(task_id))
        self.ledger.opened_demo_ids.add(int(demo_id))
        self.ledger.opened_exact_groups.append(f"{file_path}:{self.exact_group(demo_id)}")
        self.ledger.metadata_operations.append(f"dataset_metadata:{file_path}:{exact_path}")
        return result

    def exact_group_exists(self, task_id: int, demo_id: int) -> bool:
        file_path = self._validate(task_id, demo_id, "existence_probe")
        exact_path = self.exact_group(demo_id)
        with h5py.File(file_path, "r") as source:
            result = exact_path in source
        self.ledger.metadata_operations.append(f"existence_probe:{file_path}:{exact_path}")
        return bool(result)

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

