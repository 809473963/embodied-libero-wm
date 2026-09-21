"""Non-learned policy solely for verifying the live-viewer integration path."""

from __future__ import annotations

import numpy as np


class ZeroPolicy:
    def reset(self) -> None:
        pass

    def act(self, observation: dict[str, np.ndarray], language: str) -> np.ndarray:
        del observation, language
        return np.zeros(7, dtype=np.float32)


def create_policy(checkpoint: str | None, device: str) -> ZeroPolicy:
    """Factory shape required by ``scripts/run_policy_viewer.py``."""

    del checkpoint, device
    return ZeroPolicy()
