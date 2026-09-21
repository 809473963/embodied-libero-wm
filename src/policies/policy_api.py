"""Minimal public contract for policies shown in the live viewer."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class LivePolicy(Protocol):
    """A deployed policy that maps allowed observations and language to action."""

    def act(self, observation: dict[str, np.ndarray], language: str) -> np.ndarray:
        """Return one normalized LIBERO action with shape ``(7,)``."""

    def reset(self) -> None:
        """Optionally clear episode-local state before a new rollout."""


PolicyFactory = Any
