"""C2 extension: verify selected C1 predictions every two executed actions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.control import verification_decision
from src.policies.c1_rerank_policy import C1RerankPolicy, PROJECT_ROOT


class C2VerifyReplanPolicy(C1RerankPolicy):
    """Expose plan-time predictions and frozen-threshold verification decisions."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        config: str | Path = PROJECT_ROOT / "configs/c2_verify_replan.yaml",
    ) -> None:
        self.c2_config_path = Path(config)
        self.c2_config = yaml.safe_load(self.c2_config_path.read_text())
        if self.c2_config["contract_version"] != "c2-v1":
            raise ValueError("expected c2-v1 configuration")
        c1_config = PROJECT_ROOT / self.c2_config["c1_config"]
        super().__init__(checkpoint=checkpoint, device=device, config=c1_config)
        threshold_path = PROJECT_ROOT / self.c2_config["threshold_selection"]
        threshold_hash = hashlib.sha256(threshold_path.read_bytes()).hexdigest()
        if threshold_hash != self.c2_config["threshold_selection_sha256"]:
            raise ValueError("C2 threshold selection hash differs from frozen D20")
        threshold_result = yaml.safe_load(threshold_path.read_text())
        if threshold_result["status"] != "PASS" or not threshold_result["d21_authorized"]:
            raise ValueError("D20 did not authorize D21")
        selected = threshold_result["selected"]
        verification = self.c2_config["verification"]
        if float(verification["mismatch_threshold"]) != float(selected["mismatch_threshold"]):
            raise ValueError("C2 mismatch threshold differs from D20")
        if float(verification["uncertainty_threshold"]) != float(
            selected["uncertainty_threshold"]
        ):
            raise ValueError("C2 uncertainty threshold differs from D20")
        if verification["horizons"] != [2, 4, 6, 8]:
            raise ValueError("C2-v1 freezes verification at h2/h4/h6/h8")
        self.mismatch_threshold = float(selected["mismatch_threshold"])
        self.uncertainty_threshold = float(selected["uncertainty_threshold"])

    def _encode_actual_image(self, observation: dict[str, np.ndarray]) -> torch.Tensor:
        if "agentview_image" not in observation:
            raise KeyError("C2 verification requires current agentview_image")
        image = np.asarray(observation["agentview_image"], dtype=np.uint8)
        if image.shape != (128, 128, 3):
            raise ValueError(f"expected agentview_image [128,128,3], got {image.shape}")
        pixels = self.processor(images=[np.ascontiguousarray(image)], return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        with torch.inference_mode(), self._autocast():
            actual = self.clip.get_image_features(pixel_values=pixels)
            actual = actual / actual.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return actual.float()

    def verify(
        self,
        observation: dict[str, np.ndarray],
        horizon: int,
        remaining_actions: int,
    ) -> dict[str, Any]:
        if horizon not in {2, 4, 6, 8}:
            raise ValueError("C2 verification horizon must be one of 2,4,6,8")
        if self._last_selected_prediction is None or self._last_selected_uncertainty is None:
            raise RuntimeError("C2 verify called without a selected plan prediction")
        actual = self._encode_actual_image(observation)
        result = verification_decision(
            self._last_selected_prediction[horizon - 1 : horizon],
            actual,
            float(self._last_selected_uncertainty[horizon - 1]),
            self.mismatch_threshold,
            self.uncertainty_threshold,
            remaining_actions,
        )
        return {"horizon": horizon, **result}


def create_policy(*, checkpoint: str | None, device: str) -> C2VerifyReplanPolicy:
    if checkpoint is None:
        checkpoint = str(PROJECT_ROOT / "results/c0/seed11/best.pt")
    return C2VerifyReplanPolicy(checkpoint=checkpoint, device=device)
