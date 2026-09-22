"""C1: frozen C0 actor plus D15 world-model pre-execution reranking."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.control import generate_antithetic_candidates, score_candidates
from src.policies.c0_actor_policy import C0ActorPolicy
from src.world_models import LatentDynamicsEnsemble, LatentDynamicsModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class C1RerankPolicy(C0ActorPolicy):
    """Generate N=3 chunks and select one before execution; never replan early."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        config: str | Path = PROJECT_ROOT / "configs/c1_rerank.yaml",
    ) -> None:
        super().__init__(checkpoint=checkpoint, device=device)
        self.config_path = Path(config)
        self.config = yaml.safe_load(self.config_path.read_text())
        if self.config["contract_version"] != "c1-v1":
            raise ValueError("expected c1-v1 configuration")
        candidate_config = self.config["candidates"]
        if int(candidate_config["count"]) != 3 or candidate_config["base_candidate_index"] != 0:
            raise ValueError("C1-v1 freezes three candidates with base at index zero")
        if candidate_config["noise_std"][6] != 0.0:
            raise ValueError("C1-v1 candidate perturbations may not flip the gripper")
        checkpoint_path = Path(checkpoint)
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_hash != self.config["actor"]["checkpoint_sha256"]:
            raise ValueError("C1 actor checkpoint differs from the frozen C0 actor")
        self.candidate_seed = int(candidate_config["seed"])
        self.noise_std = torch.tensor(candidate_config["noise_std"], dtype=torch.float32)
        ensemble_config_path = PROJECT_ROOT / self.config["ensemble"]["config"]
        ensemble_config = yaml.safe_load(ensemble_config_path.read_text())
        members = []
        reference = None
        for relative_path in ensemble_config["member_checkpoints"]:
            state = torch.load(PROJECT_ROOT / relative_path, map_location="cpu", weights_only=False)
            comparable = {
                "contract": state["d13_data_contract_version"],
                "store": state["d13_store_sha256"],
                "normalization": state["normalization"],
                "model": state["config"]["model"],
            }
            reference = comparable if reference is None else reference
            if comparable != reference:
                raise ValueError("C1 ensemble members do not share the same contract")
            model = LatentDynamicsModel(
                **{key: int(value) for key, value in state["config"]["model"].items()}
            )
            model.load_state_dict(state["model_state"])
            model.eval()
            members.append(model)
        assert reference is not None
        self.ensemble = LatentDynamicsEnsemble(members).to(self.device).eval()
        for parameter in self.ensemble.parameters():
            parameter.requires_grad_(False)
        normalization = reference["normalization"]
        self.wm_proprio_mean = torch.tensor(
            normalization["proprio"]["mean"], dtype=torch.float32, device=self.device
        ).view(1, 12)
        self.wm_proprio_std = torch.tensor(
            normalization["proprio"]["std"], dtype=torch.float32, device=self.device
        ).view(1, 12)
        self.wm_action_mean = torch.tensor(
            normalization["action"]["mean"], dtype=torch.float32, device=self.device
        ).view(1, 1, 7)
        self.wm_action_std = torch.tensor(
            normalization["action"]["std"], dtype=torch.float32, device=self.device
        ).view(1, 1, 7)
        self._rollout_seed = 0
        self._plan_index = 0
        self._last_plan_diagnostics: dict[str, Any] | None = None
        self._last_selected_prediction: torch.Tensor | None = None
        self._last_selected_uncertainty: torch.Tensor | None = None

    def reset(self, seed: int | None = None) -> None:
        super().reset()
        self._rollout_seed = 0 if seed is None else int(seed)
        self._plan_index = 0
        self._last_plan_diagnostics = None
        self._last_selected_prediction = None
        self._last_selected_uncertainty = None

    @staticmethod
    def _proprio(observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
                np.asarray(observation["robot0_joint_pos"], dtype=np.float32).reshape(7),
            )
        )

    def plan_chunk(
        self, observation: dict[str, np.ndarray], language: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        required = ("agentview_image", "robot0_eef_pos", "robot0_gripper_qpos", "robot0_joint_pos")
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(f"C1 observation is missing {missing}")
        image = np.asarray(observation["agentview_image"], dtype=np.uint8)
        if image.shape != (128, 128, 3):
            raise ValueError(f"expected agentview_image [128,128,3], got {image.shape}")
        pixels = self.processor(images=[np.ascontiguousarray(image)], return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        proprio = torch.from_numpy(self._proprio(observation)).view(1, 12).to(self.device)
        language_feature = self._language_features(language)
        with torch.inference_mode(), self._autocast():
            z0 = self.clip.get_image_features(pixel_values=pixels)
            z0 = z0 / z0.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            actor_output = self.model(
                z0.float(),
                language_feature,
                (proprio - self.proprio_mean) / self.proprio_std,
            )
            raw_base = actor_output * self.action_std + self.action_mean
        base_chunk = raw_base[0].float().clamp(-1.0, 1.0)
        plan_seed = self.candidate_seed + self._rollout_seed * 1_000_003 + self._plan_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(plan_seed)
        candidates = generate_antithetic_candidates(base_chunk, self.noise_std, generator)
        with torch.inference_mode():
            normalized_proprio = (proprio - self.wm_proprio_mean) / self.wm_proprio_std
            normalized_actions = (candidates - self.wm_action_mean) / self.wm_action_std
            repeated_z0 = z0.float().expand(3, -1)
            repeated_proprio = normalized_proprio.expand(3, -1)
            repeated_language = language_feature.expand(3, -1)
            mean_prediction, uncertainty, _ = self.ensemble.predict(
                repeated_z0, repeated_proprio, repeated_language, normalized_actions
            )
            rerank = self.config["rerank"]
            components = score_candidates(
                z0.float(),
                language_feature,
                candidates,
                mean_prediction,
                uncertainty,
                self.wm_action_std,
                goal_progress_weight=float(rerank["goal_progress_weight"]),
                uncertainty_penalty_weight=float(rerank["uncertainty_penalty_weight"]),
                base_deviation_penalty_weight=float(rerank["base_deviation_penalty_weight"]),
            )
        selected = int(torch.argmax(components["score"]).item())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        diagnostics: dict[str, Any] = {
            "plan_index": self._plan_index,
            "candidate_seed": plan_seed,
            "selected_candidate": selected,
            "scores": components["score"].float().cpu().tolist(),
            "goal_progress": components["goal_progress"].float().cpu().tolist(),
            "mean_uncertainty": components["mean_uncertainty"].float().cpu().tolist(),
            "base_deviation": components["base_deviation"].float().cpu().tolist(),
            "final_language_similarity": components["final_language_similarity"].float().cpu().tolist(),
            "raw_base_clipping": self._clipping_statistics(raw_base.float().cpu().numpy())[0],
            "planning_ms": elapsed_ms,
            "executed_full_chunk": True,
            "early_replan": False,
        }
        if not all(
            np.isfinite(np.asarray(diagnostics[key], dtype=np.float64)).all()
            for key in ("scores", "goal_progress", "mean_uncertainty", "base_deviation")
        ):
            raise FloatingPointError("C1 planning diagnostics contain NaN/Inf")
        self._plan_index += 1
        self._last_plan_diagnostics = diagnostics
        # C1 never consumes these tensors. C2 uses this protected plan-time
        # state to compare the selected candidate with later real observations.
        self._last_selected_prediction = mean_prediction[selected].detach().clone()
        self._last_selected_uncertainty = uncertainty[selected].detach().clone()
        return candidates[selected].float().cpu().numpy(), diagnostics

    def _plan(self, observation: dict[str, np.ndarray], language: str) -> np.ndarray:
        chunk, _ = self.plan_chunk(observation, language)
        return chunk


def create_policy(*, checkpoint: str | None, device: str) -> C1RerankPolicy:
    if checkpoint is None:
        checkpoint = str(PROJECT_ROOT / "results/c0/seed11/best.pt")
    return C1RerankPolicy(checkpoint=checkpoint, device=device)
