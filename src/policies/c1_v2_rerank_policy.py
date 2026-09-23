"""C1-v2: frozen C0-v2 actor plus frozen D15 pre-execution reranking."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.control import prepare_candidate_action_interface, score_candidates
from src.policies.c0_actor_policy import C0ActorPolicy
from src.world_models import LatentDynamicsEnsemble, LatentDynamicsModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class C1V2RerankPolicy(C0ActorPolicy):
    """Select one of three causal C0-v2 chunks and execute it open loop."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        config: str | Path = PROJECT_ROOT / "configs/c1_v2_rerank.yaml",
    ) -> None:
        checkpoint_path = Path(checkpoint)
        config_path = Path(config)
        configuration = yaml.safe_load(config_path.read_text())
        if configuration["contract_version"] != "c1-v2":
            raise ValueError("expected c1-v2 configuration")
        if file_sha256(checkpoint_path) != configuration["actor"]["checkpoint_sha256"]:
            raise ValueError("C1-v2 actor differs from frozen C0-v2 checkpoint")
        actor_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_contract = {
            "version": "c0-v2-causal-alignment-v1",
            "pairing": "obs[k] -> actions[k+1:k+9]",
            "action_start_offset": 1,
            "horizon": 8,
        }
        if actor_state.get("supervision_contract") != expected_contract:
            raise ValueError("C1-v2 requires the frozen causal C0-v2 actor contract")
        super().__init__(checkpoint=checkpoint_path, device=device)

        self.config_path = config_path
        self.config = configuration
        candidate_config = configuration["candidates"]
        if int(candidate_config["count"]) != 3 or int(
            candidate_config["base_candidate_index"]
        ) != 0:
            raise ValueError("C1-v2 freezes N=3 with actor default at index zero")
        if float(candidate_config["noise_std"][6]) != 0.0:
            raise ValueError("C1-v2 perturbations may not flip the gripper")
        ensemble_config_path = PROJECT_ROOT / configuration["ensemble"]["config"]
        if file_sha256(ensemble_config_path) != configuration["ensemble"]["config_sha256"]:
            raise ValueError("C1-v2 ensemble configuration hash differs")
        ensemble_config = yaml.safe_load(ensemble_config_path.read_text())
        members = []
        reference: dict[str, Any] | None = None
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
                raise ValueError("C1-v2 ensemble members do not share the same contract")
            model = LatentDynamicsModel(
                **{key: int(value) for key, value in state["config"]["model"].items()}
            )
            model.load_state_dict(state["model_state"])
            model.eval()
            members.append(model)
        if reference is None:
            raise ValueError("C1-v2 ensemble is empty")
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
        self.candidate_seed = int(candidate_config["seed"])
        self.noise_std = torch.tensor(candidate_config["noise_std"], dtype=torch.float32)
        self._rollout_seed = 0
        self._plan_index = 0
        self._last_plan_tensors: dict[str, np.ndarray] | None = None

    def reset(self, seed: int | None = None) -> None:
        super().reset()
        self._rollout_seed = 0 if seed is None else int(seed)
        self._plan_index = 0
        self._last_plan_tensors = None

    @staticmethod
    def _proprio(observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
                np.asarray(observation["robot0_joint_pos"], dtype=np.float32).reshape(7),
            )
        )

    def _actor_context(
        self, observation: dict[str, np.ndarray], language: str
    ) -> dict[str, torch.Tensor]:
        required = (
            "agentview_image",
            "robot0_eef_pos",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(f"C1-v2 observation is missing {missing}")
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
            actor_normalized = self.model(
                z0.float(),
                language_feature,
                (proprio - self.proprio_mean) / self.proprio_std,
            )
        return {
            "z0": z0.float(),
            "proprio": proprio.float(),
            "language": language_feature.float(),
            "actor_normalized": actor_normalized[0].float(),
        }

    def encode_observation(self, observation: dict[str, np.ndarray]) -> torch.Tensor:
        image = np.asarray(observation["agentview_image"], dtype=np.uint8)
        if image.shape != (128, 128, 3):
            raise ValueError(f"expected agentview_image [128,128,3], got {image.shape}")
        pixels = self.processor(images=[np.ascontiguousarray(image)], return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        with torch.inference_mode(), self._autocast():
            latent = self.clip.get_image_features(pixel_values=pixels)
            latent = latent / latent.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return latent.float()

    def _interface(
        self, actor_normalized: torch.Tensor, plan_seed: int
    ) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(plan_seed)
        return prepare_candidate_action_interface(
            actor_normalized,
            self.action_mean,
            self.action_std,
            self.wm_action_mean,
            self.wm_action_std,
            self.noise_std,
            generator,
        )

    def _predict(
        self,
        context: dict[str, torch.Tensor],
        wm_normalized_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_count = wm_normalized_candidates.shape[0]
        normalized_proprio = (
            context["proprio"] - self.wm_proprio_mean
        ) / self.wm_proprio_std
        with torch.inference_mode():
            mean_prediction, uncertainty, _ = self.ensemble.predict(
                context["z0"].expand(candidate_count, -1),
                normalized_proprio.expand(candidate_count, -1),
                context["language"].expand(candidate_count, -1),
                wm_normalized_candidates,
            )
        return mean_prediction.float(), uncertainty.float()

    @staticmethod
    def _outside_summary(values: torch.Tensor) -> dict[str, float | int]:
        outside = (values < -1.0) | (values > 1.0)
        return {
            "elements": int(values.numel()),
            "outside_elements": int(outside.sum().item()),
            "clipped_element_ratio": float(outside.float().mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    def _interface_diagnostics(
        self, interface: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        return {
            "actor_normalized_range": [
                float(interface["actor_normalized"].min().item()),
                float(interface["actor_normalized"].max().item()),
            ],
            "actor_denormalized_preclip": self._outside_summary(
                interface["actor_denormalized_preclip"]
            ),
            "candidate_preclip": self._outside_summary(interface["candidate_preclip"]),
            "raw_candidate_range": [
                float(interface["candidates"].min().item()),
                float(interface["candidates"].max().item()),
            ],
            "wm_normalized_candidate_range": [
                float(interface["wm_normalized_candidates"].min().item()),
                float(interface["wm_normalized_candidates"].max().item()),
            ],
        }

    def _retain_plan_tensors(
        self,
        interface: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> None:
        self._last_plan_tensors = {
            "actor_normalized": interface["actor_normalized"].detach().cpu().numpy(),
            "actor_denormalized_preclip": interface[
                "actor_denormalized_preclip"
            ].detach().cpu().numpy(),
            "candidate_preclip": interface["candidate_preclip"].detach().cpu().numpy(),
            "candidates": interface["candidates"].detach().cpu().numpy(),
            "wm_normalized_candidates": interface[
                "wm_normalized_candidates"
            ].detach().cpu().numpy(),
            "ensemble_prediction": prediction.detach().cpu().numpy(),
            "ensemble_uncertainty": uncertainty.detach().cpu().numpy(),
        }

    def plan_actor_default(
        self, observation: dict[str, np.ndarray], language: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Plan frozen C0-v2 while the world model remains logging-only."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        context = self._actor_context(observation, language)
        plan_seed = self.candidate_seed + self._rollout_seed * 1_000_003 + self._plan_index
        full_interface = self._interface(context["actor_normalized"], plan_seed)
        interface = {
            key: (value[:1] if key in {"candidate_preclip", "candidates", "wm_normalized_candidates"} else value)
            for key, value in full_interface.items()
        }
        prediction, uncertainty = self._predict(
            context, interface["wm_normalized_candidates"]
        )
        self._retain_plan_tensors(interface, prediction, uncertainty)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        diagnostics = {
            "plan_index": self._plan_index,
            "candidate_seed": plan_seed,
            "selected_candidate": 0,
            "replaced_actor_default": False,
            "planning_ms": (time.perf_counter() - started) * 1000.0,
            "candidate_uncertainty": uncertainty.cpu().tolist(),
            "interface": self._interface_diagnostics(interface),
            "executed_full_chunk": True,
            "early_replan": False,
        }
        self._plan_index += 1
        return interface["candidates"][0].cpu().numpy(), diagnostics

    def plan_chunk(
        self, observation: dict[str, np.ndarray], language: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        context = self._actor_context(observation, language)
        plan_seed = self.candidate_seed + self._rollout_seed * 1_000_003 + self._plan_index
        interface = self._interface(context["actor_normalized"], plan_seed)
        prediction, uncertainty = self._predict(
            context, interface["wm_normalized_candidates"]
        )
        rerank = self.config["rerank"]
        components = score_candidates(
            context["z0"],
            context["language"],
            interface["candidates"],
            prediction,
            uncertainty,
            self.wm_action_std,
            goal_progress_weight=float(rerank["goal_progress_weight"]),
            uncertainty_penalty_weight=float(rerank["uncertainty_penalty_weight"]),
            base_deviation_penalty_weight=float(
                rerank["base_deviation_penalty_weight"]
            ),
        )
        selected = int(torch.argmax(components["score"]).item())
        self._retain_plan_tensors(interface, prediction, uncertainty)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        selected_default_divergence = 1.0 - F.cosine_similarity(
            prediction[selected].float(), prediction[0].float(), dim=-1
        )
        diagnostics: dict[str, Any] = {
            "plan_index": self._plan_index,
            "candidate_seed": plan_seed,
            "selected_candidate": selected,
            "replaced_actor_default": selected != 0,
            "actor_default_score": float(components["score"][0].item()),
            "selected_score": float(components["score"][selected].item()),
            "score_gain_vs_default": float(
                (components["score"][selected] - components["score"][0]).item()
            ),
            "scores": components["score"].float().cpu().tolist(),
            "goal_progress": components["goal_progress"].float().cpu().tolist(),
            "mean_uncertainty": components["mean_uncertainty"].float().cpu().tolist(),
            "base_deviation": components["base_deviation"].float().cpu().tolist(),
            "final_language_similarity": components[
                "final_language_similarity"
            ].float().cpu().tolist(),
            "candidate_uncertainty": uncertainty.float().cpu().tolist(),
            "selected_default_prediction_cosine_distance": selected_default_divergence.cpu().tolist(),
            "interface": self._interface_diagnostics(interface),
            "selected_candidate_preclip": self._outside_summary(
                interface["candidate_preclip"][selected]
            ),
            "selected_raw_candidate_range": [
                float(interface["candidates"][selected].min().item()),
                float(interface["candidates"][selected].max().item()),
            ],
            "planning_ms": (time.perf_counter() - started) * 1000.0,
            "executed_full_chunk": True,
            "early_replan": False,
        }
        numeric = [
            diagnostics["scores"],
            diagnostics["goal_progress"],
            diagnostics["mean_uncertainty"],
            diagnostics["base_deviation"],
            diagnostics["candidate_uncertainty"],
        ]
        if not all(np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in numeric):
            raise FloatingPointError("C1-v2 planning diagnostics contain NaN/Inf")
        self._plan_index += 1
        return interface["candidates"][selected].cpu().numpy(), diagnostics

    def _plan(self, observation: dict[str, np.ndarray], language: str) -> np.ndarray:
        chunk, _ = self.plan_chunk(observation, language)
        return chunk


def create_policy(*, checkpoint: str | None, device: str) -> C1V2RerankPolicy:
    if checkpoint is None:
        checkpoint = str(PROJECT_ROOT / "results/c0_v2/seed11/best.pt")
    return C1V2RerankPolicy(checkpoint=checkpoint, device=device)
