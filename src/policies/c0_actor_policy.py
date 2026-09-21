"""Deploy a trained C0 action-chunk actor in the live LIBERO viewer."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor

from src.c0 import C0Actor


MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class C0ActorPolicy:
    """Convert the online C0 contract to one buffered 7-D action at a time."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        if checkpoint is None:
            raise ValueError("C0ActorPolicy requires --checkpoint")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for C0 policy but is unavailable")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("method") != "c0_actor_only":
            raise ValueError("checkpoint is not a C0 actor checkpoint")
        model_state = state["model_state"]
        fusion_dim = int(model_state["vision_encoder.0.weight"].shape[0])
        feature_dim = int(model_state["vision_encoder.0.weight"].shape[1])
        proprio_dim = int(model_state["proprio_encoder.0.weight"].shape[1])
        output_dim = int(model_state["action_head.4.weight"].shape[0])
        if output_dim % 7:
            raise ValueError("C0 checkpoint output is not divisible into 7-D LIBERO actions")
        self.model = C0Actor(
            feature_dim=feature_dim,
            proprio_dim=proprio_dim,
            fusion_dim=fusion_dim,
            action_horizon=output_dim // 7,
            action_dim=7,
        ).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        normalization = state["normalization"]
        self.action_mean = torch.tensor(
            normalization["action"]["mean"], dtype=torch.float32, device=self.device
        ).view(1, 1, 7)
        self.action_std = torch.tensor(
            normalization["action"]["std"], dtype=torch.float32, device=self.device
        ).view(1, 1, 7)
        self.proprio_mean = torch.tensor(
            normalization["proprio"]["mean"], dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.proprio_std = torch.tensor(
            normalization["proprio"]["std"], dtype=torch.float32, device=self.device
        ).view(1, -1)
        if self.proprio_mean.shape[1] != 12:
            raise ValueError("this adapter requires the contract-v1 12-D C0 proprioception")

        project_root = Path(__file__).resolve().parents[2]
        snapshot = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            # extraction stores the pinned snapshot under this exact cache root;
            # using its parent would redownload the same 600 MiB model.
            cache_dir=project_root / ".cache/huggingface/hub",
            allow_patterns=["*.json", "vocab.json", "merges.txt", "pytorch_model.bin"],
            # Evaluation must use the exact feature-extraction snapshot already
            # recorded in the repository.  Do not let a transient proxy or
            # Hugging Face availability alter (or block) a rollout.
            local_files_only=True,
        )
        self.processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
        self.clip = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(self.device)
        self.clip.eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self._language_cache: dict[str, torch.Tensor] = {}
        self._chunk = np.empty((0, 7), dtype=np.float32)
        self._index = 0

    def reset(self) -> None:
        self._chunk = np.empty((0, 7), dtype=np.float32)
        self._index = 0

    def _autocast(self) -> contextlib.AbstractContextManager[None]:
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _language_features(self, language: str) -> torch.Tensor:
        cached = self._language_cache.get(language)
        if cached is not None:
            return cached
        tokens = self.processor(text=[language], padding=True, return_tensors="pt")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.inference_mode(), self._autocast():
            features = self.clip.get_text_features(**tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cached = features.float()
        self._language_cache[language] = cached
        return cached

    def plan_batch(
        self, observations: list[dict[str, np.ndarray]], languages: list[str]
    ) -> np.ndarray:
        """Return independent action chunks for a batch of live C0 inputs.

        Batch members do not share observations, actions, or state; batching is
        purely a GPU inference optimization for fixed evaluation episodes.
        """
        if not observations or len(observations) != len(languages):
            raise ValueError("observations and languages must be non-empty and equal length")
        required = ("agentview_image", "robot0_eef_pos", "robot0_gripper_qpos", "robot0_joint_pos")
        images = []
        proprio_rows = []
        for observation in observations:
            missing = [key for key in required if key not in observation]
            if missing:
                raise KeyError(f"C0 observation is missing {missing}")
            image = np.asarray(observation["agentview_image"], dtype=np.uint8)
            if image.shape != (128, 128, 3):
                raise ValueError(f"expected agentview_image [128,128,3], got {image.shape}")
            images.append(np.ascontiguousarray(image))
            proprio_rows.append(
                np.concatenate(
                    (
                        np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
                        np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
                        np.asarray(observation["robot0_joint_pos"], dtype=np.float32).reshape(7),
                    )
                )
            )
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        proprio_tensor = torch.from_numpy(np.stack(proprio_rows)).to(self.device)
        language_features = torch.cat(
            [self._language_features(language) for language in languages], dim=0
        )
        with torch.inference_mode(), self._autocast():
            image_features = self.clip.get_image_features(pixel_values=pixels)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            action_chunk = self.model(
                image_features.float(),
                language_features,
                (proprio_tensor - self.proprio_mean) / self.proprio_std,
            )
            action_chunk = action_chunk * self.action_std + self.action_mean
        return np.clip(action_chunk.float().cpu().numpy(), -1.0, 1.0)

    def _plan(self, observation: dict[str, np.ndarray], language: str) -> np.ndarray:
        return self.plan_batch([observation], [language])[0]

    def act(self, observation: dict[str, np.ndarray], language: str) -> np.ndarray:
        if self._index >= len(self._chunk):
            self._chunk = self._plan(observation, language)
            self._index = 0
        action = self._chunk[self._index]
        self._index += 1
        return action


def create_policy(*, checkpoint: str | None, device: str) -> C0ActorPolicy:
    """Factory compatible with ``scripts/run_policy_viewer.py``."""

    return C0ActorPolicy(checkpoint=checkpoint, device=device)
