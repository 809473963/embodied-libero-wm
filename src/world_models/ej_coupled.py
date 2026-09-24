"""WM-v2C-EJ explicit EE/joint physical-to-visual coupling."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .proprio_aux import LatentDynamicsProprioAuxModel


EJ_PROPRIO_INDICES = (0, 1, 2, 5, 6, 7, 8, 9, 10, 11)
EJ_EE_INPUT = slice(0, 3)
EJ_JOINT_INPUT = slice(3, 10)
AdapterMode = Literal["full_ej", "ee_only", "joints_only"]


class LatentDynamicsEJCoupledModel(LatentDynamicsProprioAuxModel):
    """WM-v2B plus a small predicted-EE/joint-to-visual residual adapter.

    The legacy 12-D physical head is retained for exact WM-v2B weight loading,
    but the adapter input is the fixed 10-D EE-position/joint subset. Gripper
    dimensions 3 and 4 have no path into the adapter.
    """

    def __init__(
        self,
        physical_adapter_hidden_dim: int = 64,
        physical_adapter_bottleneck_dim: int = 128,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        self.ej_dim = len(EJ_PROPRIO_INDICES)
        self.physical_adapter_hidden_dim = int(physical_adapter_hidden_dim)
        self.physical_adapter_bottleneck_dim = int(
            physical_adapter_bottleneck_dim
        )
        self.physical_visual_adapter = nn.Sequential(
            nn.Linear(self.ej_dim, self.physical_adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.physical_adapter_hidden_dim,
                self.physical_adapter_bottleneck_dim,
            ),
            nn.GELU(),
            nn.Linear(self.physical_adapter_bottleneck_dim, self.latent_dim),
        )
        # Preserve the frozen WM-v2B visual prediction exactly at initialization.
        final = self.physical_visual_adapter[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def select_ej(predicted_proprio_delta: torch.Tensor) -> torch.Tensor:
        if predicted_proprio_delta.ndim != 3 or predicted_proprio_delta.shape[-1] != 12:
            raise ValueError(
                "expected predicted proprio delta [B,H,12], got "
                f"{tuple(predicted_proprio_delta.shape)}"
            )
        return predicted_proprio_delta[..., EJ_PROPRIO_INDICES]

    @staticmethod
    def ablate_adapter_input(
        predicted_ej_delta: torch.Tensor, mode: AdapterMode
    ) -> torch.Tensor:
        if predicted_ej_delta.ndim != 3 or predicted_ej_delta.shape[-1] != 10:
            raise ValueError(
                f"expected predicted EJ delta [B,H,10], got {tuple(predicted_ej_delta.shape)}"
            )
        if mode == "full_ej":
            return predicted_ej_delta
        result = predicted_ej_delta.clone()
        if mode == "ee_only":
            result[..., EJ_JOINT_INPUT] = 0.0
        elif mode == "joints_only":
            result[..., EJ_EE_INPUT] = 0.0
        else:
            raise ValueError(f"unsupported adapter mode: {mode}")
        return result

    def forward_with_components(
        self,
        z0: torch.Tensor,
        proprio0: torch.Tensor,
        language: torch.Tensor,
        actions: torch.Tensor,
        adapter_mode: AdapterMode = "full_ej",
    ) -> dict[str, torch.Tensor]:
        if z0.ndim != 2 or z0.shape[-1] != self.latent_dim:
            raise ValueError(f"expected z0 [B,{self.latent_dim}], got {tuple(z0.shape)}")
        batch_size = z0.shape[0]
        if proprio0.shape != (batch_size, self.proprio_dim):
            raise ValueError(
                f"expected proprio0 [B,{self.proprio_dim}], got {tuple(proprio0.shape)}"
            )
        if language.shape != (batch_size, self.latent_dim):
            raise ValueError(
                f"expected language [B,{self.latent_dim}], got {tuple(language.shape)}"
            )
        if actions.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError(
                f"expected actions [B,{self.horizon},{self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )

        language_projection = self.language_projection(language)
        fused_state = torch.cat(
            (
                self.z_projection(z0),
                self.proprio_projection(proprio0),
                language_projection,
            ),
            dim=-1,
        )
        initial_hidden = self.initial_hidden(fused_state).unsqueeze(0)
        recurrent_input = torch.cat(
            (
                self.action_projection(actions),
                language_projection.unsqueeze(1).expand(-1, self.horizon, -1),
            ),
            dim=-1,
        )
        recurrent_output, _ = self.gru(recurrent_input, initial_hidden)
        base_visual_delta = self.delta_head(recurrent_output)
        predicted_proprio_delta = self.proprio_delta_head(recurrent_output)
        predicted_ej_delta = self.select_ej(predicted_proprio_delta)
        adapter_input = self.ablate_adapter_input(predicted_ej_delta, adapter_mode)
        physical_visual_residual = self.physical_visual_adapter(adapter_input)
        final_visual_delta = base_visual_delta + physical_visual_residual
        pred_z = F.normalize(
            z0.unsqueeze(1) + final_visual_delta, p=2, dim=-1, eps=1.0e-8
        )
        return {
            "pred_z": pred_z,
            "pred_proprio_delta": predicted_proprio_delta,
            "pred_ej_delta": predicted_ej_delta,
            "adapter_input": adapter_input,
            "base_visual_delta": base_visual_delta,
            "physical_visual_residual": physical_visual_residual,
            "final_visual_delta": final_visual_delta,
            "recurrent_output": recurrent_output,
        }

    def forward(
        self,
        z0: torch.Tensor,
        proprio0: torch.Tensor,
        language: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        components = self.forward_with_components(z0, proprio0, language, actions)
        return components["pred_z"], components["pred_proprio_delta"]

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.physical_visual_adapter.parameters())


def ej_delta_error(
    predicted_proprio_delta: torch.Tensor,
    target_proprio_delta: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 over the fixed EE-position and joint dimensions only."""

    if predicted_proprio_delta.shape != target_proprio_delta.shape:
        raise ValueError("prediction and target must share [B,H,12]")
    if predicted_proprio_delta.ndim != 3 or predicted_proprio_delta.shape[-2:] != (8, 12):
        raise ValueError(
            f"expected prediction and target [B,8,12], got {tuple(predicted_proprio_delta.shape)}"
        )
    prediction = predicted_proprio_delta[..., EJ_PROPRIO_INDICES]
    target = target_proprio_delta[..., EJ_PROPRIO_INDICES]
    return F.smooth_l1_loss(
        prediction.float(), target.float(), beta=beta, reduction="none"
    ).mean(dim=-1)


def physical_visual_residual_loss(residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 3 or residual.shape[-2:] != (8, 512):
        raise ValueError(f"expected residual [B,8,512], got {tuple(residual.shape)}")
    return residual.float().square().mean(dim=-1)
