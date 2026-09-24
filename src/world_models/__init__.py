"""World-model components introduced after the frozen D13 data contract."""

from .latent_dynamics import LatentDynamicsModel, cosine_distance
from .ensemble import LatentDynamicsEnsemble, ensemble_statistics
from .normalization import apply_world_model_normalization, compute_world_model_normalization
from .proprio_aux import (
    FutureProprioWorldModelDataset,
    LatentDynamicsProprioAuxModel,
    PROPRIO_GROUPS,
    proprio_delta_error,
    standardized_proprio_delta_target,
)
from .ej_coupled import (
    EJ_PROPRIO_INDICES,
    LatentDynamicsEJCoupledModel,
    ej_delta_error,
    physical_visual_residual_loss,
)

__all__ = [
    "LatentDynamicsModel",
    "LatentDynamicsProprioAuxModel",
    "LatentDynamicsEJCoupledModel",
    "LatentDynamicsEnsemble",
    "FutureProprioWorldModelDataset",
    "PROPRIO_GROUPS",
    "EJ_PROPRIO_INDICES",
    "apply_world_model_normalization",
    "compute_world_model_normalization",
    "cosine_distance",
    "ensemble_statistics",
    "ej_delta_error",
    "physical_visual_residual_loss",
    "proprio_delta_error",
    "standardized_proprio_delta_target",
]
