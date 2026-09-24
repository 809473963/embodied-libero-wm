"""Small diagnostic probes that are not deployable world models."""

from .physical_visual import (
    ContextConditionedPhysicalVisualProbe,
    FrozenContextExtractor,
    fit_ridge,
    predict_ridge,
)

__all__ = [
    "ContextConditionedPhysicalVisualProbe",
    "FrozenContextExtractor",
    "fit_ridge",
    "predict_ridge",
]
