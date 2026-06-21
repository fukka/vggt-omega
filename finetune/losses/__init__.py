# Copyright (c) 2026.
"""Self-supervised and distillation losses for egocentric finetuning."""
from __future__ import annotations

from .distillation import (
    affine_invariant_l1,
    gradient_matching_loss,
    multiview_consistency_loss,
    ordinal_distill_loss,
    ssi_loss,
    to_disparity,
)
from .dynamic import combine_masks, flow_residual_mask, rigid_flow
from .photometric import edge_aware_smoothness, photometric_error
from .self_supervised import compute_self_supervised_losses

__all__ = [
    "compute_self_supervised_losses",
    "photometric_error",
    "edge_aware_smoothness",
    "affine_invariant_l1",
    "ssi_loss",
    "gradient_matching_loss",
    "multiview_consistency_loss",
    "ordinal_distill_loss",
    "to_disparity",
    "rigid_flow",
    "flow_residual_mask",
    "combine_masks",
]
