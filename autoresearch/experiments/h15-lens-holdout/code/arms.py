"""What each arm's FiLM conditioner is shown, for a given lens.

Shared by the trainer and the evaluator so the two cannot disagree about what
an arm means. The evaluator has to build the field for a lens the trainer never
saw, so "just save the tensors" is not available -- the RULE has to be shared,
and it has to be deterministic.

    jac          the real (log_area, log_aniso, theta/theta_max) of THIS lens
    mismatched   the real field of a DIFFERENT lens -- smooth, plausible,
                 correct-looking geometry that does not describe this image
    shuffled     this lens's own field under a per-lens permutation of token
                 positions -- same values, no spatial correspondence
    none         no conditioner at all (plain LoRA, the standing baseline)

WHY THERE ARE TWO CONTROLS
--------------------------
`shuffled` is H12's control and it is distribution-matched but destroys
smoothness, so a win over it could in principle be "a smooth field is easier to
fit" rather than "this is the right geometry". `mismatched` closes that: it is
a real lens's real field, equally smooth, equally structured, and wrong. A win
over BOTH is the only reading that means the network used the geometry of the
image in front of it.

WHY THE PERMUTATION IS PER LENS
-------------------------------
One permutation shared by every lens would preserve the value<->geometry
correspondence up to a single fixed relabelling of positions, which the network
can learn once and then use everywhere -- a control that is not a control. A
different permutation per lens is what makes the arriving values uninformative
about the geometry at that position.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Sequence

import torch

ARMS = ("jac", "mismatched", "shuffled", "none")


def _lens_seed(base_seed: int, lens: str) -> int:
    """A per-lens seed that does not depend on list order or dict iteration.

    Derived from the lens NAME, so adding a lens to the family later cannot
    change the permutation an existing lens gets -- otherwise re-running an arm
    after the family grows would silently be a different experiment.
    """
    h = hashlib.sha256(f"{base_seed}:{lens}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def mismatched_partner(lens: str, all_names: Sequence[str],
                       train_names: Sequence[str]) -> str:
    """The training lens whose field `mismatched` shows instead of `lens`.

    Deterministic and never the lens itself, so the arm is a function of the
    lens exactly as `jac` and `shuffled` are -- three arms that differ in what
    the field says, not in how it is drawn.
    """
    i = list(all_names).index(lens)
    partner = list(train_names)[(i + 1) % len(train_names)]
    assert partner != lens, f"mismatched partner for {lens} is itself"
    return partner


def arm_field(arm: str, lens: str, fields: Dict[str, torch.Tensor],
              all_names: Sequence[str], train_names: Sequence[str],
              base_seed: int) -> torch.Tensor:
    """The (P, 3) field this arm shows for this lens."""
    if arm == "jac":
        return fields[lens]
    if arm == "mismatched":
        return fields[mismatched_partner(lens, all_names, train_names)]
    if arm == "shuffled":
        f = fields[lens]
        g = torch.Generator().manual_seed(_lens_seed(base_seed, lens))
        return f[torch.randperm(f.shape[0], generator=g)].contiguous()
    if arm == "none":
        raise ValueError("arm 'none' has no conditioner; do not call arm_field")
    raise ValueError(f"unknown arm {arm!r}")
