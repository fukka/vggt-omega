"""Feed the backbone an upright frame. Everything else stays in the stored frame.

HOW THE REPO CAME TO RUN SIDEWAYS
---------------------------------
ADT stores Aria RGB in **native sensor orientation**, which is a quarter turn
off upright. The repo has two ADT loaders and they disagree:

* `raytun3r.data.ADTSequence` — the canonical one, used by `depthfisheye`,
  `fovbench` and the bench rows — documents it and rotates: *"Frames are stored
  rotated 90 deg CW on Aria; a 270 deg CCW rotation is applied to RGB and depth
  alike"*, paired with `aria_intrinsics(rotated=True)`.
* `AriaLocalPairs` — written for **H1.3**, the classical pose experiment — says
  *"frames are used exactly as stored (native sensor orientation), so the camera
  uses the native (rotated=False) intrinsics"*.

For H1.3 that was **correct**. SIFT matching and MAGSAC pose are invariant to a
quarter turn as long as the camera model matches the pixels, which it did; the
hand-eye gate passed at 0.77-0.96 deg on it. The mistake was inheriting that
loader unexamined when the line crossed from classical geometry into a
**pretrained depth network**, which is emphatically not rotation invariant.
H5's `Seq` wraps `AriaLocalPairs`, and H12, H14, H15 and H9 all import H5's
`Seq`, so six hypotheses ran on sideways frames. Nothing failed, because every
check the project ran was internally consistent — the camera did match the
pixels. It took someone looking at a picture.

Measured on seq136, 30 frames, frozen DA3-Small, prediction rotated back so
every row is scored on identical pixels, GT and masks:

    rotation   whole   near_rim   near_ctr   center    far
    k=0        0.5503   1.3925     0.5420    0.3241   0.2558   <- what ran
    k=1        0.6132   1.4843     0.9461    0.3562   0.2774
    k=2        0.5505   1.3669     0.5529    0.3199   0.2614
    k=3        0.1975   0.4101     0.3230    0.2051   0.1148   <- upright

k=0 and k=2 are the two sideways orientations and agree to the fourth decimal;
k=1 is upside-down and worst; k=3 is upright. Exactly the shape of a model with
an up-prior.

WHY THE FIX IS HERE AND NOT IN THE LOADER
-----------------------------------------
Rotating what `AriaLocalPairs.image()` returns would break everything that pairs
the image with the camera: H14's rig builds its sampling grids from the camera,
H15's lens family warps between cameras, the theta grid bins the zones, and the
GT and the pose conjugation are all in the stored frame. All of that is
verified and none of it cares about orientation.

Only the **backbone** cares. So the turn is applied at that one boundary: the
image goes in upright, the depth comes back rotated into the stored frame, and
the predicted pose is un-rolled. Nothing else in the pipeline moves.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["UPRIGHT_K", "to_model", "from_model", "unroll_R", "unroll_t",
           "forward_range", "forward_z"]

#: Quarter turns (`torch.rot90`) taking a stored ADT frame to upright.
UPRIGHT_K = 3

#: Which way the roll goes when undoing it on a predicted pose. Determined by
#: measurement, not by derivation, because getting it backwards is silent --
#: it produces a plausible-looking pose that is simply wrong. On seq136, 40
#: adjacent pairs, frozen DA3-Small, median rotation error / RRA@15:
#:
#:     unrotated (what every run did)   12.07 deg   0.550
#:     upright, no unroll               28.93 deg   0.125
#:     upright, unroll sign=+1          40.68 deg   0.075
#:     upright, unroll sign=-1           5.77 deg   0.925   <-
#:
#: So the turn more than halves the pose error as well as the depth error, and
#: the two wrong choices are both worse than not turning at all.
UNROLL_SIGN = -1.0


def to_model(x: Tensor, k: int = UPRIGHT_K) -> Tensor:
    """Stored frame -> upright. Works on (..., H, W)."""
    return torch.rot90(x, k, dims=(-2, -1)) if k % 4 else x


def from_model(x: Tensor, k: int = UPRIGHT_K) -> Tensor:
    """Upright -> stored frame. The exact inverse of :func:`to_model`."""
    return torch.rot90(x, -k, dims=(-2, -1)) if k % 4 else x


def _roll(k: int, sign: float) -> Tensor:
    a = sign * k * math.pi / 2.0
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def unroll_R(R: Tensor, k: int = UPRIGHT_K, sign: float = UNROLL_SIGN) -> Tensor:
    """A rotation predicted in the upright frame, expressed in the stored one.

    Turning the image rolls the camera frame about its optical axis, so a
    relative rotation predicted there is ``Rz R Rz^T``. Undoing it is a
    conjugation, which leaves the rotation ANGLE unchanged but not the axis --
    and `rotation_error_deg` compares against a GT expressed in the stored
    frame, so the axis is what it is measuring.
    """
    Rz = _roll(k, sign).to(R.dtype).to(R.device)
    return Rz.transpose(0, 1) @ R @ Rz


def unroll_t(t: Tensor, k: int = UPRIGHT_K, sign: float = UNROLL_SIGN) -> Tensor:
    Rz = _roll(k, sign).to(t.dtype).to(t.device)
    return t @ Rz            # rows are vectors: t Rz == (Rz^T t^T)^T


def forward_z(bb, img: Tensor, k: int = UPRIGHT_K) -> Tensor:
    """One frame in, PLANAR Z on the stored grid out.

    The backbone must be installed with ``depth_convention="z"``, and this is
    not a convenience -- it is the only correct order. `_finalize` would
    otherwise divide by ``cos(theta)`` read from the camera's grid while the
    prediction sits in the ROTATED frame, and Aria's principal point is 4.5 px
    off the frame centre at 504, which is about 1 degree of theta at the rim:
    a 2.5% radially-varying error, silently, in the direction this whole
    project measures. Planar z is invariant under a roll about the optical
    axis, so rotating it back is exact and the conversion happens afterwards,
    once, against the untouched camera.
    """
    if getattr(bb, "depth_convention", None) != "z":
        raise RuntimeError(
            "forward_z needs the backbone installed with depth_convention='z'; "
            f"got {getattr(bb, 'depth_convention', None)!r}. Converting inside "
            "forward() would apply the camera's cos map to a rotated prediction.")
    return from_model(bb.forward(to_model(img, k)[None, None]).depth[0], k)


def forward_range(bb, img: Tensor, cos_t: Tensor, k: int = UPRIGHT_K) -> Tensor:
    """Euclidean range on the stored grid. ``cos_t`` is the camera's own map."""
    z = forward_z(bb, img, k)
    return z / cos_t.to(z.device).clamp_min(1e-6)
