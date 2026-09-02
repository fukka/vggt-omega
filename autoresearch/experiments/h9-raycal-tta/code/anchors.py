"""Metric range anchors from cross-frame parallax, in bearing space.

WHY THIS EXISTS
---------------
This project's own measurements say two things that point the same way:

* **The rim GIVES alignment and does not RECEIVE fusion.** Widening the
  admitted field improves rotation on 17/17 pairs (H1.1) and the frozen model's
  pose collapses without the rim (runs 004-007), while six attempts to make the
  rim *receive* cross-frame or region-targeted help lost to their own controls
  (H5, H6, H7, MoE, H12, H15).
* **The depth failure is a radially-modulated range COMPRESSION, not noise** --
  dispersion 2-10% everywhere, bias up to 3.3x at the near rim (run_009). The
  model sees the rim fine; it reads it out wrong.

Put together: stop trying to make the rim receive, and let its giving produce
the depth. Wide-field parallax triangulates metric range at the very cells the
compression ruins, and those anchors are enough to fit and remove the field --
**with no depth labels anywhere**. Pose comes from video (Aria ships MPS SLAM
poses), matches come from the images.

TRIANGULATION, IN FRAME i
-------------------------
Camera j's centre in frame i is ``C = -R^T t`` and its ray direction for a
matched pixel is ``u2 = R^T d_j``. With ``u1 = d_i`` the two rays are
``lambda1 u1`` and ``C + lambda2 u2``; the least-squares closest approach gives

    lambda1 = ((u1.C) - c (u2.C)) / (1 - c^2),   c = u1.u2

and ``range_i = lambda1`` because ``u1`` is a unit bearing. Everything is done
on unit rays, never on pixels: on a fisheye the pixel metric is not the angular
metric, and a triangulation written in pixels is wrong by exactly the radial
factor this experiment exists to measure.

THE TWO GATES, BOTH MANDATORY
-----------------------------
* **Parallax.** ``1 - c^2`` is the conditioning of the solve. Near-parallel
  rays give a range that is arbitrarily large and arbitrarily wrong. #22
  measured this on real footage from the other side: adjacent frames buy
  ~nothing (ratios 0.98-1.07) while stride 10 buys 10-13%, because adjacent
  frames have no baseline.
* **Motion.** The brainstorm's cross-critique caught this and none of the three
  agents saw it alone: moving hands poison static-scene triangulation *exactly*
  in the worst cells, since hand pixels are 80%+ beyond 41 deg at median
  0.26-0.94 m (#28). A point is kept only if two independent partner frames
  agree on its range. A moving point cannot agree with itself.

Pure torch geometry, no backbone and no data, so the whole module is testable
on a CPU in seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = ["triangulate", "AnchorSet", "anchors_from_pairs"]


def _orthonormalise(R: Tensor) -> Tensor:
    """Nearest rotation to ``R``, by SVD.

    This function uses ``R^T`` as ``R^-1`` twice -- for camera j's centre and
    for its ray directions -- so a rotation that is not quite orthonormal is
    not a small error, it is amplified by the same ``1/(1 - c^2)`` that
    amplifies everything else here. A rotation composed in float32 is off by
    ~1e-7, which arrives as ~1e-3 of relative range at one degree of parallax:
    found by this experiment's own test fixture building its rotations in
    float32, where it looked exactly like a solver bug.
    """
    U, _, Vh = torch.linalg.svd(R)
    out = U @ Vh
    if float(torch.det(out)) < 0:
        U = U.clone()
        U[:, -1] *= -1
        out = U @ Vh
    return out


def triangulate(u1: Tensor, u2_j: Tensor, R: Tensor, t: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
    """Range in frame i for matched unit bearings.

    ``u1`` (N, 3) bearings in frame i; ``u2_j`` (N, 3) bearings in frame j;
    ``R``, ``t`` take a point from frame i to frame j (``X_j = R X_i + t``).

    Returns ``(range_i, parallax_rad, ok)``. ``ok`` is the cheirality test --
    both depths positive -- and is False wherever the solve is degenerate.
    """
    # Solved in float64 and cast back. The 1/(1 - c^2) below is the whole
    # conditioning of the problem: at a 12 cm baseline and 6 m range the
    # parallax is about 1 degree, c is within 1e-4 of 1, and float32 leaves
    # 1.4e-3 of relative range error purely in the arithmetic -- a systematic
    # error at exactly the far/rim cells the anchors exist to pin down.
    dt = u1.dtype
    u1 = u1.double()
    u2_j = u2_j.double()
    R = _orthonormalise(R.double())
    t = t.double().reshape(3)
    C = -(R.transpose(0, 1) @ t)                 # camera j's centre, in frame i
    u2 = u2_j @ R                                # = (R^T d_j)^T, rows are rays
    u2 = torch.nn.functional.normalize(u2, dim=-1)
    c = (u1 * u2).sum(-1)
    denom = (1.0 - c * c)
    b1 = (u1 * C).sum(-1)
    b2 = (u2 * C).sum(-1)
    safe = denom.abs() > 1e-9
    d = torch.where(safe, denom, torch.ones_like(denom))
    lam1 = (b1 - c * b2) / d
    lam2 = (c * b1 - b2) / d
    ok = safe & (lam1 > 0) & (lam2 > 0) & torch.isfinite(lam1)
    parallax = torch.acos(c.clamp(-1.0, 1.0))
    return lam1.to(dt), parallax.to(dt), ok


@dataclass
class AnchorSet:
    """Anchors for one reference frame: where they are and what range they say."""

    uv: Tensor            # (N, 2) pixel coordinates in the reference frame
    theta: Tensor         # (N,) incidence angle, radians
    rng: Tensor           # (N,) triangulated euclidean range, metres
    parallax: Tensor      # (N,) triangulation angle, radians
    n_seen: Tensor        # (N,) how many partner frames agreed

    def __len__(self) -> int:
        return int(self.uv.shape[0])

    def filtered(self, keep: Tensor) -> "AnchorSet":
        return AnchorSet(self.uv[keep], self.theta[keep], self.rng[keep],
                         self.parallax[keep], self.n_seen[keep])


def anchors_from_pairs(camera, uv_ref: Tensor, partners,
                       *, min_parallax_deg: float = 1.0,
                       agree_tol: float = 0.10,
                       max_range_m: float = 12.0) -> AnchorSet:
    """Triangulate one reference frame against several partners, and gate.

    ``partners`` is a sequence of ``(uv_match, R, t)``: for each partner, where
    each reference pixel landed in that frame, and the pose taking a point from
    the reference frame to it. A pixel becomes an anchor only if at least two
    partners produce a well-conditioned, positive-depth solution AND their
    ranges agree to ``agree_tol`` in log space. That agreement is the motion
    gate: a hand that moved between the two partner frames cannot satisfy it.

    The kept range is the MEDIAN over agreeing partners, not the mean: with
    three partners one bad match should not move the anchor, and a mean lets it.
    """
    u1 = camera.unproject(uv_ref)
    u1 = torch.nn.functional.normalize(u1, dim=-1)
    rows, oks = [], []
    for uv_j, R, t in partners:
        u2 = torch.nn.functional.normalize(camera.unproject(uv_j), dim=-1)
        rng, par, ok = triangulate(u1, u2, R, t)
        ok = ok & (par >= math.radians(min_parallax_deg)) & (rng <= max_range_m)
        rows.append(rng)
        oks.append(ok)
    if not rows:
        z = torch.zeros(0)
        return AnchorSet(torch.zeros(0, 2), z, z, z, z.long())

    Rng = torch.stack(rows, 0)                    # (P, N)
    OK = torch.stack(oks, 0)
    n_seen = OK.sum(0)

    # Log-median over the partners that solved, then agreement against it.
    big = torch.full_like(Rng, float("nan"))
    logs = torch.where(OK, Rng.clamp_min(1e-6).log(), big)
    med = torch.nanmedian(logs, dim=0).values
    agree = OK & ((logs - med).abs() <= agree_tol)
    n_agree = agree.sum(0)

    keep = n_agree >= 2
    logs_a = torch.where(agree, logs, torch.full_like(logs, float("nan")))
    rng_final = torch.nanmedian(logs_a, dim=0).values.exp()

    theta = torch.acos(u1[..., 2].clamp(-1.0, 1.0))
    par_med = torch.nanmedian(
        torch.where(agree, torch.stack([p for p in _parallaxes(camera, u1, partners)], 0),
                    torch.full_like(Rng, float("nan"))), dim=0).values
    a = AnchorSet(uv_ref, theta, rng_final, par_med, n_agree)
    return a.filtered(keep & torch.isfinite(rng_final))


def _parallaxes(camera, u1: Tensor, partners):
    for uv_j, R, t in partners:
        u2 = torch.nn.functional.normalize(camera.unproject(uv_j), dim=-1)
        yield triangulate(u1, u2, R, t)[1]
