"""CPU tests for the lens family and its warps. No weights, no data.

Run on the box (this Mac has no torch -- POLICY.md, 2026-08-22 note):
    python -m pytest autoresearch/experiments/h15-lens-holdout/code/ -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raytun3r.cameras import from_aria  # noqa: E402
import lens_family as LF  # noqa: E402

SIZE = 252                      # small on purpose: these tests are geometry
ALL = LF.TRAIN_LENSES + LF.HELDOUT_LENSES


def aria(size: int = SIZE):
    return from_aria(size, size, rotated=False)


def tmax() -> float:
    return float(aria().theta_max)


# --------------------------------------------------------------- construction

@pytest.mark.parametrize("shape", ALL)
def test_every_lens_fills_the_same_disc(shape):
    """The cone maps onto one disc, identical for every lens in the family.

    This is what makes the augmentation a pure radial re-distribution: same
    rays, same pixels-worth of image, only the radial law differs.
    """
    cam = LF.make_lens(shape, SIZE, tmax())
    r = cam.fx * float(LF.SHAPES[shape][0](np.array([tmax()]), np)[0])
    assert r == pytest.approx(SIZE / 2.0 - 1.0, abs=1e-6)


@pytest.mark.parametrize("shape", ALL)
def test_the_inverse_is_the_inverse(shape):
    cam = LF.make_lens(shape, SIZE, tmax())
    th = torch.linspace(0.0, tmax(), 97, dtype=torch.float64)
    back = cam.theta_of_r(cam.r_of_theta(th))
    assert float((back - th).abs().max()) < 1e-6


def test_make_lens_refuses_a_lens_that_turns_over_inside_the_cone():
    """A KB4 read past its turnover stops describing the physical lens.

    Inverting past one returns a plausible WRONG angle, which reads downstream
    as "the model is bad at the rim" -- the exact conclusion under measurement.
    """
    with pytest.raises(ValueError, match="turns over"):
        LF.make_lens("kb4x1.5", SIZE, math.radians(75.0))


@pytest.mark.parametrize("shape", ALL)
def test_the_token_field_is_zero_on_axis_and_finite(shape):
    cam = LF.make_lens(shape, SIZE, tmax())
    f = LF.token_field(cam, SIZE, patch=14)
    assert torch.isfinite(f).all()
    centre = f[:, 2].argmin()
    assert abs(float(f[centre, 0])) < 0.05
    assert abs(float(f[centre, 1])) < 0.05


# ----------------------------------------------------------------- the warps

@pytest.mark.parametrize("shape", ALL)
def test_the_warp_has_no_void(shape):
    """Every pixel of the destination disc has a source ray -- by construction.

    Warping to a WIDER lens would leave a hole (3.3% median / 21.6% worst on
    the ScanNet++ route, dataset-scope sec. 3.2) and warping to a NARROWER one
    would throw the rim away. Fixing the cone avoids both, and this is the
    assertion that it actually did.
    """
    src, dst = aria(), LF.make_lens(shape, SIZE, tmax())
    _, valid = LF.grid_between(src, dst)
    cone = dst.valid_mask(SIZE, SIZE)
    assert float((valid & cone).sum()) / float(cone.sum()) > 0.9995


@pytest.mark.parametrize("shape", ALL)
def test_planar_z_is_invariant_under_a_lens_warp(shape):
    """Resample GT, convert nothing. The rule, checked numerically.

    A direction-only field is rendered independently in each camera; the source
    one is warped onto the destination grid; they must agree. Same rays in both
    cameras means the same theta, so planar z carries across untouched. Getting
    this wrong is a smooth radial error that no scale alignment can absorb --
    the class of bug that invalidated #38 v1.
    """
    src, dst = aria(), LF.make_lens(shape, SIZE, tmax())
    grid, valid = LF.grid_between(src, dst)

    def z_field(cam):
        rays = cam.ray_grid(cam.height, cam.width)
        rng = 2.0 + 0.7 * rays[..., 0] + 0.4 * rays[..., 1] + 0.3 * rays[..., 2]
        return rng * rays[..., 2]                      # range -> planar z

    moved = LF.warp(z_field(src), grid)
    direct = z_field(dst)
    theta = torch.acos(dst.ray_grid(SIZE, SIZE)[..., 2].clamp(-1, 1))
    inner = valid & (theta < dst.theta_max - math.radians(1.5))
    err = ((moved - direct).abs() / direct.abs())[inner]
    assert float(err.max()) < 5e-3, f"{shape}: {float(err.max()):.2e}"


def test_the_identity_warp_is_the_identity():
    """The native arm goes through the same resampler as every other lens.

    Otherwise the native lens would be the only one without interpolation blur
    and "which lens is easier" would be partly about the resampler.
    """
    cam = aria()
    grid, valid = LF.grid_between(cam, cam)
    rays = cam.ray_grid(SIZE, SIZE)
    field = 2.0 + 0.7 * rays[..., 0] + 0.4 * rays[..., 1]
    err = ((LF.warp(field, grid) - field).abs() / field.abs())[valid]
    assert float(err.max()) < 1e-4


def test_nearest_sampling_invents_no_new_values():
    """GT depth must be resampled with `nearest`, so the check is that it can be."""
    src, dst = aria(), LF.make_lens("stereographic", SIZE, tmax())
    grid, valid = LF.grid_between(src, dst)
    labels = torch.randint(0, 5, (SIZE, SIZE)).float()
    out = LF.warp(labels, grid, mode="nearest")[valid]
    assert set(np.unique(out.numpy()).tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}


# ------------------------------------------------- the identifiability claim

def test_held_out_lenses_are_interpolative_in_field_space():
    """The test must be transfer, not extrapolation.

    If a held-out lens's rim field sat outside the training range, every arm
    would fail and the experiment would be about extrapolation instead of about
    whether the geometry channel carries anything.
    """
    def rim(shape):
        d, dp = LF.SHAPES[shape]
        la, ln = LF.J.log_area_aniso(np.array([tmax()]),
                                     lambda t: d(t, np), lambda t: dp(t, np))
        return float(la[0]), float(ln[0])

    tr = np.array([rim(s) for s in LF.TRAIN_LENSES])
    for s in LF.HELDOUT_LENSES:
        a, n = rim(s)
        assert tr[:, 0].min() <= a <= tr[:, 0].max(), f"{s}: log_area {a} outside"
        assert tr[:, 1].min() <= n <= tr[:, 1].max(), f"{s}: log_aniso {n} outside"


def test_the_real_field_is_consistent_across_lenses_and_a_shuffle_is_not():
    """The whole reason H12's single-lens setting could not decide its question.

    Under the REAL field, the value at a token position is a function of that
    position's geometry, so two different lenses' fields are strongly related
    across positions. Under a per-lens shuffle they are unrelated. On ONE lens
    that difference is invisible -- either way the network sees one fixed
    position->value map it can memorise -- which is exactly why `jac` and
    `shuffled` came out equal there, and why the decider has to be a lens the
    model never saw.
    """
    g = torch.Generator().manual_seed(0)
    cams = {s: LF.make_lens(s, SIZE, tmax()) for s in ("aria_kb4", "equidistant",
                                                       "rectilinear")}
    fields = {s: LF.token_field(c, SIZE, patch=14) for s, c in cams.items()}
    shuf = {s: f[torch.randperm(f.shape[0], generator=g)]
            for s, f in fields.items()}

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))

    real = corr(fields["aria_kb4"][:, 0], fields["rectilinear"][:, 0])
    fake = corr(shuf["aria_kb4"][:, 0], shuf["rectilinear"][:, 0])
    assert real > 0.9, f"real cross-lens correlation only {real:.3f}"
    assert abs(fake) < 0.2, f"shuffled fields still correlated at {fake:.3f}"
