"""CPU tests for the anchors and the field fit. No weights, no data, no GPU."""
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
import anchors as AN  # noqa: E402
import raycal as RC  # noqa: E402


def cam(size: int = 252):
    return from_aria(size, size, rotated=False)


def synthetic_scene(n: int = 900, seed: int = 0):
    """Points in front of the camera, inside the imaged cone, at known range."""
    g = torch.Generator().manual_seed(seed)
    c = cam()
    # Sample bearings inside the cone, then push each out to its own range.
    th = torch.rand(n, generator=g) * (c.theta_max * 0.97)
    ph = torch.rand(n, generator=g) * 2 * math.pi
    d = torch.stack([torch.sin(th) * torch.cos(ph),
                     torch.sin(th) * torch.sin(ph), torch.cos(th)], dim=-1)
    rng = 0.4 + 6.0 * torch.rand(n, generator=g)
    return c, d, rng, d * rng[:, None]


def pose(rx=0.03, ry=-0.02, rz=0.01, tx=0.12, ty=0.02, tz=0.03):
    """A small rotation and a real translation -- i.e. an actual baseline."""
    def Rx(a): return torch.tensor([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                                    [0, math.sin(a), math.cos(a)]])
    def Ry(a): return torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                                    [-math.sin(a), 0, math.cos(a)]])
    def Rz(a): return torch.tensor([[math.cos(a), -math.sin(a), 0],
                                    [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return Rz(rz) @ Ry(ry) @ Rx(rx), torch.tensor([tx, ty, tz])


# ------------------------------------------------------------------ geometry

def test_triangulation_recovers_the_range_it_was_given():
    """The whole point: metric range at the rim, from parallax alone."""
    c, d, rng, X = synthetic_scene()
    R, t = pose()
    Xj = X @ R.transpose(0, 1) + t
    u2 = torch.nn.functional.normalize(Xj, dim=-1)
    got, par, ok = AN.triangulate(d, u2, R, t)
    assert bool(ok.all())
    rel = ((got - rng).abs() / rng)[ok]
    assert float(rel.max()) < 1e-4, f"max rel err {float(rel.max()):.2e}"


def test_it_works_at_the_rim_as_well_as_at_the_centre():
    """If it did not, the anchors would be useless exactly where they matter."""
    c, d, rng, X = synthetic_scene(n=2000, seed=3)
    R, t = pose()
    u2 = torch.nn.functional.normalize(X @ R.transpose(0, 1) + t, dim=-1)
    got, par, ok = AN.triangulate(d, u2, R, t)
    th = torch.acos(d[..., 2].clamp(-1, 1))
    rim = ok & (th > math.radians(38))
    ctr = ok & (th < math.radians(11))
    assert int(rim.sum()) > 50 and int(ctr.sum()) > 20
    err = ((got - rng).abs() / rng)
    assert float(err[rim].max()) < 1e-4
    assert float(err[ctr].max()) < 1e-4


def test_pure_rotation_gives_no_parallax_and_is_rejected():
    """No baseline, no triangulation -- and #22 measured the same thing on real
    footage from the other side: adjacent frames buy ~nothing."""
    c, d, rng, X = synthetic_scene()
    R, _ = pose(tx=0.0, ty=0.0, tz=0.0)
    u2 = torch.nn.functional.normalize(X @ R.transpose(0, 1), dim=-1)
    _, par, ok = AN.triangulate(d, u2, R, torch.zeros(3))
    assert float(par.max()) < math.radians(1e-3)
    assert int(ok.sum()) == 0


def test_the_motion_gate_drops_a_point_that_moved():
    """Two partners, one of which sees the point somewhere else.

    A hand that moved between the partner frames cannot make two independent
    triangulations agree, and the anchors it would otherwise poison are exactly
    the near-rim cells this project is about (#28: 80%+ of hand pixels beyond
    41 deg, median 0.26-0.94 m).
    """
    c, d, rng, X = synthetic_scene(n=400, seed=5)
    uv = c.project(d)
    R1, t1 = pose(tx=0.12)
    R2, t2 = pose(rx=-0.02, ry=0.03, tx=-0.10, tz=0.05)
    Xm = X.clone()
    moved = torch.zeros(X.shape[0], dtype=torch.bool)
    moved[:60] = True
    Xm[moved] = Xm[moved] * 1.6                     # they moved along the ray
    uv1 = c.project(torch.nn.functional.normalize(X @ R1.transpose(0, 1) + t1, dim=-1))
    uv2 = c.project(torch.nn.functional.normalize(Xm @ R2.transpose(0, 1) + t2, dim=-1))
    out = AN.anchors_from_pairs(c, uv, [(uv1, R1, t1), (uv2, R2, t2)])
    # Every surviving anchor must be a static one, and most static ones survive.
    assert len(out) > 200
    kept_uv = out.uv
    dist = torch.cdist(kept_uv, uv[moved])
    assert float(dist.min()) > 1e-3, "a moved point survived the agreement gate"


def test_a_single_partner_yields_nothing():
    """Two agreeing partners are required, so one frame cannot make an anchor."""
    c, d, rng, X = synthetic_scene(n=200)
    R, t = pose()
    uv = c.project(d)
    uv1 = c.project(torch.nn.functional.normalize(X @ R.transpose(0, 1) + t, dim=-1))
    assert len(AN.anchors_from_pairs(c, uv, [(uv1, R, t)])) == 0


# ----------------------------------------------------------------- the field

def _compressed(true_r, theta, g_of_theta, c0=0.0):
    """Synthesise the measured failure: a radially varying log-linear squeeze."""
    g = g_of_theta(theta)
    return np.exp(c0 + g * np.log(true_r))


def test_the_fit_recovers_a_known_radial_compression():
    rng = np.random.default_rng(0)
    n = 6000
    theta = rng.uniform(0, 0.95, n)
    true_r = np.exp(rng.uniform(np.log(0.4), np.log(8.0), n))
    g_of = lambda th: 0.95 - 0.55 * (th / 0.95)      # 0.95 on axis -> 0.40 at rim
    pred = _compressed(true_r, theta, g_of)
    field = RC.fit_field("raycal", theta, pred, true_r, theta_max=0.95)
    mid = 0.5 * (np.array(field["edges"][:-1]) + np.array(field["edges"][1:]))
    assert np.allclose(field["g"], g_of(mid), atol=0.03)
    back = RC.apply_field(pred, theta, field)
    assert float(np.abs(back / true_r - 1).max()) < 0.06


def test_the_global_arm_cannot_represent_a_radial_field():
    """The control that decides whether RADIAL structure is doing the work."""
    rng = np.random.default_rng(1)
    n = 6000
    theta = rng.uniform(0, 0.95, n)
    true_r = np.exp(rng.uniform(np.log(0.4), np.log(8.0), n))
    pred = _compressed(true_r, theta, lambda th: 0.95 - 0.55 * (th / 0.95))
    per_bin = RC.apply_field(pred, theta, RC.fit_field("raycal", theta, pred, true_r, 0.95))
    glob = RC.apply_field(pred, theta, RC.fit_field("global", theta, pred, true_r, 0.95))
    e_bin = float(np.abs(per_bin / true_r - 1).mean())
    e_glo = float(np.abs(glob / true_r - 1).mean())
    assert e_bin < 0.5 * e_glo, f"per-bin {e_bin:.3f} vs global {e_glo:.3f}"


def test_shuffling_theta_destroys_the_fit_it_would_otherwise_get():
    rng = np.random.default_rng(2)
    n = 6000
    theta = rng.uniform(0, 0.95, n)
    true_r = np.exp(rng.uniform(np.log(0.4), np.log(8.0), n))
    pred = _compressed(true_r, theta, lambda th: 0.95 - 0.55 * (th / 0.95))
    real = RC.apply_field(pred, theta, RC.fit_field("raycal", theta, pred, true_r, 0.95))
    shuf = RC.apply_field(pred, theta, RC.fit_field("shuffled", theta, pred, true_r, 0.95))
    assert float(np.abs(real / true_r - 1).mean()) < float(np.abs(shuf / true_r - 1).mean())


def test_the_correction_is_monotone_so_it_cannot_be_many_to_one():
    """The mechanism that killed H2.1, excluded by construction here.

    run_010's output-indexed table pushed the majority's fix onto minorities
    because the compression makes predicted depth many-to-one in true depth. A
    per-bin log-linear map with g > 0 is strictly increasing, so two different
    predictions at the same theta can never be sent to the same range.
    """
    field = {"edges": np.linspace(0, 0.95, 9).tolist(),
             "g": [0.4] * 8, "c": [0.1] * 8}
    theta = np.full(500, 0.5)
    pred = np.linspace(0.2, 9.0, 500)
    out = RC.apply_field(pred, theta, field)
    assert bool(np.all(np.diff(out) > 0))
