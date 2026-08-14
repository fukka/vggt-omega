# Copyright (c) 2026.
"""The depth-convention check, on scenes whose convention is known by construction.

The measurement in ``verify_depth_convention`` answers one question — is ego-synth's
``d`` planar z or euclidean range — and the dangerous failure is not a wrong
number. It is a check that **can only ever say "z"**: match on something that
quietly assumes z, and the answer comes back z on any input at all, including
data that is range.

So the load-bearing test here is the second one. A synthetic take is built under
each convention in turn, and the check must return that convention and not the
other. Everything else is scaffolding for that.

CPU-only, no data, no weights, no network.
"""
from __future__ import annotations

import numpy as np
import pytest

from slambench import camera as C
from slambench import verify_depth_convention as V

#: The same stand-in lens ``test_camera`` uses: right magnitudes for a 2880
#: sensor, not a copy of any device's. Tests must not carry data out of a
#: licensed release.
PARAMS = (1200.0, 1440.0, 1440.0,
          0.40, -0.50, 0.24, 0.99, -1.57, 0.61,
          -3.7e-4, -7.4e-4,
          -5.8e-4, -2.2e-4, 1.9e-4, -2.0e-4)


def cam(size: int = 896) -> C.Fisheye624:
    return C.Fisheye624(PARAMS, 2880, 2880, dataset="test",
                        take="t").rescale(size)


def a_pose(seed: int = 0) -> np.ndarray:
    """A camera pose that is not the identity, so a dropped rotation shows up."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = q
    T[:3, 3] = rng.normal(scale=2.0, size=3)
    return T


def synth_take(convention: str, n: int = 4000, seed: int = 0,
               size: int = 896):
    """A frame's stored rows plus the world cloud they came from.

    Points are placed by choosing a pixel and a distance, so every one of them
    is genuinely visible and the incidence angles span the lens rather than
    clustering on axis. ``convention`` decides what goes in the ``d`` column and
    is the only thing that differs between the two builds.
    """
    c = cam(size)
    rng = np.random.default_rng(seed)
    # Spread over the frame, then keep what unprojects to a sane forward ray.
    u = rng.uniform(0.06 * size, 0.94 * size, n)
    v = rng.uniform(0.06 * size, 0.94 * size, n)
    rays = c.unproject(u, v)
    ok = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 0.15)
    u, v, rays = u[ok], v[ok], rays[ok]
    dist = rng.uniform(0.8, 6.0, u.size)          # metres along the ray
    p_cam = rays * dist[:, None]                  # the true 3-D points

    z = p_cam[:, 2]
    rng_ = np.linalg.norm(p_cam, axis=1)
    d = z if convention == "z" else rng_

    T = a_pose(seed)
    world = p_cam @ T[:3, :3].T + T[:3, 3]
    uvd = np.stack([u, v, d], axis=1)
    return c, uvd, T, world


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #

def test_matching_is_direction_only_and_unique():
    """Matches land on the right points, and duplicates along a ray are dropped."""
    c, uvd, T, world = synth_take("z", n=1500, seed=3)
    got = V.frame_pairs(c, uvd, T, world, V.MATCH_TOL_PX / c.f)
    assert got is not None
    assert got["n_unique"] > 200
    # The correspondence is right: under the convention it was built with, d and
    # z agree to the floor.
    assert np.median(np.abs(got["d"] - got["z"])) < 1e-6

    # A second point at twice the distance along every ray is pure ambiguity,
    # and must cost matches rather than being resolved by depth.
    doubled = np.vstack([world, (2.0 * (world - T[:3, 3]) @ T[:3, :3]
                                 @ T[:3, :3].T) + T[:3, 3]])
    fewer = V.frame_pairs(c, uvd, T, doubled, V.MATCH_TOL_PX / c.f)
    assert fewer is None or int(fewer["n_unique"]) < int(got["n_unique"])


def test_bins_cover_the_span_and_count_every_point():
    theta = np.array([1.0, 14.0, 16.0, 29.0, 31.0, 44.0, 46.0, 80.0])
    resid = {"z": np.zeros(8)}
    rows = V.bin_stats(theta, resid)
    assert [r["n"] for r in rows] == [2, 2, 2, 2]
    assert sum(r["n"] for r in rows) == theta.size


def test_verdict_refuses_without_reach():
    """Coverage is checked before the medians are compared, not after."""
    theta = np.full(5000, 10.0)                    # plenty of points, no reach
    v = V.verdict(theta, np.zeros(5000), np.full(5000, 0.9))
    assert v.startswith("undecided")
    assert "beyond" in v


def test_verdict_refuses_a_narrow_margin():
    theta = np.full(5000, 50.0)
    v = V.verdict(theta, np.full(5000, 0.10), np.full(5000, 0.12))
    assert v.startswith("undecided")


# --------------------------------------------------------------------------- #
# The one that matters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("truth", ["z", "range"])
def test_the_check_recovers_the_convention_it_was_built_with(truth):
    """**The falsifiability test.** Range data must read range, not z.

    A check that returns "z" on a take built as range is not a check, and its
    agreement with the data card would mean nothing. This is the reason the
    matching in :func:`verify_depth_convention.frame_pairs` is on ray direction
    alone.
    """
    c, uvd, T, world = synth_take(truth, n=9000, seed=11)
    got = V.frame_pairs(c, uvd, T, world, V.MATCH_TOL_PX / c.f)
    assert got is not None

    rel_z = (got["d"] - got["z"]) / got["z"]
    rel_r = (got["d"] - got["range"]) / got["range"]
    v = V.verdict(got["theta"], rel_z, rel_r)
    assert v.startswith(truth), v

    # And the loser is wrong by a *specific* amount, not merely a large one.
    # The two hypotheses predict different functions of theta -- d=z implies the
    # range residual is 1-cos(theta), d=range implies the z residual is
    # sec(theta)-1 -- so pinning the shape is what separates "the other reading
    # is wrong" from "the other reading is wrong in exactly the way this one
    # being right would make it".
    far = got["theta"] >= 40.0
    assert far.sum() > 200
    th = np.radians(np.median(got["theta"][far]))
    loser = rel_r if truth == "z" else rel_z
    want = (1.0 - np.cos(th)) if truth == "z" else (1.0 / np.cos(th) - 1.0)
    assert np.median(np.abs(loser[far])) == pytest.approx(want, rel=0.10)


def test_reconstruction_error_separates_the_two_readings():
    """The independent reading: no matching at all, same conclusion."""
    c, uvd, T, world = synth_take("z", n=6000, seed=5)
    rc = V.reconstruction_error(c, uvd, T, V.build_tree(world))
    assert rc is not None
    far = rc["theta"] >= 30.0
    assert far.sum() > 100
    assert np.median(rc["z"][far]) < 1e-6
    assert np.median(rc["range"][far]) > 0.05


def test_reconstruction_error_flips_with_the_convention():
    c, uvd, T, world = synth_take("range", n=6000, seed=5)
    rc = V.reconstruction_error(c, uvd, T, V.build_tree(world))
    assert rc is not None
    far = rc["theta"] >= 30.0
    assert np.median(rc["range"][far]) < 1e-6
    assert np.median(rc["z"][far]) > 0.05


def test_brute_force_nn_matches_scipy():
    """The scipy-free fallback is the definition of the answer, not an approximation."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(500, 3))
    query = rng.normal(size=(64, 3))
    d_tree, i_tree = V._nn(query, V.build_tree(cloud))
    d_brute, i_brute = V._nn(query, cloud)
    assert np.allclose(d_tree, d_brute)
    assert np.array_equal(i_tree, i_brute)
