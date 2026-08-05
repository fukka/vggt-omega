"""SO(3)/SE(3) plumbing and the ray <-> spherical-angle conversions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.geometry import (
    angle_between,
    geodesic_angle,
    make_se3,
    matrix_to_rot6d,
    random_rotations,
    rays_to_spherical,
    relative_pose,
    rot6d_to_matrix,
    se3_compose,
    se3_inverse,
    spherical_to_rays,
    transform_points,
    wrap_angle,
)


def test_rot6d_round_trip():
    R = random_rotations(64, seed=0)
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(R)), R, atol=1e-5)


def test_rot6d_output_is_a_rotation():
    torch.manual_seed(1)
    R = rot6d_to_matrix(torch.randn(32, 6))          # arbitrary, un-normalized input
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(32), atol=1e-5)


def test_geodesic_angle_matches_a_known_rotation():
    for deg in (0.0, 12.5, 90.0, 179.0):
        a = math.radians(deg)
        Rz = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                           [math.sin(a), math.cos(a), 0.0],
                           [0.0, 0.0, 1.0]])
        got = geodesic_angle(Rz.unsqueeze(0), torch.eye(3).unsqueeze(0))
        assert abs(float(got) - a) < 1e-5


def test_geodesic_angle_is_bi_invariant():
    R1, R2, Q = random_rotations(16, seed=2), random_rotations(16, seed=3), random_rotations(16, seed=4)
    base = geodesic_angle(R1, R2)
    assert torch.allclose(geodesic_angle(Q @ R1, Q @ R2), base, atol=1e-5)
    assert torch.allclose(geodesic_angle(R1 @ Q, R2 @ Q), base, atol=1e-5)


def test_angle_between_ignores_magnitude():
    v = torch.tensor([[1.0, 0.0, 0.0]])
    w = torch.tensor([[0.0, 5.0, 0.0]])
    assert abs(float(angle_between(v, w)) - math.pi / 2) < 1e-6
    assert abs(float(angle_between(v, 3.0 * v))) < 1e-6


def test_se3_inverse_and_compose():
    R, t = random_rotations(8, seed=5), torch.randn(8, 3)
    T = make_se3(R, t)
    eye = torch.eye(4).expand_as(T)
    assert torch.allclose(se3_compose(T, se3_inverse(T)), eye, atol=1e-5)
    pts = torch.randn(8, 100, 3)
    assert torch.allclose(transform_points(se3_inverse(T), transform_points(T, pts)), pts, atol=1e-4)


def test_relative_pose_recovers_a_known_motion():
    """relative_pose(T1, T2) maps camera-2 coordinates into camera-1."""
    T1, T2 = make_se3(random_rotations(4, seed=6), torch.randn(4, 3)), make_se3(
        random_rotations(4, seed=7), torch.randn(4, 3)
    )
    R21, t21 = relative_pose(T1, T2)
    x2 = torch.randn(4, 50, 3)
    world = transform_points(se3_inverse(T2), x2)      # cam2 -> world
    x1 = transform_points(T1, world)                   # world -> cam1
    assert torch.allclose(x2 @ R21.transpose(-1, -2) + t21.unsqueeze(1), x1, atol=1e-4)


def test_spherical_gradients_are_finite_at_the_poles():
    """The on-axis pixel hits z=1 and x=y=0 *exactly* on every real camera.

    Regression: with arccos(z) the backward pass through an accurate ray field
    is all-NaN from the first step, which is exactly what a UniK3D-initialized
    Ray Module produces.
    """
    d = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    theta, phi = rays_to_spherical(d)
    (theta.sum() + phi.sum()).backward()
    assert torch.isfinite(d.grad).all(), d.grad
    # And the value is still exact on axis, not clamped away from it
    # (1e-6 is float32's resolution near pi, not slack in the guard).
    assert float(theta[0]) < 1e-9 and abs(float(theta[1]) - math.pi) < 1e-6


def test_spherical_round_trip():
    torch.manual_seed(8)
    d = torch.randn(1000, 3, dtype=torch.float64)
    d = d / d.norm(dim=-1, keepdim=True)
    theta, phi = rays_to_spherical(d)
    assert torch.allclose(spherical_to_rays(theta, phi), d, atol=1e-9)


def test_spherical_conventions():
    """theta is incidence from +z (optical axis); phi is image-plane azimuth from +x."""
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    theta, _ = rays_to_spherical(axis)
    assert abs(float(theta)) < 1e-7

    right = torch.tensor([[1.0, 0.0, 0.0]])
    theta, phi = rays_to_spherical(right)
    assert abs(float(theta) - math.pi / 2) < 1e-6 and abs(float(phi)) < 1e-6

    down = torch.tensor([[0.0, 1.0, 0.0]])
    _, phi = rays_to_spherical(down)
    assert abs(float(phi) - math.pi / 2) < 1e-6


def test_wrap_angle_handles_the_seam():
    assert abs(float(wrap_angle(torch.tensor(3.0 * math.pi)))) - math.pi < 1e-6
    near_seam = wrap_angle(torch.tensor(math.pi - 0.01) - torch.tensor(-math.pi + 0.01))
    assert abs(float(near_seam)) < 0.03      # 0.02 rad apart across the seam, not ~2pi
