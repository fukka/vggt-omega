"""Pin the KB4 inversion accuracy across the full imaged cone.

The plain-Newton ``theta_of_r`` initialised at ``theta = r``; with positive
k1 that overshoots into the flat region near the polynomial turnover, where
the step explodes and the iteration lands up to ~1 px wrong for rays in the
outermost degrees of the cone. Theta-binned aggregates never noticed (a 1 px
slip is far below bin width) — differentiable warping did (H5 losses,
2026-08-23). Bisection-safeguarded Newton fixes it; this test keeps it fixed.
"""

import math

import torch

from raytun3r.cameras import KannalaBrandt


def _roundtrip_max_err(cam: KannalaBrandt, n: int = 257) -> float:
    ys, xs = torch.meshgrid(
        torch.linspace(0, cam.height - 1, n),
        torch.linspace(0, cam.width - 1, n), indexing="ij")
    uv = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
    rays = cam.unproject(uv)
    theta = torch.acos(rays[:, 2].clamp(-1, 1))
    inside = theta <= cam.theta_max
    rt = cam.project(rays)
    return float((rt - uv).norm(dim=-1)[inside].max())


def test_aria_kb4_roundtrip_subpixel():
    # Aria 214-1 coefficients at a 252-px working grid (the H5 test setup
    # that exposed the bug).
    cam = KannalaBrandt(fx=43.7, fy=43.7, cx=125.6, cy=125.9,
                        width=252, height=252,
                        k=(0.3852, -0.4442, 0.5591, -0.3254),
                        theta_max=math.radians(54.83))
    assert _roundtrip_max_err(cam) < 1e-3


def test_scannetpp_like_roundtrip_subpixel():
    # An OPENCV_FISHEYE-style calibration with milder coefficients.
    cam = KannalaBrandt(fx=200.0, fy=200.0, cx=251.5, cy=167.5,
                        width=504, height=336,
                        k=(-0.03, 0.02, -0.01, 0.002),
                        theta_max=math.radians(84.8))
    assert _roundtrip_max_err(cam) < 1e-3
