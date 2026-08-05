"""Curriculum sampling and panorama -> lens synthesis."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cam3r.cameras import Pinhole, aria_214_1_kb4
from cam3r.data import (
    CurriculumSampler,
    TwoViewSource,
    look_at_rotation,
    random_camera_for,
    synthesize_view,
)


class _FakeDataset:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return {"i": i}


def _sources(hetero=False):
    return [
        TwoViewSource(_FakeDataset(10), kind="kb4", name="adt", supports_heterogeneous=hetero),
        TwoViewSource(_FakeDataset(20), kind="erp", name="2d3ds", supports_heterogeneous=hetero),
        TwoViewSource(_FakeDataset(30), kind="pinhole", name="megadepth"),
    ]


# ----------------------------------------------------------------- curriculum

def test_phase_one_never_asks_for_cross_lens_pairs():
    s = CurriculumSampler(_sources(hetero=True), phase=1, seed=0)
    assert not s.can_do_heterogeneous()
    assert all(not s.sample().heterogeneous for _ in range(200))


def test_phase_two_produces_cross_lens_draws():
    s = CurriculumSampler(_sources(hetero=True), phase=2, hetero_ratio=0.5, seed=0)
    draws = [s.sample() for _ in range(300)]
    hetero = [d for d in draws if d.heterogeneous]
    assert hetero, "phase 2 produced no heterogeneous draws"
    for d in hetero:
        assert s.sources[d.source].supports_heterogeneous


def test_heterogeneous_needs_a_source_that_can_render_both_lenses():
    """Two independent datasets cannot form a pair: no shared scene, no GT pose."""
    s = CurriculumSampler(_sources(hetero=False), phase=2, seed=0)
    assert not s.can_do_heterogeneous()
    assert all(not s.sample().heterogeneous for _ in range(100))


def test_indices_are_always_in_range():
    s = CurriculumSampler(_sources(hetero=True), phase=2, seed=1)
    for _ in range(300):
        d = s.sample()
        assert 0 <= d.index < len(s.sources[d.source])


def test_sampler_is_reproducible():
    first = CurriculumSampler(_sources(hetero=True), phase=2, seed=7)
    second = CurriculumSampler(_sources(hetero=True), phase=2, seed=7)
    assert [first.sample() for _ in range(20)] == [second.sample() for _ in range(20)]


def test_different_seeds_diverge():
    a = CurriculumSampler(_sources(hetero=True), phase=2, seed=1)
    b = CurriculumSampler(_sources(hetero=True), phase=2, seed=2)
    assert [a.sample() for _ in range(20)] != [b.sample() for _ in range(20)]


def test_rejects_bad_phase():
    with pytest.raises(ValueError):
        CurriculumSampler(_sources(), phase=3)


# ----------------------------------------------------------------- synthesis

def _test_panorama(Hp: int = 64, Wp: int = 128) -> torch.Tensor:
    """A panorama whose colour encodes ERP position, so resampling is checkable."""
    ys, xs = torch.meshgrid(torch.arange(Hp, dtype=torch.float32),
                            torch.arange(Wp, dtype=torch.float32), indexing="ij")
    return torch.stack([xs / Wp, ys / Hp, torch.zeros_like(xs)], dim=0)


def test_synthesized_view_has_the_right_shapes_and_unit_rays():
    out = synthesize_view(_test_panorama(), Pinhole.from_fov(32, 32, 90.0), (32, 32))
    assert out["image"].shape == (3, 32, 32)
    assert out["rays"].shape == (3, 32, 32)
    assert torch.allclose(out["rays"].norm(dim=0), torch.ones(32, 32), atol=1e-5)
    assert out["valid"].shape == (32, 32)


def test_centre_pixel_samples_the_panorama_centre():
    """Identity rotation looks along +z, which is the middle column of the ERP."""
    pano = _test_panorama()
    cam = Pinhole(f_ := 40.0, f_, 16.0, 16.0)
    out = synthesize_view(pano, cam, (33, 33))
    got = out["image"][:, 16, 16]
    assert abs(float(got[0]) - 0.5) < 0.02, f"longitude {float(got[0]):.3f}, expected ~0.5"
    assert abs(float(got[1]) - 0.5) < 0.02, f"latitude {float(got[1]):.3f}, expected ~0.5"


def test_rotation_shifts_the_sampled_longitude():
    pano = _test_panorama()
    cam = Pinhole(40.0, 40.0, 16.0, 16.0)
    base = synthesize_view(pano, cam, (33, 33))["image"][0, 16, 16]
    turned = synthesize_view(pano, cam, (33, 33),
                             rotation=look_at_rotation(math.radians(90.0), 0.0))["image"][0, 16, 16]
    # 90 deg of yaw is a quarter of the panorama width.
    assert abs(abs(float(turned) - float(base)) - 0.25) < 0.03


def test_depth_becomes_a_radial_pointmap():
    pano = _test_panorama()
    depth = torch.full((64, 128), 3.0)
    out = synthesize_view(pano, Pinhole.from_fov(32, 32, 90.0), (32, 32), depth=depth)
    assert torch.allclose(out["points"], out["rays"] * 3.0, atol=1e-5)
    assert torch.allclose(out["points"].norm(dim=0), torch.full((32, 32), 3.0), atol=1e-4)
    assert bool(out["valid"].all())


def test_zero_depth_is_marked_invalid():
    depth = torch.full((64, 128), 3.0)
    depth[:32] = 0.0
    out = synthesize_view(_test_panorama(), Pinhole.from_fov(32, 32, 120.0), (32, 32), depth=depth)
    assert not bool(out["valid"].all()) and bool(out["valid"].any())


def test_fisheye_synthesis_keeps_the_cone_mask():
    cam = aria_214_1_kb4(48, 48)
    out = synthesize_view(_test_panorama(), cam, (48, 48))
    assert not bool(out["valid"][0, 0]), "corner is outside the imaged cone"
    assert bool(out["valid"][24, 24])


def test_random_cameras_are_of_the_requested_family():
    import random

    rng = random.Random(0)
    assert random_camera_for("pinhole", 32, 32, rng).kind == "pinhole"
    assert random_camera_for("kb4", 32, 32, rng).kind == "kb4"
    assert random_camera_for("erp", 32, 32, rng).kind == "erp"
    with pytest.raises(ValueError):
        random_camera_for("nope", 32, 32, rng)


def test_random_fisheye_covers_a_wide_field():
    import random

    cam = random_camera_for("kb4", 96, 96, random.Random(3))
    rays, valid = cam.ray_field(96, 96)
    theta = torch.arccos(rays[..., 2].clamp(-1, 1))
    assert math.degrees(float(theta[valid].max())) > 60.0
