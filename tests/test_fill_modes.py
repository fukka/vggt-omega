# Copyright (c) 2026.
"""Every fill arm must actually fill, and the one that must invent nothing must not.

The fill ladder answers the question the whole idea hangs on: how much of the
oracle's gain does a fill that INVENTS NOTHING already capture? That question is
only answerable if each rung does what its name says, and the failure mode is
silent -- a mode that fills nothing returns successfully and scores exactly like
black, which reads as a clean "this fill does not help" rather than as a bug.
``mirror`` did exactly that on this repo's geometry before it was made to refuse.

The holes here are the two this repo actually has, and neither is a border band:
the fisheye's is four corners past the KB4 turnover, the rectified one is four
wedges outside the cone. Both are ENCLOSED by valid pixels, which is what breaks
reflection-about-the-bounding-box and what makes nearest-valid well defined.

CPU-only, no data, no weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from finetune.data.fill import FILL_MODES, apply_fill

S = 96


def _corner_hole():
    """Four corners invalid — the fisheye disc's complement, in miniature."""
    yy, xx = np.mgrid[:S, :S]
    r = np.hypot(yy - (S - 1) / 2.0, xx - (S - 1) / 2.0)
    return r <= S * 0.48


def _border_hole():
    """Top and bottom bands invalid — the shape reflection is actually for."""
    m = np.ones((S, S), bool)
    m[: S // 8] = False
    m[-S // 8:] = False
    return m


def _image(rng):
    """Textured, so a fill that copies real pixels is distinguishable from one
    that invents them: a flat image would make every mode look identical."""
    img = rng.random((S, S, 3), dtype=np.float32)
    img[:, : S // 2] *= 0.4                      # a dark half, so 'nearest valid is
    return np.clip(img, 0.0, 1.0)                # already dark' is exercised


def _invented_fraction(filled, img, valid):
    """Fraction of filled pixels whose exact 8-bit triple appears nowhere valid."""
    q = lambda a: (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    palette = set(map(tuple, q(img[valid]).tolist()))
    out = q(filled[~valid])
    return float(np.mean([tuple(x) not in palette for x in out]))


@pytest.mark.parametrize("mode", [m for m in FILL_MODES if m not in ("oracle", "mirror")])
def test_fill_changes_the_hole_and_only_the_hole(mode):
    rng = np.random.default_rng(0)
    valid = _corner_hole()
    img = _image(rng)
    img[~valid] = 0.0                            # the masked arm holds zeros
    out = apply_fill(img, valid, mode)

    assert out.shape == img.shape and out.dtype == np.float32
    assert np.allclose(out[valid], img[valid]), f"{mode} touched valid pixels"
    if mode == "black":
        assert np.allclose(out[~valid], 0.0)
        return
    changed = float((np.abs(out[~valid] - img[~valid]).max(-1) > 1e-6).mean())
    assert changed > 0.5, (
        f"{mode} left {1 - changed:.1%} of the hole unchanged; an arm that does "
        f"not fill scores like black and reads as a clean negative result")


def test_replicate_invents_nothing():
    """The rung the decision hangs on: if the curve saturates at a fill that only
    copies real pixels, a generative filler is buying whatever is left."""
    rng = np.random.default_rng(1)
    valid = _corner_hole()
    img = _image(rng)
    img[~valid] = 0.0
    out = apply_fill(img, valid, "replicate")
    assert _invented_fraction(out, img, valid) == 0.0


def test_mirror_refuses_an_enclosed_hole():
    """Reflection about the valid region's bounding box cannot reach a hole that
    the valid region surrounds. It used to return the input unchanged."""
    rng = np.random.default_rng(2)
    valid = _corner_hole()
    img = _image(rng)
    img[~valid] = 0.0
    with pytest.raises(ValueError, match="cannot fill this hole"):
        apply_fill(img, valid, "mirror")


def test_mirror_still_fills_a_border_band():
    """...and must keep working for the shape it is for, or the guard is a ban."""
    rng = np.random.default_rng(3)
    valid = _border_hole()
    img = _image(rng)
    img[~valid] = 0.0
    out = apply_fill(img, valid, "mirror")
    changed = float((np.abs(out[~valid] - img[~valid]).max(-1) > 1e-6).mean())
    assert changed > 0.5
    assert _invented_fraction(out, img, valid) == 0.0


def test_unknown_mode_is_rejected():
    valid = _corner_hole()
    img = np.zeros((S, S, 3), np.float32)
    with pytest.raises(ValueError, match="unknown fill mode"):
        apply_fill(img, valid, "diffusion")
