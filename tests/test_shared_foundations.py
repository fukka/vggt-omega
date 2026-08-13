# Copyright (c) 2026.
"""The two things both experiments stand on, pinned against outside derivations.

``finetune/aria_calibration.py`` and ``finetune/eval/manifest.py`` exist because
each was written more than once — the lens three times, the digest rule twice —
and in the lens's case the copies disagreed. A shared module only helps if it is
also *right*, so nothing below checks it against itself:

* the storage rotation is derived from ``numpy.rot90`` on a labelled grid, which
  is the operation every ADT loader in this repo actually applies;
* the digest is checked against the literal implementation both splits carried
  before it was extracted, so the extraction cannot have moved a published
  digest.

CPU-only, no data, no weights.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import aria_calibration as A          # noqa: E402
from finetune.eval import manifest as M             # noqa: E402


# --------------------------------------------------------------------------- #
# The lens
# --------------------------------------------------------------------------- #

def test_the_storage_rotation_is_the_one_numpy_actually_performs():
    """``np.rot90(frame, k=3)`` is what the loaders apply; ask numpy where a
    pixel lands rather than trusting any of the three formulas that existed.

    This is the assertion the old arrangement could not make. ``rectify.py`` had
    ``cx' = W - cy`` and the other two had ``cx' = (H-1) - cy``; both are
    plausible on sight, and only the grid settles it.
    """
    H = W = 9
    src = np.arange(H * W).reshape(H, W)
    dst = np.rot90(src, k=3)
    for v in range(H):
        for u in range(W):
            i, j = np.argwhere(dst == src[v, u])[0]
            assert (j, i) == ((H - 1) - v, u), (
                f"src (u={u}, v={v}) landed at (u'={j}, v'={i}), "
                f"not ((H-1)-v, u) = ({(H - 1) - v}, {u})")

    # ...and that is what `intrinsics` implements, for the principal point.
    fx, fy, cx, cy = A.intrinsics(H, W, rotated=True)
    ux, uy, ucx, ucy = A.intrinsics(H, W, rotated=False)
    assert cx == pytest.approx((H - 1) - ucy)
    assert cy == pytest.approx(ucx)
    assert (fx, fy) == (uy, ux), "a quarter turn swaps the focal axes"


def test_the_off_by_one_that_used_to_be_in_rectify_is_gone():
    """``W - cy`` sits exactly one pixel right of ``(H-1) - cy``, at every size.

    Kept as its own test because the failure it guards has no symptom: both arms
    still produce a picture, the depths still line up with their own images, and
    the only trace is that two arms of one experiment sample the lens a pixel
    apart.
    """
    for n in (1408, 896, 704, 518):
        _, _, cx, _ = A.intrinsics(n, n, rotated=True)
        wrong = n - A.CY_NATIVE * (n / A.NATIVE)
        assert wrong - cx == pytest.approx(1.0), n


@pytest.mark.parametrize("size", [1408, 896, 704, 518])
def test_all_three_consumers_describe_one_lens(size):
    """The point of the module: the FOV experiment's fisheye arm, its rectified
    arm and the baselines must not be able to disagree about this camera."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "VGGT-360-fisheye"))
    from utils.fisheye_cam import aria_intrinsics as fov_fisheye
    from finetune.eval.baselines.aria_fisheye import aria_intrinsics as baselines
    from finetune.data.rectify import FisheyeRectifier

    a = fov_fisheye(size, size)
    b = baselines(size, size)
    K, _, _ = FisheyeRectifier("aria-214-1")._intrinsics(size, size)

    assert (a.fx, a.fy, a.cx, a.cy) == (b.fx, b.fy, b.cx, b.cy)
    assert K[0, 0] == pytest.approx(a.fx)
    assert K[1, 1] == pytest.approx(a.fy)
    assert K[0, 2] == pytest.approx(a.cx)
    assert K[1, 2] == pytest.approx(a.cy)
    assert tuple(a.k) == A.KB4


def test_the_radial_coefficients_are_never_rescaled_or_rotated():
    """KB4 acts on an angle, so it is invariant to both. A resize or a turn that
    touched them would be describing a different lens."""
    for size in (1408, 518):
        for rot in (True, False):
            fx, fy, cx, cy = A.intrinsics(size, size, rotated=rot)
            assert A.KB4 == (0.3852, -0.4442, 0.5591, -0.3254)
            assert fx > 0 and fy > 0


# --------------------------------------------------------------------------- #
# The comparability token
# --------------------------------------------------------------------------- #

def _digest_as_both_splits_used_to_compute_it(protocol, keys):
    """The literal body both ``Split.digest`` properties carried, inlined."""
    h = hashlib.sha1(protocol.encode())
    for k in keys:
        h.update(k.encode())
    return h.hexdigest()[:12]


def test_extracting_the_digest_rule_did_not_move_a_single_digest():
    """Every published digest — 601fcb22767e, 8ca25fd0ebd2, fcc6c600f83b — was
    produced by the inlined version. If the shared one differs by so much as a
    character, every result on the `results` branch stops matching the runs that
    made it."""
    for protocol in ("adt-fov-v1", "egosynth-slam-v1"):
        for keys in ([], ["a"], [f"seq{i}:{i}" for i in range(200)]):
            assert M.digest(protocol, keys) == \
                _digest_as_both_splits_used_to_compute_it(protocol, keys)


def test_the_protocol_token_separates_the_two_experiments():
    """The same frame list under two protocols must never collide — that is what
    stops a run of one experiment being read as a run of the other."""
    keys = [f"take/clip:{i}" for i in range(50)]
    assert M.digest("adt-fov-v1", keys) != M.digest("egosynth-slam-v1", keys)


def test_reordering_the_frames_changes_the_digest():
    """Order is part of the identity: pooling is float addition, which is not
    associative, so two runs over the same frames in a different order are not
    guaranteed to produce the same digits."""
    keys = [f"f{i}" for i in range(20)]
    assert M.digest("adt-fov-v1", keys) != M.digest("adt-fov-v1", keys[::-1])


def test_even_spacing_covers_the_span_and_never_repeats_a_frame():
    """A prefix would sample one room and one activity; a repeat would weight
    itself double in every pooled number."""
    items = list(range(121))
    got = M.evenly_spaced(items, 25)
    assert len(got) == len(set(got)) == 25
    assert got[0] == 0 and got[-1] == 120
    assert got == sorted(got)
    assert M.evenly_spaced(items, 500) == items          # asks for more than it has
    assert M.evenly_spaced(items, 1) == [60]             # the middle, not the first
    assert M.evenly_spaced([], 10) == [] and M.evenly_spaced(items, 0) == []


def test_both_splits_now_share_that_one_rule():
    """Stated as a dependency rather than as a coincidence: the two ``Split``
    classes stay separate — different frames, different protocols, different
    build — and only the rule underneath them is shared."""
    from fovbench import split as F
    from slambench import split as S

    assert F._evenly_spaced is M.evenly_spaced
    assert S._evenly_spaced is M.evenly_spaced
    assert F.PROTOCOL != S.PROTOCOL
    assert F.Split is not S.Split and F.Frame is not S.Frame
