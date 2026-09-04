"""Every ADT loader in this repo must agree on which way is up.

On 2026-09-03/04 three were in play at once, each internally self-consistent,
so nothing ever failed:

    AriaLocalPairs (H1/H5/H12/H14/H15/H9)   k=0   sideways
    raytun3r.data.ADTSequence (depthfisheye) k=1   upside down
    fovbench.run                             k=3   upright

Measured on seq136 with frozen DA3-Small and the prediction rotated back, k=3
is better than k=0 by 64% whole-image and 71% near-rim, and k=1 is the worst of
the four -- worse than not turning at all. The near_rim/centre ratio, which the
whole diagnosis line rests on, is 4.30x at k=0 and 2.00x at k=3.

This test costs nothing and needs no data: it reads the three declared defaults
and asserts they resolve to the same quarter turn. A future edit that moves one
of them fails here instead of quietly producing a plausible number.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "autoresearch" / "experiments" / "common"))

#: `ADTSequence.rotation` names how far the stored frame is from upright; this
#: is the map it applies to get the quarter turns that undo it.
_ADT_DEG_TO_K = {0: 0, 90: 3, 180: 2, 270: 1}


def _default(fn, name):
    p = inspect.signature(fn).parameters.get(name)
    assert p is not None and p.default is not inspect._empty, \
        f"{fn.__qualname__} no longer declares a default for {name!r}"
    return p.default


def test_the_upright_turn_is_three_quarter_turns():
    import upright as U
    assert U.UPRIGHT_K == 3


def test_adtsequence_and_fovbench_and_upright_all_agree():
    from raytun3r.data import ADTSequence
    import upright as U

    adt_deg = _default(ADTSequence.__init__, "rotation")
    assert adt_deg in _ADT_DEG_TO_K, f"unmapped rotation default {adt_deg}"
    adt_k = _ADT_DEG_TO_K[adt_deg]
    assert adt_k == U.UPRIGHT_K, (
        f"ADTSequence(rotation={adt_deg}) undoes with k={adt_k}, but upright is "
        f"k={U.UPRIGHT_K}. k=1 in particular is the WORST of the four "
        f"orientations, not a near miss.")


def test_fovbench_rotation_k_agrees():
    import fovbench.run as fr
    import upright as U
    ks = {name: _default(fn, "rotation_k")
          for name, fn in vars(fr).items()
          if inspect.isfunction(fn) and "rotation_k" in inspect.signature(fn).parameters}
    assert ks, "fovbench.run no longer takes rotation_k anywhere"
    for name, k in ks.items():
        assert k == U.UPRIGHT_K, f"fovbench.run.{name} defaults rotation_k={k}"
