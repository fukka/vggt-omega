# Copyright (c) 2026.
"""The multi-frame context: what the model is handed, and what is scored.

The sweep's whole claim is that a 1-frame and a 10-frame arm measure the
**identical points** and differ only in the evidence the model had. Three things
have to hold for that claim to survive, and each has a test here:

* the window is built the way the docstring says — preceding the target, at the
  requested spacing, inside the clip, never repeating a frame;
* the window reaches the model intact, and every frame of it goes through the
  baseline's own lens handling rather than only the scored one;
* the arms are scored on the same points, and a model that ignores its context
  therefore reports the identical number.

The last of those is the one that would fail silently. A model handed a context
it cannot use, or a plumbing bug that scored the wrong frame of the window, both
produce a full and plausible table.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from slambench import baselines as B
from slambench import data as D
from slambench import models as M
from slambench import run as RUN
from slambench import split as S

SAMPLE = os.environ.get(
    "EGOSYNTH_SAMPLE", os.path.expanduser("~/Desktop/ADT/ego-synth-5b-sample"))
needs_sample = pytest.mark.skipif(
    not os.path.isdir(SAMPLE),
    reason="ego-synth sample not staged (licensed; see the data card)")


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #

def test_the_window_precedes_the_target():
    """What a live camera would have: the frames before the one being asked
    about, not a window centred on it."""
    idx, tgt = S.context_window(121, 60, 5, 1)
    assert idx == [56, 57, 58, 59, 60]
    assert idx[tgt] == 60


def test_the_stride_spaces_the_window_without_moving_the_target():
    idx, tgt = S.context_window(121, 60, 4, 5)
    assert idx == [45, 50, 55, 60]
    assert idx[tgt] == 60


def test_a_window_at_the_start_of_a_clip_shifts_rather_than_repeats():
    """Clamping would hand the model the same frame several times, and a
    repeated frame is not evidence — a multi-view model may even read it as a
    stationary camera, which is a claim about the scene rather than a gap."""
    idx, tgt = S.context_window(121, 1, 5, 1)
    assert len(idx) == len(set(idx)) == 5, idx
    assert idx == [0, 1, 2, 3, 4]
    assert idx[tgt] == 1, "the target must still be the frame that is scored"


def test_a_window_at_the_end_of_a_clip_stays_inside_it():
    """A clip is its own mp4; an index past the end is a decode failure, and
    ``decode_frames`` would raise on it rather than score anything."""
    idx, tgt = S.context_window(121, 120, 5, 1)
    assert max(idx) <= 120 and idx == [116, 117, 118, 119, 120]
    assert idx[tgt] == 120


def test_a_short_clip_gives_back_what_it_has_and_keeps_the_target():
    for pool, i, n in ((3, 2, 10), (1, 0, 5), (4, 0, 3), (2, 1, 8)):
        idx, tgt = S.context_window(pool, i, n, 1)
        assert idx, (pool, i, n)
        assert len(idx) == len(set(idx)), (pool, i, n, idx)
        assert max(idx) <= max(0, pool - 1), (pool, i, n, idx)
        assert idx[tgt] == i, (pool, i, n, idx)


def test_one_frame_is_the_window_of_one():
    assert S.context_window(121, 60, 1, 1) == ([60], 0)
    assert S.context_window(0, 7, 4, 1) == ([7], 0)      # unknown clip length


# --------------------------------------------------------------------------- #
# The comparability guarantee
# --------------------------------------------------------------------------- #

def _frames(n=4):
    return [S.Frame(dataset="aea", take="t", clip="0", index=i,
                    npz=f"{i}.npz", video="v.mp4", clip_frames=121)
            for i in range(n)]


def test_the_context_is_not_in_the_digest():
    """The digest says "these runs scored the same points". A 1-frame and a
    10-frame arm *do* score the same points — that is the entire comparison —
    so folding the context in would give them different digests and make the
    harness refuse the only thing the sweep exists to measure."""
    a = S.Split(root="/r", frames=_frames())
    b = S.Split(root="/r", frames=_frames())
    assert a.digest == b.digest
    # and the clip length, which context needs, is not in it either
    for f in b.frames:
        f.clip_frames = 7
    assert a.digest == b.digest


def test_a_manifest_without_clip_lengths_refuses_a_context_run(tmp_path):
    """Guessing the clip length from the frames present would shrink every
    window silently: the split holds 25 of a clip's ~121 frames."""
    sp = S.Split(root="/r", frames=_frames())
    for f in sp.frames:
        f.clip_frames = 0                      # a manifest predating context
    p = str(tmp_path / "manifest.json")
    sp.save(p)
    a = RUN.build_parser().parse_args(
        ["--manifest", p, "--models", M.ANALYTIC, "--baselines", "raw",
         "--context-frames", "5", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as e:
        RUN.run(a)
    assert "clip length" in str(e.value)
    # ... and the same manifest is fine for a 1-frame run: it gets past this
    # check and fails later, on the frames it cannot read.
    a.context_frames = "1"
    with pytest.raises(SystemExit) as e2:
        RUN.run(a)
    assert "clip length" not in str(e2.value)


def test_a_bad_context_spec_is_refused():
    for bad in ("0", "-1", "", "1,0"):
        with pytest.raises(SystemExit):
            RUN._context_sizes(bad)
    assert RUN._context_sizes("10,1,5,3,5") == [1, 3, 5, 10]


# --------------------------------------------------------------------------- #
# What reaches the model
# --------------------------------------------------------------------------- #

class _Recorder:
    """A model that records the stack it was handed and answers for the target."""

    key = "recorder"
    input_size = D.RES
    align_mode = "scale_shift"
    supports_context = True

    def __init__(self):
        self.stacks = []

    def predict(self, rgb_u8, gt=None):
        return self.predict_stack([rgb_u8], 0, gt)

    def predict_stack(self, frames, target=-1, gt=None):
        self.stacks.append([np.asarray(f).copy() for f in frames])
        return np.full(np.asarray(frames[target]).shape[:2], 2.0, np.float32)


def _pts(n=8):
    return D.FramePoints(u=np.linspace(10, 800, n).astype(np.float32),
                         v=np.linspace(10, 800, n).astype(np.float32),
                         d=np.full(n, 2.0, np.float32),
                         inv_dist_std=np.zeros(n, np.float32),
                         dist_std=np.zeros(n, np.float32))


def _stack(k, value):
    """``k`` frames, each a constant so the recorder can identify them."""
    return [np.full((D.RES, D.RES, 3), v, np.uint8) for v in value[:k]]


def test_the_raw_baseline_hands_over_the_whole_window():
    m = _Recorder()
    bl = B.RawBaseline(m, input_size=D.RES)
    bl.predict(_stack(4, [10, 20, 30, 40]), _pts(), target=3)
    assert len(m.stacks) == 1 and len(m.stacks[0]) == 4
    assert [int(f[0, 0, 0]) for f in m.stacks[0]] == [10, 20, 30, 40]


def test_one_frame_is_not_iterated_into_896_slivers():
    """An RGB frame is itself a 3-d array, so a bare ``list(frame)`` would hand
    the model 896 one-pixel-tall images and every one of them would 'work'."""
    m = _Recorder()
    B.RawBaseline(m, input_size=D.RES).predict(
        np.full((D.RES, D.RES, 3), 7, np.uint8), _pts())
    assert len(m.stacks[0]) == 1
    assert m.stacks[0][0].shape == (D.RES, D.RES, 3)


def test_rect_derect_rectifies_every_frame_of_the_window():
    """Not only the scored one. A model handed one pinhole render among nine
    fisheye frames would be asked to match features across two different lenses,
    and whatever it did next would be about that mismatch."""
    from slambench.camera import Fisheye624
    prm = (300.0, 447.5, 447.5) + (0.0,) * 12
    cam = Fisheye624(prm, D.RES, D.RES, 0)
    m = _Recorder()
    bl = B.RectDerectBaseline(m, cam, fov_deg=110.0, rect_size=D.RES)

    # a ramp, offset per frame, so each is identifiable and none survives
    # rectification unchanged
    ramp = np.tile(np.linspace(0, 200, D.RES, dtype=np.uint8), (D.RES, 1))
    window = [np.repeat((ramp + k * 10)[..., None], 3, axis=2) for k in range(3)]
    bl.predict(window, _pts(), target=2)

    seen = m.stacks[0]
    assert len(seen) == 3
    for k, (got, src) in enumerate(zip(seen, window)):
        assert np.array_equal(got, bl.rectify(src)[0]), f"frame {k} of the window"
        assert not np.array_equal(got, src), "rectification changed nothing"


def test_a_monocular_model_refuses_a_context_rather_than_ignoring_it():
    """The failure this prevents is not a crash, it is a table: scoring the
    target alone fills every row and reads 'context does not help' when nothing
    was tried."""
    mono = M.Model(key="dav2_large", family="DAv2", size="L",
                   align_mode="disparity_scale_shift", input_size=518,
                   supports_context=False,
                   _predict=lambda rgb, gt=None: np.ones(rgb.shape[:2], np.float32))
    assert mono.predict_stack([np.zeros((8, 8, 3), np.uint8)]).shape == (8, 8)
    with pytest.raises(SystemExit) as e:
        mono.predict_stack([np.zeros((8, 8, 3), np.uint8)] * 3, target=2)
    assert "monocular" in str(e.value) and "context does not help" in str(e.value)


def test_the_run_refuses_a_monocular_model_before_it_loads_anything(monkeypatch):
    """Before, not after. Finding out on the third model that the second cannot
    take a context has already spent two model-loads to learn it — and
    ``takes_context`` answers from a class attribute, with no weights."""
    assert M.takes_context("dav2_large") is False        # the real registry
    monkeypatch.setattr(M, "available", lambda keys: (list(keys), []))
    a = RUN.build_parser().parse_args(
        ["--models", "vggt_omega,dav2_large", "--baselines", "raw",
         "--context-frames", "1,3,5,10", "--egosynth-root", "/nonexistent",
         "--out", "/tmp/never"])
    with pytest.raises(SystemExit) as e:
        RUN.run(a)
    msg = str(e.value)
    assert "dav2_large" in msg and "monocular" in msg
    assert "/nonexistent" not in msg, "it got as far as reading the release"


def test_the_standins_answer_for_the_target_frame():
    """They ignore the context on purpose — they are harness probes, and their
    score is meaningless by construction — but they must answer for the frame
    that is actually being scored, or the sweep would test nothing."""
    m = M.load_model(M.ANALYTIC, None)
    dark, bright = (np.full((32, 32, 3), v, np.uint8) for v in (0, 255))
    got = m.predict_stack([dark, dark, bright], target=2)
    assert np.allclose(got, m.predict(bright), atol=1e-6)
    assert not np.allclose(got, m.predict(dark), atol=1e-2)


def test_the_analytic_standin_does_not_depend_on_call_order():
    """It is asked for the same frame once per arm of the sweep. With a running
    generator the arms would differ by the stand-in's own noise."""
    m = M.load_model(M.ANALYTIC, None)
    f = np.full((16, 16, 3), 120, np.uint8)
    assert np.array_equal(m.predict(f), m.predict(f))


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@needs_sample
def test_every_context_arm_scores_the_identical_points(tmp_path):
    """The claim the sweep rests on, measured rather than asserted: a model that
    ignores its context reports the **same numbers** at 1, 3, 5 and 10 frames.
    Anything that scored the wrong frame of the window, or fitted the alignment
    over a different support, moves these."""
    a = RUN.build_parser().parse_args([])
    a.egosynth_root, a.out, a.device = SAMPLE, str(tmp_path), "cpu"
    a.takes, a.n_frames, a.log_every = 1, 3, 1000
    a.models, a.baselines, a.datasets = M.ANALYTIC, "raw", "aea"
    a.context_frames = "1,3,5,10"
    out = RUN.run(a)

    by = {r["context"]: r for r in out["runs"]}
    assert sorted(by) == [1, 3, 5, 10], sorted(by)
    one = by[1]
    for n, r in by.items():
        for col in ("AbsRel", "delta1", "RMSE", "gt_median", "n_frames",
                    "n_points_total"):
            assert r[col] == one[col], f"context {n} moved {col}"
    assert out["config"]["context_frames"] == [1, 3, 5, 10]
