# Copyright (c) 2026.
"""The ADT-FOV test split must be a *fixed* set of frames, not a query.

Every property here exists because the benchmark's headline comparison is
synthetic-vs-real on the same scene: if the two streams score different frames,
the difference between them is partly which frames were picked, and nothing in
the output would say so.

CPU-only; builds a fake ADT tree on disk.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from fovbench import split as S  # noqa: E402


def _write_seq(root, name, *, depth_ids, synth_ids, real_ids):
    seq = os.path.join(root, name)
    for sub, ids, ext in (("depth_npy", depth_ids, ".npy"),
                          ("videos_synthetic", synth_ids, ".jpg"),
                          ("videos_rgb", real_ids, ".jpg")):
        if ids is None:                    # stream absent entirely
            continue
        d = os.path.join(seq, sub)
        os.makedirs(d, exist_ok=True)
        for i in ids:
            p = os.path.join(d, f"frame_{i:06d}_{1000 + i}{ext}")
            if ext == ".npy":
                np.save(p, np.zeros((4, 4), np.uint16))
            else:
                open(p, "wb").write(b"x")
    return seq


@pytest.fixture()
def adt(tmp_path):
    root = str(tmp_path / "adt")
    # complete sequence
    _write_seq(root, "seqA", depth_ids=range(0, 40), synth_ids=range(0, 40),
               real_ids=range(0, 40))
    # real stream misses a few frames the synthetic stream has
    _write_seq(root, "seqB", depth_ids=range(0, 20), synth_ids=range(0, 20),
               real_ids=[i for i in range(0, 20) if i % 5])
    # no real stream at all
    _write_seq(root, "seqC", depth_ids=range(0, 20), synth_ids=range(0, 20),
               real_ids=None)
    return root


def test_split_scores_identical_frames_in_both_streams(adt):
    m = S.build_split(adt, n_frames=8)
    for f in m.frames:
        assert os.path.isfile(f.rgb["synthetic"])
        assert os.path.isfile(f.rgb["real"])
        assert os.path.isfile(f.depth)


def test_split_drops_a_sequence_missing_a_stream(adt):
    m = S.build_split(adt, n_frames=8)
    assert {f.seq for f in m.frames} == {"seqA", "seqB"}


def test_split_drops_frames_the_real_stream_lacks(adt):
    """seqB's real stream has no frame 0, 5, 10, 15 — those ids must not be in
    the manifest at all, for either stream."""
    m = S.build_split(adt, n_frames=64)
    b_ids = {int(f.frame_id) for f in m.frames if f.seq == "seqB"}
    assert b_ids and not (b_ids & {0, 5, 10, 15})


def test_split_is_deterministic(adt):
    a = S.build_split(adt, n_frames=8)
    b = S.build_split(adt, n_frames=8)
    assert a.digest == b.digest
    assert [f.key for f in a.frames] == [f.key for f in b.frames]


def test_split_digest_changes_when_the_frame_set_changes(adt):
    assert S.build_split(adt, n_frames=8).digest != \
        S.build_split(adt, n_frames=12).digest


def test_split_spreads_frames_over_the_sequence_rather_than_taking_a_prefix(adt):
    """A 40-frame sequence sampled 4 times must not return frames 0-3: adjacent
    ADT frames are near-duplicates, and a prefix would report the variance of
    one instant as the variance of a sequence."""
    m = S.build_split(adt, n_frames=4)
    ids = sorted(int(f.frame_id) for f in m.frames if f.seq == "seqA")
    assert max(ids) - min(ids) > 20


def test_split_caps_frames_per_sequence_not_in_total(adt):
    m = S.build_split(adt, n_frames=5)
    per_seq = {}
    for f in m.frames:
        per_seq[f.seq] = per_seq.get(f.seq, 0) + 1
    assert per_seq == {"seqA": 5, "seqB": 5}


def test_split_round_trips_through_json(adt, tmp_path):
    m = S.build_split(adt, n_frames=6)
    p = str(tmp_path / "manifest.json")
    m.save(p)
    back = S.Split.load(p)
    assert back.digest == m.digest
    assert [f.key for f in back.frames] == [f.key for f in m.frames]


def test_split_refuses_an_empty_root(tmp_path):
    with pytest.raises(SystemExit):
        S.build_split(str(tmp_path), n_frames=4)


def test_split_records_the_streams_it_requires(adt):
    m = S.build_split(adt, n_frames=4)
    assert m.streams == {"synthetic": "videos_synthetic", "real": "videos_rgb"}


# --------------------------------------------------------------------------- #
# Multi-frame context
# --------------------------------------------------------------------------- #

from fovbench.split import _context_window  # noqa: E402


def test_a_context_window_precedes_the_target_and_ends_on_it():
    """What a live camera would have: the frames before the one being asked
    about, at the requested spacing, with the target last."""
    idx, tgt = _context_window(100, 50, 5, 1)
    assert idx == [46, 47, 48, 49, 50]
    assert idx[tgt] == 50
    idx, tgt = _context_window(100, 50, 5, 10)
    assert idx == [10, 20, 30, 40, 50]
    assert idx[tgt] == 50


def test_one_frame_of_context_is_just_the_target():
    assert _context_window(100, 7, 1, 1) == ([7], 0)


def test_a_window_at_the_start_shifts_rather_than_repeating_a_frame():
    """Clamping would hand the model the same frame several times, and a
    repeated frame is not evidence — it would make a 5-frame run look like a
    1-frame run wearing a costume."""
    idx, tgt = _context_window(100, 2, 5, 1)
    assert len(set(idx)) == len(idx) == 5
    assert idx[tgt] == 2


def test_the_target_is_always_inside_its_own_context():
    for n_pool in (5, 20, 100):
        for i in range(n_pool):
            for n, stride in ((1, 1), (5, 1), (10, 1), (5, 10), (10, 10)):
                idx, tgt = _context_window(n_pool, i, n, stride)
                assert idx[tgt] == i, (n_pool, i, n, stride, idx, tgt)
                assert len(set(idx)) == len(idx)
                assert all(0 <= k < n_pool for k in idx)


def test_context_does_not_change_the_digest():
    """The digest says "these runs scored the same pixels", and a 1-frame and a
    10-frame run do — that is the whole comparison. Folding context in would
    give them different digests and make the harness refuse it."""
    from fovbench.split import Frame, Split
    a = Split(root="/x", frames=[Frame(seq="s", frame_id="7", depth="d",
                                       rgb={"synthetic": "a.jpg"})])
    b = Split(root="/x", frames=[Frame(seq="s", frame_id="7", depth="d",
                                       rgb={"synthetic": "a.jpg"},
                                       context={"synthetic": ["p.jpg", "a.jpg"]},
                                       target_index=1)],
              context_frames=2)
    assert a.digest == b.digest
    assert b.frames[0].stack("synthetic") == ["p.jpg", "a.jpg"]
    assert a.frames[0].stack("synthetic") == ["a.jpg"]
