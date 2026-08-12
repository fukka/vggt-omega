# Copyright (c) 2026.
"""The closing invariant: the harness reports what it was given.

Everything else in this suite checks a piece. This drives the whole path — split,
decode, read points, predict, gather, align, score, report — and asserts that a
prediction whose error is known comes back with that error and no other.

The tests that want the real release skip themselves when it is not staged. It is
licensed data and is never committed; `docs/data/ego-synth-5b-sparse-depth.md`
says where it lives.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from slambench import data as D
from slambench import metrics as MT
from slambench import models as M
from slambench import report as R
from slambench import run as RUN
from slambench import split as S

#: Where the 260 MB sample lands if it has been staged. Absent on a fresh
#: checkout, which is the normal case and not a failure.
SAMPLE = os.environ.get(
    "EGOSYNTH_SAMPLE", os.path.expanduser("~/Desktop/ADT/ego-synth-5b-sample"))
needs_sample = pytest.mark.skipif(
    not os.path.isdir(SAMPLE),
    reason="ego-synth sample not staged (licensed; see the data card)")


def _args(tmp_path, **kw):
    a = RUN.build_parser().parse_args([])
    a.egosynth_root = SAMPLE
    a.out = str(tmp_path)
    a.device = "cpu"
    a.takes, a.n_frames, a.log_every = 1, 2, 1000
    a.baselines = "raw"
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@needs_sample
def test_an_exact_prediction_reads_zero_through_the_whole_harness(tmp_path):
    """The oracle is ground truth put through a scale and shift — exactly the
    freedom an up-to-scale model has. Anything the harness adds on top would
    show here: a misread column, the sigma filter dropping the wrong points, the
    gather landing on a neighbour, an alignment fitted over the wrong support."""
    out = RUN.run(_args(tmp_path, models=M.ORACLE, datasets="aea,oxford"))
    assert out["runs"], "no run produced"
    for r in out["runs"]:
        assert r["AbsRel"] < 1e-5, (r["dataset"], r["AbsRel"])
        assert r["delta1"] == pytest.approx(1.0)
        assert r["n_frames"] > 0


@needs_sample
def test_the_four_datasets_are_reported_apart_and_differ_in_scale(tmp_path):
    """Scene scale is the axis these four differ on — indoors at ~1.2 m against
    Oxford outdoors — and every metric here is relative. Pooling them into one
    row would hide exactly that."""
    out = RUN.run(_args(tmp_path, models=M.ORACLE,
                        datasets="aea,nymeria,egoexo4d,oxford"))
    by = {r["dataset"]: r for r in out["runs"]}
    assert set(by) == {"aea", "nymeria", "egoexo4d", "oxford"}
    assert by["oxford"]["gt_median"] > 3 * by["aea"]["gt_median"], {
        k: round(v["gt_median"], 2) for k, v in by.items()}


@needs_sample
def test_the_analytic_stand_in_exercises_the_gather_end_to_end(tmp_path):
    """A dense map from image intensity: the score is meaningless on purpose, and
    what it proves is that decode -> feed -> predict -> sample at the point list
    -> report runs on the real release without a weight in sight."""
    out = RUN.run(_args(tmp_path, models=M.ANALYTIC, datasets="aea"))
    assert out["runs"] and np.isfinite(out["runs"][0]["AbsRel"])
    assert os.path.isfile(os.path.join(str(tmp_path), "report.txt"))
    assert os.path.isfile(os.path.join(str(tmp_path), "results.csv"))


@needs_sample
def test_the_split_digest_is_stable_and_covers_the_caps(tmp_path):
    """Two runs are comparable exactly when their digests match, so the caps that
    subsample the release have to be inside the digest — otherwise a 1-take run
    and a 200-take run compare as equals."""
    a = S.build(SAMPLE, ["aea"], n_frames=3, takes_per_dataset=1, verbose=False)
    b = S.build(SAMPLE, ["aea"], n_frames=3, takes_per_dataset=1, verbose=False)
    c = S.build(SAMPLE, ["aea"], n_frames=5, takes_per_dataset=1, verbose=False)
    assert a.digest == b.digest
    assert a.digest != c.digest
    p = a.save(os.path.join(str(tmp_path), "m.json"))
    assert S.Split.load(p).digest == a.digest


@needs_sample
def test_an_edited_manifest_is_refused(tmp_path):
    """The digest is the comparability token; a hand-edited frame list that kept
    the old digest would silently rescore something else."""
    a = S.build(SAMPLE, ["aea"], n_frames=3, takes_per_dataset=1, verbose=False)
    p = a.save(os.path.join(str(tmp_path), "m.json"))
    doc = json.load(open(p))
    doc["frames"] = doc["frames"][:-1]
    json.dump(doc, open(p, "w"))
    with pytest.raises(ValueError):
        S.Split.load(p)


@needs_sample
def test_the_release_matches_what_the_data_card_claims_about_it():
    """The invariants a run depends on, checked against the real files rather
    than against the document that describes them."""
    takes = D.find_takes(SAMPLE, verbose=False)
    assert {t.dataset for t in takes} == set(D.DATASETS)
    for t in takes:
        pts = D.read_points(t.npz(t.clips[0]), 0)
        assert len(pts) > 0
        assert pts.d.dtype == np.float32 and (pts.d > 0).all()
        # gotcha 2: u reaches 895.5 and rint would put it one past the end
        assert pts.u.max() <= D.RES - 0.5 and pts.v.max() <= D.RES - 0.5
        vi, ui = pts.index
        assert ui.max() <= D.RES - 1 and vi.max() <= D.RES - 1
        frames = D.decode_frames(t.video(t.clips[0]), [0])
        assert frames[0].shape == (D.RES, D.RES, 3)


def test_rect_derect_without_a_calibration_root_refuses_before_touching_data():
    """The failure has to arrive as an instruction, not as a stack trace forty
    minutes into a run."""
    a = RUN.build_parser().parse_args([])
    a.egosynth_root, a.calib_root = "/nonexistent", ""
    a.baselines = "raw,rect_derect"
    a.models = M.ORACLE
    with pytest.raises(SystemExit) as e:
        RUN.run(a)
    assert "calib-root" in str(e.value)


def test_the_report_says_absolute_absrel_is_not_cross_model():
    """DAv2 is scored in disparity space and the depth heads in depth space. A
    reader who compares those two numbers directly has been misled, so the
    report has to say so on its face."""
    payload = {"protocol": S.PROTOCOL, "digest": "d", "n_frames": 1,
               "datasets": ["aea"], "takes": ["aea/t"], "requested_models": [],
               "config": {"baselines": ["raw"], "gt_variant": "fisheye",
                          "sigma_max": 0.01, "sigma_column": "inv_dist_std",
                          "takes_per_dataset": 1, "n_frames_per_take": 1},
               "runs": [{"model": "m", "family": "f", "size": "s",
                         "params_m": 1.0, "align": "scale_shift",
                         "input_size": 518, "dataset": "aea", "baseline": "raw",
                         "AbsRel": 0.1, "delta1": 0.9, "n_frames": 1}]}
    text = R.render(payload)
    assert "NOT COMPARABLE" in text
    assert "rectifies nothing" in text
