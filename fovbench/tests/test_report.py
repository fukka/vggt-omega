# Copyright (c) 2026.
"""The two headline numbers, and what the report refuses to claim."""
from __future__ import annotations

import csv
import math
import os

import pytest

from fovbench import report as R  # noqa: E402

EDGES = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 55)]


def _bin(lo, hi, absrel, scale_ratio=1.0, n_frames=4, n_px=5000.0):
    return dict(theta_lo=lo, theta_hi=hi, AbsRel=absrel, delta1=1 - absrel,
                scale_ratio=scale_ratio, raw_scale_ratio=scale_ratio,
                anchored_ratio=scale_ratio,
                n_frames=n_frames, n_px_mean=n_px,
                SqRel=absrel, RMSE=absrel, RMSElog=absrel, log10=absrel,
                delta2=1.0, delta3=1.0, n_valid_total=int(n_px * n_frames))


def _run(model="vggt_1b", stream="synthetic", view="fisheye", bins=None, **kw):
    r = dict(model=model, family="VGGT", size="1B", params_m=1200.0,
             align="scale_shift", input_size=518, protocol="radial",
             stream=stream, view=view, overall=_bin(0, 55, 0.2),
             in_cone_frac=1.0,
             bins=bins or [_bin(lo, hi, 0.05 + 0.03 * i, 1.0 - 0.05 * i)
                           for i, (lo, hi) in enumerate(EDGES)])
    r.update(kw)
    return r


def _payload(runs):
    return dict(protocol="adt-fov-v1", digest="abc123def456", adt_root="/x",
                n_frames=10, sequences=["seqA"],
                config=dict(streams=["synthetic"], views=["fisheye"],
                            protocols=["radial"], theta_edges=[0, 55],
                            tilts=[0], azimuths=[0], window_fov=40.0,
                            depth_max_m=10.0, metric_max_depth=100.0,
                            min_in_cone_frac=0.5, analytic_bias=0.0),
                runs=runs)


def test_penalty_is_outer_over_inner_absrel():
    s = R.summarise(_run())
    assert s["pen"] == pytest.approx((0.05 + 0.03 * 5) / 0.05)
    assert s["lo"] == 0 and s["hi"] == 50


def test_drift_reads_over_prediction_toward_the_rim_as_greater_than_one():
    """anchored_ratio = median(gt/pred) after the model's affine is fitted on the
    innermost bin. Falling with eccentricity means pred is growing relative to
    GT, i.e. the model pushes the periphery away."""
    s = R.summarise(_run())
    assert s["drift"] > 1.0
    assert s["drift"] == pytest.approx(1.0 / (1.0 - 0.05 * 5))


def test_a_flat_model_scores_penalty_one():
    flat = [_bin(lo, hi, 0.08, 1.0) for lo, hi in EDGES]
    s = R.summarise(_run(bins=flat))
    assert s["pen"] == pytest.approx(1.0)
    assert s["drift"] == pytest.approx(1.0)


def test_empty_outer_bins_are_skipped_not_scored_as_zero():
    """The rectified arm's outer bins are usually empty. They must not enter the
    penalty as a 0.0 AbsRel, which would read as a perfect periphery."""
    bins = [_bin(lo, hi, 0.05 + 0.03 * i, 1.0 - 0.05 * i)
            for i, (lo, hi) in enumerate(EDGES)]
    bins[-1] = _bin(50, 55, float("nan"), float("nan"), n_frames=0, n_px=0.0)
    bins[-2] = _bin(40, 50, 0.1, 0.9, n_frames=3, n_px=12.0)   # corner slivers only
    s = R.summarise(_run(bins=bins))
    assert s["hi"] == 30                       # last bin with real coverage
    assert s["n_cells"] == 4


def test_a_single_populated_bin_refuses_to_report_a_ratio():
    bins = [_bin(lo, hi, float("nan"), float("nan"), n_frames=0, n_px=0.0)
            for lo, hi in EDGES]
    bins[0] = _bin(0, 10, 0.05)
    s = R.summarise(_run(bins=bins))
    assert math.isnan(s["pen"]) and math.isnan(s["drift"])


def test_window_runs_are_summarised_on_tilt():
    cells = [dict(tilt=t, AbsRel=0.05 + 0.01 * i, delta1=0.9,
                  scale_ratio=1.0, raw_scale_ratio=1.0 - 0.02 * i,
                  anchored_ratio=1.0 - 0.02 * i,
                  n_frames=4, n_px_mean=5000.0)
             for i, t in enumerate((0, 10, 20, 30, 40))]
    s = R.summarise(_run(protocol="window", cells=cells, bins=None))
    assert s["lo"] == 0 and s["hi"] == 40
    assert s["pen"] == pytest.approx(0.09 / 0.05)
    # Windows are separate forward passes of up-to-scale models: no drift.
    assert math.isnan(s["drift"])


def test_report_text_names_the_split_and_both_headline_columns():
    txt = R.render_report(_payload([_run()]))
    assert "abc123def456" in txt
    assert "pen" in txt and "drift" in txt
    assert "NOT comparable across models" in txt


def test_report_text_states_the_rectified_coverage_limit():
    txt = R.render_report(_payload([_run(view="rect"), _run(view="fisheye")]))
    assert "42.3 deg" in txt and "COVERAGE" in txt


def test_csv_has_one_row_per_cell_and_carries_the_digest(tmp_path):
    runs = [_run(stream="synthetic"), _run(stream="real")]
    p = str(tmp_path / "r.csv")
    R.write_csv(_payload(runs), p)
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * len(EDGES)
    assert {r["digest"] for r in rows} == {"abc123def456"}
    assert {r["stream"] for r in rows} == {"synthetic", "real"}
    assert rows[0]["cell"] == "0-10"


def test_csv_leaves_unmeasurable_cells_blank_rather_than_zero(tmp_path):
    bins = [_bin(lo, hi, 0.05) for lo, hi in EDGES]
    bins[-1] = _bin(50, 55, float("nan"), float("nan"), n_frames=0, n_px=0.0)
    p = str(tmp_path / "r.csv")
    R.write_csv(_payload([_run(bins=bins)]), p)
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    assert rows[-1]["AbsRel"] == ""
    assert rows[-1]["n_frames"] == "0"


def test_write_all_emits_every_artifact(tmp_path):
    out = R.write_all(_payload([_run(), _run(stream="real")]), str(tmp_path))
    assert os.path.isfile(out["report"]) and os.path.isfile(out["csv"])
    for f in out["figures"]:
        assert os.path.isfile(f)


def test_figures_plot_the_unaligned_column_not_the_aligned_one(tmp_path):
    """Regression: the figure that exists to show the alignment-free read-out
    must not plot ``scale_ratio``, which is measured on the ALIGNED map and
    carries the same bowl as AbsRel. Only the filename made the two look alike."""
    figs = R.write_figures(_payload([_run(), _run(stream="real")]), str(tmp_path))
    if not figs:
        pytest.skip("matplotlib not installed")
    names = {os.path.basename(f) for f in figs}
    assert "radial_raw_scale_ratio.png" in names
    assert not any("radial_scale_ratio.png" == n for n in names)


def test_coverage_table_keeps_a_row_per_render_size():
    """Views go to each model at its own token grid, so per-bin pixel counts are
    NOT shared across models. Collapsing them would print one model's coverage
    and label it as everyone's."""
    a = _run(model="vggt_1b", view="fisheye")
    b = _run(model="vggt_omega", view="fisheye")
    b["input_size"] = 512
    txt = R.render_report(_payload([a, b]))
    cov = txt.split("COVERAGE")[1]
    assert "518" in cov and "512" in cov


def test_report_shouts_when_a_requested_model_did_not_run():
    """A two-model table must not be mistakable for a four-model one."""
    p = _payload([_run()])
    p["requested_models"] = ["vggt_1b", "vggt_omega", "dav2_large", "da3_large"]
    p["skipped_models"] = [
        dict(model="vggt_omega", state="unavailable", detail="checkpoint not found"),
        dict(model="da3_large", state="unavailable", detail="pip install ..."),
    ]
    txt = R.render_report(p)
    head = txt.split("RADIAL")[0]
    assert "NOT RUN: vggt_omega" in head and "NOT RUN: da3_large" in head
    assert "2 of 4" in head


def test_figures_are_emitted_for_both_binning_axes(tmp_path):
    """Distance from the optical centre is the axis that was asked for; the
    incidence angle is the one on which the two views mean the same thing. Both
    are produced, from the same single alignment fit."""
    r = _run()
    r["radius_bins"] = [dict(b, bin_lo=lo, bin_hi=hi) for b, (lo, hi) in
                        zip(r["bins"], [(0, .2), (.2, .4), (.4, .6),
                                        (.6, .8), (.8, 1.), (1., 1.45)])]
    figs = R.write_figures(_payload([r]), str(tmp_path))
    if not figs:
        pytest.skip("matplotlib not installed")
    names = {os.path.basename(f) for f in figs}
    assert "radial_AbsRel.png" in names
    assert "radial_AbsRel_radius.png" in names
    assert "radial_delta1_radius.png" in names


def test_report_marks_drift_as_outside_the_protocol():
    """The protocol is one whole-frame fit per frame, with binning applied
    afterwards by masking. `drift` is the single column that departs from it —
    it anchors on the innermost bin — and the report has to say so where the
    number is read, not in a docstring."""
    txt = R.render_report(_payload([_run()]))
    assert "drift*" in txt
    assert "OUTSIDE THE PROTOCOL" in txt
    assert "fitted ONCE per frame" in txt
