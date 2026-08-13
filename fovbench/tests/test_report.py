# Copyright (c) 2026.
"""The two headline numbers, and what the report refuses to claim."""
from __future__ import annotations

import csv
import math
import os

import pytest

from fovbench import report as R  # noqa: E402

EDGES = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 55)]


def _bin(lo, hi, absrel, scale_ratio=1.0, n_frames=4, n_px=5000.0, **extra):
    return dict(theta_lo=lo, theta_hi=hi, AbsRel=absrel, delta1=1 - absrel,
                scale_ratio=scale_ratio, raw_scale_ratio=scale_ratio,
                anchored_ratio=scale_ratio,
                n_frames=n_frames, n_px_mean=n_px,
                SqRel=absrel, RMSE=absrel, RMSElog=absrel, log10=absrel,
                delta2=1.0, delta3=1.0, n_valid_total=int(n_px * n_frames),
                **extra)


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


def test_report_text_names_the_split_and_its_headline_column():
    txt = R.render_report(_payload([_run()]))
    assert "abc123def456" in txt
    assert "pen" in txt
    # Cross-model comparability is scoped to the alignment protocol, and the
    # reader is pointed at the column that says which is which — a blanket
    # "not comparable" would also forbid the comparisons that ARE valid.
    assert "comparable only WITHIN an alignment protocol" in txt
    assert "align=" in txt


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



def _win_cell(tilt, absrel, in_cone=1.0, n_frames=4, n_px=5000.0):
    c = _bin(0, 0, absrel, n_frames=n_frames, n_px=n_px)
    c.update(tilt=tilt, in_cone_frac=in_cone, src_px_per_out_px=0.8 + 0.01 * tilt)
    return c


def _win_run(cells, **kw):
    r = _run(**kw)
    r.update(protocol="window", cells=cells)
    r.pop("bins", None)
    return r


def test_window_pen_skips_an_aim_the_lens_clips():
    """The sweep holds the FOV fixed so that only the aim moves. A 40 deg square
    window has a 27.2 deg half-diagonal, so far-off-axis aims lose their corners
    off the lens and differ from the on-axis cell in DEAD AREA as well as aim.
    A ratio across that step is the confound the sweep exists to avoid."""
    cells = [_win_cell(0, 0.05), _win_cell(20, 0.06),
             _win_cell(30, 0.06), _win_cell(40, 0.12, in_cone=0.84)]
    s = R.summarise(_win_run(cells))
    assert s["pen"] == pytest.approx(0.06 / 0.05)     # t0 -> t30, not t0 -> t40
    assert s["clipped"] == 1


def test_a_clipped_aim_is_still_printed_and_flagged():
    cells = [_win_cell(0, 0.05), _win_cell(40, 0.12, in_cone=0.84)]
    txt = R.render_report(_payload([_win_run(cells)]))
    assert "t40!" in txt                # flagged where the number is read
    assert "0.120" in txt               # and not hidden
    assert "WINDOW GEOMETRY" in txt
    assert "in_cone" in txt


def test_radial_bins_are_never_dropped_for_cone_fraction():
    """The gate is a window-sweep concept: a radial bin is a region of ONE frame
    the model saw whole, so there is no per-bin dead area to equalise."""
    r = _run()
    r["in_cone_frac"] = 0.5
    s = R.summarise(r)
    assert s["clipped"] == 0
    assert math.isfinite(s["pen"])


def test_exactly_three_figures_are_written_and_they_are_the_stated_three(tmp_path):
    """The experiment reports AbsRel and delta1 against position in the field,
    and the GT depth they were divided by. Three pictures, no more: a dozen
    files invites quoting one panel as the result."""
    out = str(tmp_path / "figs")
    written = R.write_figures(_payload([_run(view=v, stream=s)
                                        for v in ("fisheye", "rect")
                                        for s in ("synthetic", "real")]), out)
    names = {os.path.basename(p) for p in written}
    assert names == {"AbsRel.png", "delta1.png"}      # no depth in this fixture


def test_no_table_or_legend_mentions_drift_any_more():
    """`drift` was dropped from this experiment. The number survives in the
    JSON only because `datasets_egosynth` cross-checks against it; nothing the
    reader of an ADT report sees may mention it."""
    txt = R.render_report(_payload([_run()]))
    assert "drift" not in txt.lower()


def test_the_depth_figure_is_drawn_from_bins_alone_without_profiles(tmp_path):
    """A run from before the continuous profiles existed still has per-bin
    gt_median, and the depth picture is the one the reader needs to interpret
    the other two."""
    bins = [_bin(lo, hi, 0.05, gt_median=3.0 - 0.2 * i)
            for i, (lo, hi) in enumerate(EDGES)]
    written = R.write_figures(_payload([_run(bins=bins)]), str(tmp_path / "f"))
    assert any(os.path.basename(p) == "gt_depth.png" for p in written)


def test_no_depth_figure_when_the_run_never_measured_depth(tmp_path):
    """The picture exists to show the MEASURED depth. A run that predates
    gt_median has none, and an empty page would read as "measured, and there was
    nothing there"."""
    written = R.write_figures(_payload([_run()]), str(tmp_path / "f"))
    names = {os.path.basename(p) for p in written}
    assert names == {"AbsRel.png", "delta1.png"}


def test_the_two_views_share_one_x_range_per_axis(tmp_path):
    """The rect theta panel used to stop at 52 deg while the fisheye one ran to
    55, and the two were read side by side as if they covered the same field —
    a panel that had run out of camera looked like a curve that had finished.
    One range per axis turns the missing field into blank space."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    rad = [_run(model="vggt_1b", view=v, stream="synthetic")
           for v in ("rect", "fisheye")]
    for r in rad:
        for i, c in enumerate(r["bins"]):
            c["bin_lo"], c["bin_hi"] = float(i * 10), float(i * 10 + 10)
    th = R._shared_xlim(rad, "theta")
    assert th is not None
    # spans what ANY view reaches: the raw fisheye's cone, not the rect's 52
    assert th[1] > 54.8
    # and each view's own limits sit inside it
    for view, expect in (("fisheye", 54.6), ("rect", 42.2)):
        whole, reach = R._spans(rad, view, "theta")
        assert whole == pytest.approx(expect, abs=0.5)
        assert reach <= th[1]
    # The radius panels are drawn on the RAW SENSOR, not in each view's own
    # image plane. Drawn raw they invert — rect runs to sqrt(2) in its corners
    # and the fisheye stops at 1.0, which reads as the fisheye seeing less when
    # it sees more — so on the plotted axis the rect arm must end FIRST, as it
    # does on theta.
    rr = R._shared_xlim(rad, "radius")
    assert rr[1] < 1.05
    assert R._spans(rad, "rect", "radius")[1] < R._spans(rad, "fisheye", "radius")[1]
    # and the panel's x values go through the same conversion
    assert R._plot_x([1.411], "rect", "radius", 518)[0] < 0.95
    assert R._plot_x([0.9], "fisheye", "radius", 518)[0] == pytest.approx(0.9)


def test_the_context_figure_refuses_to_draw_two_different_splits(tmp_path):
    """Every line in that figure is meant to be the same 50 frames with only
    the evidence changed. Drawing two splits together would look exactly like a
    context effect and be a difference in scenes."""
    pytest.importorskip("matplotlib")
    a = _payload([_run(model="vggt_1b", view="fisheye")])
    b = _payload([_run(model="vggt_1b", view="fisheye")])
    b["digest"] = "deadbeefcafe"
    with pytest.raises(SystemExit) as e:
        R.write_context_figure({"N=1": a, "10c": b}, str(tmp_path))
    assert "one split" in str(e.value)


def test_context_line_style_comes_from_the_config_not_the_label():
    """A mislabelled directory must not draw a 10-frame run as the baseline."""
    lying = {"config": {"context_frames": 10, "context_stride": 10}}
    colour, ls, _, legend = R._context_style(lying)
    assert ls == "--" and "stride 10" in legend and colour != "0.15"
    base = R._context_style({"config": {"context_frames": 1, "context_stride": 1}})
    assert base[1] == "-" and "N=1" in base[3]
    # consecutive and strided at the same N differ by dash, not by colour
    c5 = R._context_style({"config": {"context_frames": 5, "context_stride": 1}})
    s5 = R._context_style({"config": {"context_frames": 5, "context_stride": 10}})
    assert c5[0] == s5[0] and c5[1] != s5[1]
