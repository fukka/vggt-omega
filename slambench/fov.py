# Copyright (c) 2026.
"""Where in the field of view does depth degrade — asked of real SLAM points.

The FOV experiment in `fovbench/` asks this of a synthetic twin, where the
ground truth is a dense depth map and every pixel of the frame has one. Here the
ground truth is a few thousand triangulated points per frame, and that changes
what the question costs to ask honestly. This module is the difference.

The protocol, and the one rule it inherits unchanged
----------------------------------------------------
**One affine per frame, fitted over every one of that frame's valid points, then
frozen. Binning is a mask applied afterwards.** An up-to-scale model whose depth
bends with eccentricity has, by definition, no single scale that fits every
radius; giving each bin its own fit hands it one, and reports a flat healthy
curve for exactly the failure being looked for. That is not a stylistic choice —
it is measured, on both sides of the repository, and
:func:`~slambench.tests.test_fov.test_per_bin_alignment_would_erase_the_effect`
is the local copy of the measurement.

What sparse ground truth changes
--------------------------------
Three things, and the first is the one that can invent a result.

**1. Depth and eccentricity move together, hard.** On a dense synthetic frame the
two are only loosely related. On real egocentric footage they are not: the
wearer's hands, torso and the surface they are working over sit in the
periphery, and the room they are looking at sits in the middle. Measured on the
release, over 15 frames of one AEA take:

    theta        0-10   10-20   20-30   30-40   40-50   50-55   55-60
    median GT     2.57    2.09    1.10    1.07    0.93    0.80    0.71  m

That is a 3.6x gradient across the field. Every metric here is relative, and a
relative error grows with depth, so a bare "AbsRel against theta" curve on this
data is **partly a depth curve** and would report a gradient for a model with no
field dependence at all. So the primitive this module computes is not a curve, it
is a **two-way table** — eccentricity against distance — and the curve is derived
from it by holding distance fixed. See :func:`standardise`.

**2. A bin is a different number of points in every frame.** The central bins
carry 5 % of a frame and the 40-50 bin carries 24 %, and those shares move with
where the wearer is looking. Averaging per-frame bin means would weight a frame
that put nine points in a bin the same as one that put nine hundred there, so
cells accumulate **sums** and are divided once at the end. This is deliberately a
different estimator from ``metrics.aggregate``, which averages over frames
because it is answering a per-frame question.

**3. Rectifying truncates the field at 55 deg.** The 110 deg pinhole
``rect_derect`` uses reaches 54.9 deg on axis, and ``run``'s support
intersection scores every arm on the points *all* arms could answer for. Past
55 deg the surviving points are the ones nearer the frame diagonal, which is an
azimuthally biased subset — so a run that includes the rectified arm has nothing
honest to say out there, and :data:`DEFAULT_THETA_EDGES` stops before it.

Two protocols, as in the original
---------------------------------
``radial``  the whole frame goes to the model once, and the per-point error is
            binned by the incidence angle of the point's ray: *where in this
            image*.

``window``  a fixed 40 deg pinhole is re-aimed across the lens and the model
            sees only that: *where the camera was pointed*.

They can disagree, and the disagreement is the finding. A model that is merely
bad at image corners shows a radial gradient and a flat window curve; one whose
error is driven by the ray geometry shows both.

The window is aimed, never widened. Varying width and aim together is what made
an earlier sweep in this repository unreadable — a wide window overhangs the
imaged cone, so the fraction of dead pixels moved with the variable under test
and the result was partly a measurement of black. Here the width is fixed at
:data:`DEFAULT_WINDOW_FOV`, the dead fraction is measured per window, and a
window past :data:`MIN_IN_CONE_FRAC` is not scored at all.

A tilted pinhole is not co-axial, and that is worth 1.31x
---------------------------------------------------------
``baselines.RectDerectBaseline`` applies **no** depth conversion, and is right
not to: its pinhole shares the fisheye's axis, so planar z about one is planar z
about the other. A *tilted* window does not share it. The model predicts planar z
about the **window's** axis and the ground truth is planar z about the
**fisheye's**, and the two differ by

    d_fisheye = d_window * (r . z_fisheye) / (r . axis_window)

per point — :meth:`Window.z_to_camera`. At a 40 deg tilt that reaches 1.31 at the
window centre and varies across the window, so it is **radial and not absorbable
by the per-frame affine**: forgetting it would not scale the window arm, it would
bend it, in the same shape as the effect being measured. This is the failure
CONTEXT.md records for the fisheye port, arriving by a different door.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench import data as D          # noqa: E402
from slambench.baselines import Pinhole, _bilinear  # noqa: E402
from slambench.camera import Fisheye624  # noqa: E402

from finetune.eval.metrics import align_depth  # noqa: E402

#: Incidence-angle bin edges, degrees. Stops at 55 because that is where the
#: rectified arm stops being able to answer — see the module docstring. A
#: ``raw``-only run may pass ``--theta-edges 0,10,20,30,40,50,55,60,70``; the
#: 896 frame reaches 56.1 deg at its edge midpoint and 70.3 deg at its corner,
#: so beyond 56 a bin is corners only and says so through its own point count.
DEFAULT_THETA_EDGES: Tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0)

#: Distance strata, as a count of equal-population quantiles of the split's own
#: ground truth. Quantiles rather than fixed metres for two reasons: the four
#: datasets differ by an order of magnitude in scene scale, and equal population
#: makes the standardisation weights uniform, so :func:`standardise` is a plain
#: mean over strata rather than a weighting nobody can check by eye.
DEFAULT_DEPTH_STRATA: int = 5

#: A cell thinner than this is reported as empty rather than as a number. At
#: 5 strata x 6 bins a frame's ~2 000 points spread to ~65 a cell, so this is
#: roughly one frame's worth and the pooled cells are far above it.
MIN_CELL_POINTS: int = 256

#: Field of view of the re-aimed window, degrees. **Fixed**, and the sweep moves
#: only the aim — see the module docstring.
DEFAULT_WINDOW_FOV: float = 40.0

#: Where the window is aimed: tilt off the optical axis, and around it.
DEFAULT_TILTS: Tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0)
DEFAULT_AZIMUTHS: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

#: A window with less than this share of its pixels backed by real ones is
#: mostly black, and scoring it measures the vignette rather than the model.
MIN_IN_CONE_FRAC: float = 0.5

#: What a cell carries. Sums, so that frames pool by addition; the report
#: divides once. ``sum_log_ratio`` is the accumulable form of "median gt/pred on
#: the *unaligned* prediction" — the one statistic with no alignment in the way,
#: and so the only one that stays monotone under a radial bias. Its pooled value
#: is a geometric mean, which is the right average for a ratio and is a sum in
#: log space; a median is not accumulable and would force the run to hold every
#: point of every frame in memory.
SUM_KEYS: Tuple[str, ...] = ("n", "sum_absrel", "sum_sqrel", "sum_delta1",
                             "sum_gt", "sum_gt2", "sum_log_ratio")


# --------------------------------------------------------------------------- #
# The eccentricity axis
# --------------------------------------------------------------------------- #

def theta_of(cam: Fisheye624, pts: "D.FramePoints") -> np.ndarray:
    """Incidence angle in degrees of the ray each ground-truth point sits on.

    This is the axis the whole module bins on, and it is the one quantity here
    that needs the camera model. The ``raw`` baseline does not need a camera to
    *predict*; it needs one to be **asked this question**, which is why a FOV run
    requires ``--calib-root`` on both arms where ``run`` requires it on one.
    """
    return cam.theta_of(pts.u.astype(np.float64), pts.v.astype(np.float64))


def depth_edges_from(depths: Sequence[np.ndarray], strata: int) -> List[float]:
    """Equal-population distance strata from the split's own ground truth.

    Model-independent by construction — it reads GT and nothing else — so
    computing it in a pre-pass leaks nothing into the scores it will later
    stratify. Returned open at both ends so that no point falls outside.
    """
    g = np.concatenate([np.asarray(d, np.float64).ravel() for d in depths])
    g = g[np.isfinite(g) & (g > 0)]
    if g.size < strata * MIN_CELL_POINTS:
        raise SystemExit(
            f"[slambench] {g.size} ground-truth points is too few to cut into "
            f"{strata} distance strata of at least {MIN_CELL_POINTS}. Widen the "
            f"split (--takes / --n-frames) or lower --depth-strata.")
    qs = np.percentile(g, np.linspace(0, 100, strata + 1))
    qs[0], qs[-1] = 0.0, float("inf")
    return [float(q) for q in qs]


def _bin_index(x: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Half-open ``[lo, hi)`` bin index, or -1 for anything outside.

    NaN is passed in deliberately for the points a caller has already ruled
    out, and lands in -1 with the rest — so the comparison warnings it raises
    are noise rather than news, and are silenced here rather than left to fill
    a three-hour run's log with a line per frame.
    """
    with np.errstate(invalid="ignore"):
        idx = np.digitize(x, np.asarray(edges[1:-1], float), right=False)
        outside = ~np.isfinite(x) | (x < edges[0]) | (x >= edges[-1])
    return np.where(outside, -1, idx).astype(np.int64)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def frame_cells(pred: np.ndarray, gt: np.ndarray, theta: np.ndarray,
                align_mode: str, pos_edges: Sequence[float],
                depth_edges: Sequence[float],
                min_depth: float = 0.01, max_depth: float = 120.0,
                min_points: int = D.MIN_FRAME_POINTS
                ) -> Optional[Dict[Tuple[int, int], Dict[str, float]]]:
    """One frame's contribution to the table, under ONE frozen affine.

    Returns ``{(pos_bin, depth_stratum): {sum key: value}}``, or ``None`` when
    the frame carries too few valid points to determine the two-parameter fit —
    a frame that was not measured, which is not the same as a frame that scored
    badly and must not be pooled as though it were.

    The fit is over **all** the frame's valid points and is frozen before any
    binning happens. See the module docstring for why that is the load-bearing
    part of the protocol.
    """
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(pred) & np.isfinite(gt) & (pred > 0)
              & (gt > min_depth) & (gt < max_depth) & np.isfinite(theta))
    if int(ok.sum()) < min_points:
        return None

    aligned = np.asarray(align_depth(pred, gt, ok, mode=align_mode), np.float64)

    ti = _bin_index(np.where(ok, theta, np.nan), pos_edges)
    si = _bin_index(np.where(ok, gt, np.nan), depth_edges)
    with np.errstate(invalid="ignore"):
        live = ok & (ti >= 0) & (si >= 0) & np.isfinite(aligned) & (aligned > 0)
    if not live.any():
        return None

    a, g = aligned[live], gt[live]
    err = a - g
    absrel = np.abs(err) / g
    sqrel = err * err / g
    delta1 = (np.maximum(a / g, g / a) < 1.25).astype(np.float64)
    # The unaligned ratio, in log space so that frames pool by addition.
    log_ratio = np.log(g / pred[live])

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for t, s in zip(*np.unique(np.stack([ti[live], si[live]]), axis=1)):
        m = (ti[live] == t) & (si[live] == s)
        out[(int(t), int(s))] = {
            "n": float(m.sum()),
            "sum_absrel": float(absrel[m].sum()),
            "sum_sqrel": float(sqrel[m].sum()),
            "sum_delta1": float(delta1[m].sum()),
            "sum_gt": float(g[m].sum()),
            "sum_gt2": float((g[m] ** 2).sum()),
            "sum_log_ratio": float(log_ratio[m].sum()),
        }
    return out


class Table:
    """The pooled two-way table: **field position x distance**, as sums.

    Field position is the incidence angle of the point's ray under the
    ``radial`` protocol and the window's tilt under ``window`` — one accumulator
    and one standardisation for both, because the confound they have to control
    for is the same confound and a second implementation of the control is a
    second place for it to be got wrong. ``pos_name`` is what the report calls
    the axis; nothing here branches on it.

    Frames are added, never averaged in — the module docstring says why. The
    reductions below are the only place a division happens.
    """

    def __init__(self, pos_edges: Sequence[float],
                 depth_edges: Sequence[float], pos_name: str = "theta"):
        self.pos_name = pos_name
        self.pos_edges = [float(e) for e in pos_edges]
        self.depth_edges = [float(e) for e in depth_edges]
        self.n_pos = len(self.pos_edges) - 1
        self.n_depth = len(self.depth_edges) - 1
        self.cells: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.n_frames = 0

    def add(self, cells: Optional[Dict[Tuple[int, int], Dict[str, float]]]) -> None:
        if not cells:
            return
        self.n_frames += 1
        for key, sums in cells.items():
            tgt = self.cells.setdefault(key, {k: 0.0 for k in SUM_KEYS})
            for k in SUM_KEYS:
                tgt[k] += sums[k]

    # -- reductions --------------------------------------------------------- #
    def cell(self, t: int, s: int) -> Dict[str, float]:
        """One cell's means, or NaNs when it is too thin to report."""
        c = self.cells.get((t, s))
        nan = float("nan")
        if c is None or c["n"] < MIN_CELL_POINTS:
            return {"n": 0.0 if c is None else c["n"], "AbsRel": nan,
                    "SqRel": nan, "delta1": nan, "gt_mean": nan, "gt_std": nan,
                    "scale_ratio": nan}
        n = c["n"]
        var = max(c["sum_gt2"] / n - (c["sum_gt"] / n) ** 2, 0.0)
        return {"n": n, "AbsRel": c["sum_absrel"] / n,
                "SqRel": c["sum_sqrel"] / n, "delta1": c["sum_delta1"] / n,
                "gt_mean": c["sum_gt"] / n, "gt_std": math.sqrt(var),
                "scale_ratio": math.exp(c["sum_log_ratio"] / n)}

    def pos_row(self, t: int) -> Dict[str, float]:
        """One field-position bin, pooled over every distance — the **confounded** row.

        Reported because it is what the naive question asks and because the gap
        between it and :func:`standardise` is itself a result, not because it is
        the answer.
        """
        acc = {k: 0.0 for k in SUM_KEYS}
        for s in range(self.n_depth):
            c = self.cells.get((t, s))
            if c:
                for k in SUM_KEYS:
                    acc[k] += c[k]
        nan = float("nan")
        if acc["n"] < MIN_CELL_POINTS:
            return {"n": acc["n"], "AbsRel": nan, "SqRel": nan, "delta1": nan,
                    "gt_mean": nan, "gt_std": nan, "scale_ratio": nan}
        n = acc["n"]
        var = max(acc["sum_gt2"] / n - (acc["sum_gt"] / n) ** 2, 0.0)
        return {"n": n, "AbsRel": acc["sum_absrel"] / n,
                "SqRel": acc["sum_sqrel"] / n, "delta1": acc["sum_delta1"] / n,
                "gt_mean": acc["sum_gt"] / n, "gt_std": math.sqrt(var),
                "scale_ratio": math.exp(acc["sum_log_ratio"] / n)}

    def common_strata(self) -> List[int]:
        """Distance strata populated in **every** theta bin that has any points.

        Standardising each bin over whatever strata it happens to have would
        put the confound straight back: a bin scored on its near strata and
        another on its far ones are not being compared at fixed distance. So the
        strata are the intersection, the same set for every bin, and the ones
        dropped are reported rather than absorbed.
        """
        live_p = [t for t in range(self.n_pos)
                  if self.pos_row(t)["n"] >= MIN_CELL_POINTS]
        if not live_p:
            return []
        return [s for s in range(self.n_depth)
                if all(self.cells.get((t, s), {}).get("n", 0.0) >= MIN_CELL_POINTS
                       for t in live_p)]

    def standardised(self, key: str = "AbsRel") -> Tuple[List[float], List[int]]:
        """The depth-controlled curve: every bin at a common distance mix.

        Direct standardisation. The strata are equal-population quantiles of the
        pooled ground truth, so the weights are uniform and this is a plain mean
        over the common strata — every field-position bin scored as if it had
        been looking at the same spread of distances as every other one.

        Returns ``(curve, strata_used)``. A bin whose common strata are not all
        populated comes back NaN rather than being quietly renormalised onto a
        different distance mix from its neighbours.
        """
        strata = self.common_strata()
        curve = []
        for t in range(self.n_pos):
            vals = [self.cell(t, s)[key] for s in strata]
            curve.append(float(np.mean(vals))
                         if strata and all(np.isfinite(v) for v in vals)
                         else float("nan"))
        return curve, strata

    def to_json(self) -> dict:
        return {
            "pos_name": self.pos_name,
            "pos_edges": self.pos_edges,
            "depth_edges": self.depth_edges,
            "n_frames": self.n_frames,
            "cells": [dict(pos_bin=t, depth_stratum=s,
                           pos_lo=self.pos_edges[t],
                           pos_hi=self.pos_edges[t + 1],
                           depth_lo=self.depth_edges[s],
                           depth_hi=self.depth_edges[s + 1], **self.cell(t, s))
                      for t in range(self.n_pos) for s in range(self.n_depth)],
            "pos_rows": [dict(pos_bin=t, pos_lo=self.pos_edges[t],
                                pos_hi=self.pos_edges[t + 1],
                                **self.pos_row(t))
                           for t in range(self.n_pos)],
            "standardised_AbsRel": self.standardised("AbsRel")[0],
            "standardised_delta1": self.standardised("delta1")[0],
            "standardised_strata": self.common_strata(),
        }


def controlled(pos_rows: Sequence[dict], cells: Sequence[dict],
               key: str = "AbsRel") -> dict:
    """:meth:`Table.standardised` over an already-serialised table, plus its cost.

    The report and any later analysis read ``results.json``, not the accumulator,
    and a second implementation of the standardisation is a second place for it
    to drift. This is that one place.

    ``share`` is the part that keeps the control honest. Holding distance fixed
    means scoring every bin on the strata they *all* reach, and on this data the
    centre and the rim barely overlap in distance — the middle of the field is
    metres away and the rim is centimetres. So a standardised number can rest on
    a small and unrepresentative slice of a bin, and ``share[i]`` says what
    fraction of bin ``i``'s points it actually describes. A curve whose share is
    0.2 is a real comparison of a fifth of the data, not a comparison of the
    data; there is no way to have both, and the choice belongs to the reader.
    """
    by_t: Dict[int, Dict[int, dict]] = {}
    for c in cells:
        by_t.setdefault(int(c["pos_bin"]), {})[int(c["depth_stratum"])] = c
    live_p = [int(r["pos_bin"]) for r in pos_rows
              if r.get("n", 0.0) >= MIN_CELL_POINTS]
    strata = sorted({s for t in by_t for s in by_t[t]})
    common = [s for s in strata
              if all(by_t.get(t, {}).get(s, {}).get("n", 0.0) >= MIN_CELL_POINTS
                     for t in live_p)] if live_p else []
    curve, share = [], []
    for r in pos_rows:
        t = int(r["pos_bin"])
        vals = [by_t.get(t, {}).get(s, {}).get(key, float("nan")) for s in common]
        curve.append(float(np.mean(vals))
                     if common and all(np.isfinite(v) for v in vals)
                     else float("nan"))
        kept = sum(by_t.get(t, {}).get(s, {}).get("n", 0.0) for s in common)
        total = float(r.get("n", 0.0))
        share.append(kept / total if total else float("nan"))
    return {"curve": curve, "share": share, "strata": common}


def standardise(pos_rows: Sequence[dict], cells: Sequence[dict],
                key: str = "AbsRel") -> List[float]:
    """The depth-controlled curve alone. See :func:`controlled`."""
    return controlled(pos_rows, cells, key)["curve"]


# --------------------------------------------------------------------------- #
# The window protocol
# --------------------------------------------------------------------------- #

def _aim(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """Rotation taking window-frame directions to camera-frame directions.

    Tilt about the camera's x axis, then roll the tilt around the optical axis
    by the azimuth. Composed in that order so that ``tilt`` always means the
    same angle off axis whatever the azimuth is — which is what makes the four
    azimuths a control on each other rather than four different sweeps.
    """
    t, a = math.radians(tilt_deg), math.radians(azimuth_deg)
    ct, st = math.cos(t), math.sin(t)
    ca, sa = math.cos(a), math.sin(a)
    r_tilt = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    r_az = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return r_az @ r_tilt


@dataclass
class Window:
    """A pinhole of fixed width, aimed off the fisheye's optical axis.

    ``tilt`` 0 makes this the co-axial pinhole ``rect_derect`` already uses, and
    every method below reduces to that case exactly — which is what lets the
    window sweep's first point be checked against the other arm rather than
    taken on trust.
    """

    size: int
    fov_deg: float = DEFAULT_WINDOW_FOV
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0

    def __post_init__(self):
        self.pin = Pinhole(int(self.size), float(self.fov_deg))
        self.R = _aim(self.tilt_deg, self.azimuth_deg)

    @property
    def axis(self) -> np.ndarray:
        """The window's optical axis, in camera coordinates."""
        return self.R[:, 2]

    def rays(self) -> np.ndarray:
        """Unit camera-frame ray per window pixel, ``(size, size, 3)``."""
        d = self.pin.rays().reshape(-1, 3) @ self.R.T
        return d.reshape(self.size, self.size, 3)

    def project(self, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Camera-frame unit rays ``(N, 3)`` -> window pixels.

        Rays behind the window's own image plane come back NaN, which is not the
        same test as being behind the camera: at a 40 deg tilt a ray can be well
        inside the fisheye's cone and still have nothing to do with this window.
        """
        return self.pin.project(np.asarray(dirs, float) @ self.R)

    def z_to_camera(self, dirs: np.ndarray) -> np.ndarray:
        """Per-ray factor taking planar z about the WINDOW axis to planar z about
        the CAMERA axis — the conversion a co-axial pinhole does not need.

        ``d_camera = d_window * (r . z_camera) / (r . axis_window)``, derived in
        the module docstring. It is 1 everywhere when ``tilt`` is 0, and at a
        40 deg tilt it reaches 1.31 and varies across the window, so it bends the
        prediction rather than scaling it and the per-frame affine cannot stand
        in for it.
        """
        d = np.asarray(dirs, float)
        along = d @ self.axis
        with np.errstate(divide="ignore", invalid="ignore"):
            out = d[:, 2] / along
        return np.where(np.isfinite(out) & (along > 1e-6), out, np.nan)


class WindowView:
    """One window, on one take's camera: the render, and the points inside it.

    The map depends only on the camera and the aim, so it is built once per
    (take, window) and reused for every frame — it is the expensive part.
    """

    def __init__(self, cam: Fisheye624, win: Window):
        self.cam = cam
        self.win = win
        d = win.rays().reshape(-1, 3)
        u, v = cam.project(d)
        ok = (np.isfinite(u) & np.isfinite(v)
              & (u >= 0) & (u <= cam.width - 1)
              & (v >= 0) & (v <= cam.height - 1))
        n = win.size
        self.mapx = np.where(ok, u, -1).reshape(n, n).astype(np.float32)
        self.mapy = np.where(ok, v, -1).reshape(n, n).astype(np.float32)
        self.in_cone = ok.reshape(n, n)

    @property
    def in_cone_frac(self) -> float:
        """Share of the window's pixels backed by a real fisheye pixel.

        Reported with every window and enforced by :data:`MIN_IN_CONE_FRAC`. A
        window that overhangs the lens is mostly black, and a model scored on it
        is being scored on its response to padding — the confound that made an
        earlier sweep in this repository unreadable, and the reason the width is
        held fixed here while only the aim moves.
        """
        return float(self.in_cone.mean())

    def render(self, frame: np.ndarray) -> np.ndarray:
        """The fisheye frame as this window sees it. Zero outside the cone.

        Zero-filled rather than border-replicated for the reason
        ``baselines.rectify`` gives: a replicated pixel is a fabricated
        observation and the model would be scored on its response to one.
        """
        import cv2
        return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def sample(self, pred_win: np.ndarray, pts: "D.FramePoints"
               ) -> np.ndarray:
        """Window depth, read at the fisheye points and converted to camera z.

        Three things have to be true for a point to get a value, and each is a
        different way of not being in this window:

        * its ray must project inside the window's grid;
        * the 2x2 bilinear stencil must be **entirely** backed by real pixels —
          otherwise the value returned is part real depth and part the model's
          answer to black padding, and nothing downstream could tell. The
          co-axial arm never needs this check (a 110 deg pinhole is fully backed
          by this lens); a tilted window is exactly where it starts to matter;
        * the conversion to camera-frame z must be finite.
        """
        n = self.win.size
        if pred_win.shape != (n, n):
            raise SystemExit(
                f"[slambench] the window is {n}x{n} but the model answered with "
                f"a {pred_win.shape[0]}x{pred_win.shape[1]} map; every point "
                f"would be read at the wrong pixel")
        d = self.cam.unproject(pts.u.astype(np.float64),
                               pts.v.astype(np.float64))
        u, v = self.win.project(d)
        with np.errstate(invalid="ignore"):
            inside = (np.isfinite(u) & np.isfinite(v)
                      & (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1))
        out = np.full(u.shape, np.nan, np.float32)
        if not inside.any():
            return out
        x0 = np.floor(u[inside]).astype(np.int32)
        y0 = np.floor(v[inside]).astype(np.int32)
        x1 = np.minimum(x0 + 1, n - 1)
        y1 = np.minimum(y0 + 1, n - 1)
        whole = (self.in_cone[y0, x0] & self.in_cone[y0, x1]
                 & self.in_cone[y1, x0] & self.in_cone[y1, x1])
        idx = np.where(inside)[0][whole]
        if not idx.size:
            return out
        val = _bilinear(pred_win, u[idx], v[idx])
        out[idx] = (val * self.win.z_to_camera(d[idx])).astype(np.float32)
        return out
