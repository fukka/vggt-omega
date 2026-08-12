# Copyright (c) 2026.
"""ego-synth 5B sparse SLAM depth, as a fovbench ground truth.

Semi-dense MPS SLAM points projected into every frame of four egocentric Aria
datasets (``aea``, ``nymeria``, ``egoexo4d``, ``oxford``). Format, provenance
and the ten gotchas are in `docs/data/ego-synth-5b-sparse-depth.md`; this module
is that document made executable, and every guard below cites the gotcha it
implements.

Why this GT is worth a second dataset path at all: it carries **both** the raw
fisheye and a 110 deg rectified pinhole *of the same frame*. On ADT the raw and
rectified arms are two constructions of one image, made here by re-rendering;
here they are two streams the producer wrote side by side, so ``rect`` vs
``fisheye`` is like-for-like by construction rather than by our own warp.

Three things make it a different scoring problem from ADT
---------------------------------------------------------

**1. The GT is a point list, not a map — so gather, never scatter.**
``u``/``v`` are float16 and quantise to half a pixel above 512, so scattering
the points into an 896 array collides them: 5 292 points land on 4 150 distinct
pixels on the frame measured, and the last write wins (gotcha 2). The metrics
are per-point anyway, so gathering the *prediction* at the point list is both
lossless and less code. Nothing here ever builds a dense GT.

The scoring code this feeds is unchanged: :func:`fovbench.geometry.bin_by` and
everything under it work through boolean masks and elementwise ops, so they take
a flat ``(N,)`` point list exactly as they take an ``(H, W)`` map. The frozen-fit
rule in `fovbench/README.md` therefore holds verbatim — only the *support*
changes from a dense mask to a point list.
``tests/test_egosynth.py::test_the_point_path_is_the_same_estimator_as_the_dense_one``
pins that equivalence rather than assuming it.

**2. Theta exists on ``rect`` and not on ``fisheye``.**
``meta.rectification`` gives a complete pinhole (110 deg, focal 313.693 px,
render 896, principal point at the render centre), so an incidence angle is
computable on the rectified stream. **No fisheye camera model ships with this
data** — computing theta there needs the Aria calibration out of the source VRS.
Radius binning works on both; theta binning works only on ``rect``, and asking
for it on ``fisheye`` raises :class:`ThetaUnavailable` rather than returning a
plausible wrong number. `fovbench/README.md` already says radius and theta are
not interchangeable; this is that distinction with teeth.

**3. A single frame cannot be binned.**
A frame carries 1 300-5 300 points, unevenly spread over the field: on the
``nymeria`` frame checked, every point sits between theta 30.2 and 63.6 deg —
*nothing* within 30 deg of the axis (gotcha 10). So binning per frame and
averaging bins across frames, which is what the ADT path does, would average
over cells that are empty at random. Here the **fit stays per frame** (each
forward pass of an up-to-scale model has its own arbitrary scale, so it must)
and the *aligned* points are pooled across frames before a single binning pass.
:class:`PointPool` is that, and it is the only structural difference from the
ADT driver.

A note on what the report's columns mean here
---------------------------------------------
* The report's ``stream`` column carries the **dataset**, not a sensor variant.
  That is deliberate: scene scale is the axis along which these four differ
  (median depth ~1.2 m indoors to ~5.3 m at Oxford, p99 23 m), every metric in
  this benchmark is relative and grows with depth, and pooling four scene scales
  into one bin is precisely the confound ``gt_median`` exists to expose.
* ``raw_scale_ratio`` is **NaN** for every egosynth run. It is ``median(gt/pred)``
  on the *unaligned* prediction, and each frame's unaligned prediction is in its
  own arbitrary scale — pooling them across frames would report the spread of
  those constants. ``drift*`` survives pooling because it is measured after a
  per-frame anchored fit; see :func:`PointPool.profile`.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fovbench import _REPO  # noqa: F401  (import registers sys.path)

from fovbench import geometry as G  # noqa: E402

from finetune.eval.metrics import align_depth, depth_metrics  # noqa: E402

#: Everything in this release was resized to 896x896 for Wan-5B video training
#: (gotcha 1). The source Aria frames are 1408x1408; any calibration brought in
#: from elsewhere must be scaled by 896/1408 = 0.636364 first.
RES: int = 896

#: The four datasets that have sparse depth. ``data_hot3d_5b`` exists but its
#: sparse depth is 0 bytes, and ``data_combined_5b`` only symlinks these four.
DATASETS: Tuple[str, ...] = ("aea", "nymeria", "egoexo4d", "oxford")

#: fovbench's view names -> this dataset's stream directories. The benchmark has
#: said ``rect``/``fisheye`` since the ADT path; the producer wrote
#: ``rectified``/``fisheye``. One mapping, in one place.
VIEW_TO_VARIANT: Dict[str, str] = {"rect": "rectified", "fisheye": "fisheye"}

#: Incidence-angle bin edges for the 110 deg rectified pinhole. It reaches 55 deg
#: at the middle of an edge and 63.65 deg into a corner, so the ADT edges (which
#: stop at 55, the Aria cone) would drop every corner point. The last bin is
#: corner-only by construction and the figures ring it as thin.
THETA_EDGES: Tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 63.7)

#: Default cut on ``inv_dist_std`` — MPS inverse-distance 1-sigma, in 1/metres,
#: which is scale-invariant triangulation quality. The GT ships **unfiltered on
#: purpose** so the consumer chooses; this is the choice, and
#: :func:`fovbench.run.run` writes it into ``results.json`` so a number can never
#: be read without it. 0.01 keeps ~the well-triangulated points and is what the
#: data card's own worked example uses.
DEFAULT_SIGMA_MAX: float = 0.01

#: A frame with fewer than this many surviving points is not scored. The affine
#: has two parameters and the fit is over the whole frame, so this is a floor on
#: the fit, not on any bin.
MIN_FRAME_POINTS: int = 256

#: Anchor-band floor, in points. The ADT path uses 64 *pixels* out of ~250 000,
#: i.e. 0.03 % of a frame; a frame here carries 1 300-5 300 points, so the same
#: literal 64 is 1-5 % of one. That is the right order for "this band can carry
#: a two-parameter fit" and is kept identical so the two paths' drift columns
#: are not silently fitted under different conditions.
MIN_ANCHOR_POINTS: int = 64

#: A band must be anchorable in at least this share of the pooled frames before
#: it can be the anchor. See :func:`PointPool.profile` — pooling frames that
#: anchored on *different* bands is what this prevents.
MIN_ANCHOR_FRAME_FRAC: float = 0.5


class ThetaUnavailable(RuntimeError):
    """Raised when theta is asked for on the raw fisheye stream.

    There is no fisheye camera model anywhere in this data — not in
    ``meta.json``, not in the npz. The incidence angle of a raw-fisheye point
    cannot be computed from these files at all; it needs the Aria calibration
    from the source VRS. Radius is available on both streams and is what the
    fisheye arm is binned by.
    """


# --------------------------------------------------------------------------- #
# Take discovery and metadata
# --------------------------------------------------------------------------- #

@dataclass
class Rectification:
    """The pinhole the ``rectified`` stream was rendered through.

    Read from ``meta.rectification``, never assumed. The principal point is not
    stored there, but it is determined: ``focal_px`` is exactly
    ``(render_size / 2) / tan(fov / 2)`` (448 / tan 55 deg = 313.693), which
    places the centre on the render's midline at ``render_size / 2``. Note that
    this is the *producer's* convention and differs by half a pixel from
    fovbench's own ``W / 2 - 0.5``; the GT's ``u``/``v`` were produced under the
    former, so the former is what labels them.
    """

    fov_deg: float
    focal_px: float
    render_size: int

    @property
    def cx(self) -> float:
        return self.render_size / 2.0

    @property
    def cy(self) -> float:
        return self.render_size / 2.0

    @property
    def theta_max_deg(self) -> float:
        """Incidence angle at a corner — the outermost direction imaged."""
        r = math.hypot(self.render_size / 2.0, self.render_size / 2.0)
        return math.degrees(math.atan(r / self.focal_px))

    @classmethod
    def from_meta(cls, meta: dict) -> "Rectification":
        r = meta["rectification"]
        return cls(fov_deg=float(r["fov_deg"]), focal_px=float(r["focal_px"]),
                   render_size=int(r["render_size"]))


@dataclass
class Take:
    """One recording's worth of clips, and the geometry they share."""

    dataset: str
    name: str
    path: str
    rect: Rectification
    clips: List[str] = field(default_factory=list)   # clip ids, as strings
    #: clip id -> frame count, read from ``meta.clips[].num_frames`` rather than
    #: assumed. Every clip in the release measured 121, but a clip that is short
    #: must shorten its own frame list, not be discovered as a decode failure.
    clip_frames: Dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.name}"

    def frames_in(self, clip: str) -> int:
        return int(self.clip_frames.get(clip, 0))

    def npz(self, clip: str) -> str:
        return os.path.join(self.path, "sparse_depth", f"{clip}.npz")

    def video(self, clip: str, view: str) -> str:
        return os.path.join(self.path, VIEW_TO_VARIANT[view], f"{clip}.mp4")

    def valid_mask_path(self) -> str:
        return os.path.join(self.path, "rectified_valid_mask.png")


def find_takes(root: str, datasets: Sequence[str] = DATASETS,
               limit: int = 0, verbose: bool = True) -> List[Take]:
    """Takes under ``root`` that carry sparse depth, in a stable order.

    Filtered on the presence of ``sparse_depth/*.npz``, never on the directory
    listing: ``egoexo4d`` has 1 090 populated take dirs out of 2 380 (gotcha 9),
    so a listing-based walk would hand back 1 290 takes with no GT.

    ``limit`` caps the takes taken *per dataset* — 0 for all of them. The cap is
    a real subsample and is therefore recorded in the manifest, printed in the
    report header, and included in the split digest.
    """
    out: List[Take] = []
    for ds in datasets:
        ds_dir = os.path.join(root, ds)
        if not os.path.isdir(ds_dir):
            raise SystemExit(
                f"[fovbench] --egosynth-root {root!r} has no {ds!r} directory. "
                f"Expected one of {DATASETS} (note: 'data_combined_5b' only "
                f"symlinks these and 'data_hot3d_5b' has no sparse depth).")
        found: List[Take] = []
        for name in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, name)
            meta_path = os.path.join(path, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            clips = sorted(
                (os.path.basename(p)[:-4]
                 for p in glob.glob(os.path.join(path, "sparse_depth", "*.npz"))),
                key=_clip_sort_key)
            if not clips:
                continue
            with open(meta_path) as fh:
                meta = json.load(fh)
            n_by_clip = {str(c["clip_name"]): int(c["num_frames"])
                         for c in meta.get("clips", ())}
            # A clip with depth but no row in the clip table has no frame count
            # and so no honest frame list; drop it rather than guess 121.
            clips = [c for c in clips if c in n_by_clip]
            if not clips:
                continue
            found.append(Take(dataset=ds, name=name, path=path,
                              rect=Rectification.from_meta(meta), clips=clips,
                              clip_frames={c: n_by_clip[c] for c in clips}))
        kept = found[:limit] if limit else found
        if verbose:
            note = f"{len(kept)} of {len(found)} takes" if limit else f"{len(found)} takes"
            print(f"  [fovbench] egosynth {ds}: {note}, "
                  f"{sum(len(t.clips) for t in kept)} clips")
        out.extend(kept)
    if not out:
        raise SystemExit(f"[fovbench] no take under {root!r} has sparse_depth/*.npz")
    return out


def _clip_sort_key(clip: str) -> Tuple[int, str]:
    """Clip ids are the start frame index in the source recording, so sort them
    numerically — ``'10000'`` before ``'2096'`` is the wrong order for a stride."""
    return (int(clip), clip) if clip.isdigit() else (1 << 62, clip)


# --------------------------------------------------------------------------- #
# Reading the ground truth
# --------------------------------------------------------------------------- #

def read_card(npz_path: str) -> dict:
    """The per-file ``meta`` key — the authority on that file's conventions.

    The data card is a description of the release; this is the file's own
    statement about itself, and the card says to read it before trusting the
    card. Cheap enough to read per clip.
    """
    z = np.load(npz_path, allow_pickle=True)
    return json.loads(str(z["meta"][0]))


@dataclass
class FramePoints:
    """The GT of one (clip, frame, view): a point list, never a map."""

    view: str                 # "rect" | "fisheye"
    u: np.ndarray             # (N,) float32, pixel x in the 896 frame
    v: np.ndarray             # (N,) float32, pixel y
    d: np.ndarray             # (N,) float32 metres, planar z about the camera axis
    inv_dist_std: np.ndarray  # (N,) float32, 1/m — scale-invariant triangulation quality
    dist_std: np.ndarray      # (N,) float32, m   — the depth's own 1-sigma
    radius: np.ndarray        # (N,) float32, distance from the optical centre, half-widths
    theta: Optional[np.ndarray]   # (N,) float32 deg — rect only; None on fisheye

    def __len__(self) -> int:
        return int(self.u.size)

    @property
    def index(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(vi, ui)`` int32, clipped — the gather index into an 896 map.

        ``np.rint`` is not enough on its own: ``u`` reaches exactly 895.5 and
        rint rounds half to even, so ``rint(895.5) == 896``, one past the end of
        an 896 array (gotcha 2). Clipping is the fix and it is not optional.
        """
        ui = np.clip(np.rint(self.u), 0, RES - 1).astype(np.int32)
        vi = np.clip(np.rint(self.v), 0, RES - 1).astype(np.int32)
        return vi, ui


def read_points(npz_path: str, view: str, frame: int, rect: Rectification,
                sigma_max: Optional[float] = DEFAULT_SIGMA_MAX,
                valid_mask: Optional[np.ndarray] = None) -> FramePoints:
    """One frame's points of one view, as float32 and ready to score.

    Casting out of float16 first is not cosmetic. The stored depth is ~0.05 %
    relative (~5 mm at 10 m, ~6 cm at the 120 m cap, gotcha 3) and float16
    arithmetic on top of that is needlessly lossy; every metric here is computed
    in float64 downstream of this cast.

    ``sigma_max`` cuts on ``inv_dist_std``, the scale-invariant column. The meta
    is explicit that it is *not* ``1/dist_std`` and not derivable from
    ``dist_std`` and depth, so the two columns are kept separately and only this
    one is cut on. ``None`` keeps every point.
    """
    variant = VIEW_TO_VARIANT[view]
    z = np.load(npz_path, allow_pickle=True)
    sel = z[f"{variant}_frame"].astype(np.int32) == int(frame)
    uvd = z[f"{variant}_uvd"][sel].astype(np.float32)
    inv = z[f"{variant}_inv_dist_std"][sel].astype(np.float32)
    dst = z[f"{variant}_dist_std"][sel].astype(np.float32)

    if sigma_max is not None and uvd.size:
        keep = np.isfinite(inv) & (inv < float(sigma_max))
        uvd, inv, dst = uvd[keep], inv[keep], dst[keep]

    u, v, d = (uvd[:, 0], uvd[:, 1], uvd[:, 2]) if uvd.size else (
        np.zeros(0, np.float32),) * 3

    if valid_mask is not None and u.size:
        # The rectified render does not fill its frame — the imaged cone covers
        # 86.1 % of it — and a point outside that cone has no pixel behind it.
        ui = np.clip(np.rint(u), 0, RES - 1).astype(np.int32)
        vi = np.clip(np.rint(v), 0, RES - 1).astype(np.int32)
        keep = valid_mask[vi, ui]
        u, v, d, inv, dst = u[keep], v[keep], d[keep], inv[keep], dst[keep]

    # Coordinates are read off the FLOAT u,v, not the rounded gather index: the
    # rounding exists only to index a map and would otherwise put up to half a
    # pixel of quantisation into the axis the errors are binned by.
    radius = (np.hypot(u - rect.cx, v - rect.cy) / (rect.render_size / 2.0)
              ).astype(np.float32)
    theta = None
    if view == "rect":
        theta = np.degrees(np.arctan(
            np.hypot(u - rect.cx, v - rect.cy) / rect.focal_px)).astype(np.float32)
    return FramePoints(view=view, u=u, v=v, d=d, inv_dist_std=inv, dist_std=dst,
                       radius=radius, theta=theta)


def coord_of(pts: FramePoints, axis: str) -> np.ndarray:
    """The binning coordinate of ``axis``, or refuse if this view has none."""
    if axis == "radius":
        return pts.radius
    if axis == "theta":
        if pts.theta is None:
            raise ThetaUnavailable(
                "theta is not computable on the raw fisheye stream of ego-synth "
                "5B: no fisheye camera model ships with this data (it needs the "
                "Aria calibration from the source VRS). Bin the fisheye arm by "
                "radius, or drop 'fisheye' from --views. Radius and theta are "
                "not interchangeable — see fovbench/README.md.")
        return pts.theta
    raise ValueError(f"unknown binning axis {axis!r} (choose 'theta' or 'radius')")


def load_valid_mask(path: str) -> Optional[np.ndarray]:
    """``rectified_valid_mask.png`` as a bool array, or None if absent."""
    if not os.path.isfile(path):
        return None
    import cv2
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

def decode_frames(path: str, indices: Iterable[int]) -> Dict[int, np.ndarray]:
    """Decode the given frame indices of one clip mp4 to RGB uint8.

    Read sequentially rather than seeking. A clip is 121 frames, inter-coded, so
    ``CAP_PROP_POS_FRAMES`` seeking is both slower in aggregate and — on some
    builds — off by a frame or two, which would silently score a prediction
    against the *next* frame's points. Sequential decode cannot do that.

    Frames are already upright (``meta.args.pose_orientation = "upright"``), so
    unlike the ADT path there is no 270-deg rotation here (gotcha 5).
    """
    import cv2
    want = sorted({int(i) for i in indices})
    if not want:
        return {}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"[fovbench] cannot open {path!r} — is OpenCV built "
                         f"with video support?")
    out: Dict[int, np.ndarray] = {}
    try:
        last, i = want[-1], 0
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                out[i] = np.ascontiguousarray(frame[..., ::-1])   # BGR -> RGB
            i += 1
    finally:
        cap.release()
    missing = [i for i in want if i not in out]
    if missing:
        raise SystemExit(f"[fovbench] {path!r} ended before frame(s) {missing}; "
                         f"the clip is short or truncated")
    return out


def sample_prediction(pred: np.ndarray, pts: FramePoints) -> np.ndarray:
    """The model's depth **gathered at the point list** — never the reverse.

    When the prediction is already at the GT's own 896 grid this is literally
    ``pred[vi, ui]``, which is the protocol the data card states. When the model
    was run at its own token grid instead (518 for VGGT/DA3, 512 for
    VGGT-Omega — the ADT path's rule, so that nothing is resampled between the
    view and the network), the points are mapped into that grid on the
    pixel-centre convention ``x' = (x + 0.5) * s - 0.5`` and sampled bilinearly.
    Which of the two ran is recorded in ``results.json``.

    The direction matters: scattering the GT into a map instead would drop ~20 %
    of the points to pixel collisions (gotcha 2) and pick an arbitrary winner
    among those that collide.
    """
    if pred.ndim != 2:
        raise ValueError(f"expected an (H, W) depth map, got shape {pred.shape}")
    H, W = pred.shape
    if (H, W) == (RES, RES):
        vi, ui = pts.index
        return pred[vi, ui].astype(np.float32)

    sx, sy = W / float(RES), H / float(RES)
    x = np.clip((pts.u + 0.5) * sx - 0.5, 0, W - 1)
    y = np.clip((pts.v + 0.5) * sy - 0.5, 0, H - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    p = pred.astype(np.float32)
    top = p[y0, x0] * (1 - fx) + p[y0, x1] * fx
    bot = p[y1, x0] * (1 - fx) + p[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


# --------------------------------------------------------------------------- #
# Pooling: fit per frame, bin once
# --------------------------------------------------------------------------- #

def _anchorable(gt: np.ndarray, coord: np.ndarray, lo: float, hi: float,
                mask: np.ndarray) -> Optional[np.ndarray]:
    """The band mask, if that band can determine a two-parameter affine.

    Same rule as :func:`fovbench.geometry.anchored_ratios` — populated *and*
    conditioned, where conditioned means an interquartile range worth
    ``MIN_ANCHOR_SPREAD`` of the median. Split out of that function so the fit
    can happen per frame while the ratio is taken after pooling; the rule itself
    is imported, not restated, so the two paths cannot drift apart.
    """
    b = mask & (coord >= lo) & (coord < hi)
    if int(b.sum()) < MIN_ANCHOR_POINTS:
        return None
    if G._relative_spread(gt, b) < G.MIN_ANCHOR_SPREAD:
        return None
    return b


class PointPool:
    """Per-frame fits, pooled across frames, binned once.

    The ADT driver bins each frame and averages the bins. That is not available
    here: a frame carries 1 300-5 300 points and can populate no bin at all
    within 30 deg of the axis (gotcha 10), so per-frame bins are empty at random
    and their average is over whichever frames happened to fill them.

    What is preserved exactly is the thing the protocol turns on — **the scale
    and shift are fitted once per frame, over every one of that frame's valid
    points, and then frozen.** Each forward pass of an up-to-scale model has its
    own arbitrary scale, so the fit cannot be pooled; the *aligned* depths can
    be, because after alignment they are all in the GT's metric units. Binning
    then happens once, on the pooled points, and is still a masking step applied
    to an already-frozen prediction.

    Raw per-frame arrays are held until :meth:`profile` so that the drift
    anchor can be chosen globally — see there.
    """

    def __init__(self, align_mode: str):
        self.align_mode = align_mode
        self._frames: List[dict] = []

    def add_frame(self, pred: np.ndarray, pts: FramePoints, axes: Sequence[str],
                  max_depth: float = 100.0, min_depth: float = 0.01) -> bool:
        """Accumulate one frame. Returns False if it was too thin to score."""
        gt = pts.d.astype(np.float32)
        ok = (np.isfinite(pred) & np.isfinite(gt) & (pred > 0)
              & (gt > min_depth) & (gt < max_depth))
        if int(ok.sum()) < MIN_FRAME_POINTS:
            return False
        self._frames.append({
            "pred": pred[ok], "gt": gt[ok],
            "coords": {a: coord_of(pts, a)[ok] for a in axes}})
        return True

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def profile(self, edges_by_axis: Dict[str, Sequence[float]],
                min_depth: float = 0.01, max_depth: float = 100.0,
                depth_strata: int = G.DEPTH_STRATA) -> Optional[dict]:
        """Metrics per bin on every axis, from one frozen fit per frame.

        The drift anchor is chosen **globally, before any fit**: the innermost
        band that at least ``MIN_ANCHOR_FRAME_FRAC`` of the frames can both
        populate and condition. Choosing it per frame instead would anchor one
        frame on 0-10 deg and another — a ``nymeria`` frame with nothing inside
        30 deg — on 30-40 deg, and pooling those two would compare quantities
        normalised at different eccentricities. Frames that cannot populate the
        chosen band contribute to every column except ``anchored_ratio``, and
        the count that did is reported as ``n_anchor_frames``.
        """
        if not self._frames:
            return None

        aligned = np.concatenate(
            [align_depth(f["pred"], f["gt"], np.ones(f["gt"].shape, bool),
                         mode=self.align_mode) for f in self._frames])
        gt = np.concatenate([f["gt"] for f in self._frames])
        mask = np.ones(gt.shape, bool)

        out = {"align": self.align_mode, "n_frame_valid": int(gt.size),
               "n_pool_frames": len(self._frames),
               "overall": depth_metrics(aligned, gt, mask, min_depth, max_depth)}
        # Unaligned ratios cannot be pooled: each frame's raw prediction is in
        # its own arbitrary scale, so their median would report the spread of
        # those constants rather than anything about depth. Say NaN, not a number.
        out["overall"]["raw_scale_ratio"] = float("nan")
        out["overall"].update(G._gt_stats(gt, mask))

        for axis, edges in edges_by_axis.items():
            edges = [float(e) for e in edges]
            coord = np.concatenate([f["coords"][axis] for f in self._frames])
            anchored, anchor_bin, n_anchor = self._anchored(axis, edges)
            std = G.standardise_by_depth(aligned, gt, mask, coord, edges,
                                         depth_strata, min_depth, max_depth)
            bins = []
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                m = mask & (coord >= lo) & (coord < hi)
                met = depth_metrics(aligned, gt, m, min_depth, max_depth)
                n = met.get("n_valid", 0)
                met["n_valid"] = 0 if not np.isfinite(n) else int(n)
                met["n_bin"] = int(m.sum())
                met["raw_scale_ratio"] = float("nan")
                met["anchored_ratio"] = (G.raw_scale_ratio(anchored, gt, m)
                                         if anchored is not None else float("nan"))
                met.update(G._gt_stats(gt, m))
                met.update(std[i])
                met["bin_lo"], met["bin_hi"] = lo, hi
                bins.append(met)
            out[axis] = bins
            out[f"{axis}_anchor_bin"] = anchor_bin
            out[f"{axis}_anchor_frames"] = n_anchor
        return out

    def _anchored(self, axis: str, edges: Sequence[float]
                  ) -> Tuple[Optional[np.ndarray], int, int]:
        """Pooled prediction re-fitted per frame on one shared anchor band."""
        bands = list(zip(edges[:-1], edges[1:]))
        need = max(1, int(math.ceil(MIN_ANCHOR_FRAME_FRAC * len(self._frames))))
        for j, (lo, hi) in enumerate(bands):
            per_frame = [_anchorable(f["gt"], f["coords"][axis], lo, hi,
                                     np.ones(f["gt"].shape, bool))
                         for f in self._frames]
            if sum(b is not None for b in per_frame) < need:
                continue
            parts = []
            for f, b in zip(self._frames, per_frame):
                if b is None:
                    parts.append(np.full(f["gt"].shape, np.nan, np.float32))
                else:
                    parts.append(align_depth(f["pred"], f["gt"], b,
                                             mode=self.align_mode))
            return (np.concatenate(parts), j,
                    int(sum(b is not None for b in per_frame)))
        return None, -1, 0
