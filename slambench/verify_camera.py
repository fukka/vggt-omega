# Copyright (c) 2026.
"""Does the fetched calibration actually describe ego-synth's fisheye stream?

    python -m slambench.verify_camera --egosynth-root ... --calib-root ...

``rect_derect`` maps every prediction through the camera model. An error there
does not make a score worse — it samples a different part of the image — so the
model and its orientation have to be *checked*, not assumed, and
``camera.require_verified`` refuses to hand out a camera until they are.

What makes a check possible at all
----------------------------------
ego-synth ships the same SLAM points projected **twice**: into the raw fisheye,
and into a pinhole whose every parameter is stated (110 deg, focal 313.693,
render 896, principal point at :func:`rect_centre`). The two streams are
co-axial — the producer's own depths agree verbatim between them — so a
rectified point back-projects to a ray in the same camera frame the fisheye
point lives in.

Push those rays through a candidate camera and they must land where the fisheye
points actually are. That closes the loop with no free parameters.

**It is blind to the depth convention, by construction.** :func:`predicted_pixels`
builds ``xyz = d * [(u-c)/f, (v-c)/f, 1]``, which is what a *planar z* reading of
``d`` gives; a *range* reading gives ``d`` times the same vector normalised. The
two differ only by the positive scalar ``|ray|``, and :meth:`Fisheye624.project`
begins by forming ``x/z, y/z`` — so both hypotheses produce a bit-identical
pixel and this file cannot tell them apart however small its residual gets. It
verifies the camera model and its orientation, which is all it claims. What
``d`` means is ticket 016.

**It passes.** On both staged datasets the 90 deg rotation reads NN median
0.29 px with 96.9 % of points inside 1 px, and the other three quarter turns sit
at 4-6 px. Getting there took three corrections and none of them was a fit; the
last was in this file, and :func:`rect_centre` is where it is written down.

Why this compares point *clouds* and not point *pairs*
------------------------------------------------------
The obvious version pairs a rectified point with "the fisheye point at the same
depth" and measures the residual. That was tried and it is **contaminated**: the
two lists are different subsets of the same points, depth is stored as float16,
and two unrelated points sharing a depth get paired. About 90 % of such pairs
were false, which put the median residual at 5-15 px while genuinely matched
pairs sat at 0.20-0.50 px — a statistic reporting the contamination rate rather
than the camera.

So no pairing is attempted. Each predicted pixel is compared to the **nearest**
actual fisheye pixel, and the statistic is how close the two clouds are as sets.
A wrong camera or a wrong orientation displaces the whole cloud, which no
nearest-neighbour search can rescue. The discrimination is wide: a frame carries
1 300-5 300 points in an 896 frame, so the mean spacing is 12-25 px and the
chance of a randomly placed point falling within 1 px of a real one is 0.5-2 %,
against the ~100 % a correct model must produce.

The four quarter turns are their own control. Exactly one of them should pass;
if two do, the statistic is not discriminating and the answer is "unverified",
not "probably that one".

Two statistics, because the first one lies about magnitude
----------------------------------------------------------
The nearest-neighbour distance is a good **detector** and a bad **ruler**. In a
cloud this dense there is nearly always *some* unrelated point a few pixels away,
so the statistic saturates at roughly the local point spacing however wrong the
camera is. Measured on a *wrong* rotation of the take below: NN median 4.12 px,
while the distance to the point actually being predicted was **13.20 px**. The
twin residual is what made the crop and the centre findable at all — the NN
statistic reads 4 px whether the camera is 6 px out or 13.

So :func:`twin_residual` is reported beside it. It uses the subset of points
whose depth appears **exactly once** in the fisheye set — about 13 % of a frame,
unambiguous by construction rather than by luck, which is what the earlier
contaminated pairing lacked. Membership is not the problem here: 99.8 % of
rectified points do have a same-depth twin, so the subset is representative
rather than a leftover.

Read them together. NN says *whether* the camera is right; the twin residual says
*by how much* it is wrong.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench import camera as C  # noqa: E402
from slambench import data as D  # noqa: E402

#: The producer's rectified render, from ``meta.rectification``. Read per take
#: rather than assumed, but every take in the release measured the same.
RECT_FOV_DEG = 110.0

#: A rotation passes only if it also beats the runner-up by this factor on the
#: within-1px share. Two rotations that both look good mean the statistic is not
#: discriminating on this take, which is a reason to report nothing.
MIN_MARGIN = 3.0

#: ...and only if it puts at least this share of points within a pixel. Chance is
#: 0.5-2 %, so this is far above the null while leaving room for the float16
#: quantisation of the stored coordinates.
MIN_WITHIN_1PX = 0.80


@dataclass
class RotationResult:
    rotation: int
    median_px: float
    within_half: float
    within_1: float
    within_2: float
    n_points: int
    twin_px: float = float("nan")
    twin_within_2: float = float("nan")
    n_twin: int = 0

    def __str__(self) -> str:
        return (f"rot {self.rotation * 90:3d} deg | NN median {self.median_px:6.2f} px "
                f"<1px {100 * self.within_1:5.1f}% "
                f"<2px {100 * self.within_2:5.1f}% "
                f"| twin {self.twin_px:7.2f} px "
                f"<2px {100 * self.twin_within_2:5.1f}% (n={self.n_twin})")


def rectification_of(take_dir: str) -> Tuple[float, float, int]:
    """``(fov_deg, focal_px, render_size)`` from the take's own ``meta.json``."""
    with open(os.path.join(take_dir, "meta.json")) as fh:
        r = json.load(fh)["rectification"]
    return float(r["fov_deg"]), float(r["focal_px"]), int(r["render_size"])


def rect_centre(render: int) -> float:
    """The rectified render's principal point: ``(N - 1) / 2``, not ``N / 2``.

    A pixel's centre is at its integer coordinate, so an ``N`` square image is
    centred on ``(N - 1) / 2``. This is what ``projectaria_tools``'
    ``get_linear_camera_calibration`` uses, and what ``baselines.Pinhole`` uses;
    this function exists because *this file* used ``N / 2`` and the half pixel
    was the last of the residual.

    It does not arrive at the fisheye as half a pixel. The back-projection
    divides by the pinhole's focal and the fisheye projection multiplies by its
    own, so the error is magnified by ``386.28 / 313.69 = 1.23`` to 0.62 px —
    and the residual it left behind was a constant ``(+0.71, +0.71)``, flat in
    radius, on both datasets. Correcting it takes the twin residual from 1.03 px
    to 0.30 px.
    """
    return (render - 1) / 2.0


def read_variant(npz_path: str, variant: str, frame: int) -> np.ndarray:
    """``(N, 3)`` of ``u, v, depth`` for one frame of one stream, float32.

    ``rectified`` is read **only here**. It is not ground truth for this
    evaluation — scoring against a pre-rectified point set would bake one
    baseline's choice into the measurement — and this is the one place it earns
    its keep, as an independent statement of where each ray went.
    """
    z = np.load(npz_path, allow_pickle=True)
    sel = z[f"{variant}_frame"].astype(np.int32) == int(frame)
    return z[f"{variant}_uvd"][sel].astype(np.float32)


def predicted_pixels(rect_uvd: np.ndarray, focal: float, centre: float,
                     cam: C.Fisheye624) -> np.ndarray:
    """Rectified points -> rays -> the fisheye pixels a candidate camera predicts."""
    u, v, d = rect_uvd[:, 0], rect_uvd[:, 1], rect_uvd[:, 2]
    x = (u - centre) / focal * d
    y = (v - centre) / focal * d
    xyz = np.stack([x, y, d], axis=1).astype(np.float64)
    pu, pv = cam.project(xyz)
    return np.stack([pu, pv], axis=1)


def twin_residual(pred: np.ndarray, rect_depth: np.ndarray,
                  fish: np.ndarray) -> Tuple[float, float, int]:
    """``(median px, share within 2 px, n)`` against the point actually predicted.

    The honest magnitude, on the subset where "which fisheye point is this?" has
    one answer: those whose depth occurs **exactly once** among the fisheye
    points. Requiring uniqueness is what the earlier, discarded pairing failed to
    do — it matched on depth alone and paired unrelated points that happened to
    share one.

    Returns NaNs when the subset is too small to say anything.
    """
    seen: Dict[float, List[int]] = {}
    for i, d in enumerate(fish[:, 2].astype(np.float64)):
        seen.setdefault(round(float(d), 6), []).append(i)
    idx, rows = [], []
    for r, d in enumerate(rect_depth.astype(np.float64)):
        cand = seen.get(round(float(d), 6), ())
        if len(cand) == 1:
            idx.append(cand[0])
            rows.append(r)
    if len(rows) < 32:
        return float("nan"), float("nan"), len(rows)
    p = pred[rows]
    q = fish[idx][:, :2]
    ok = np.isfinite(p).all(axis=1)
    if ok.sum() < 32:
        return float("nan"), float("nan"), int(ok.sum())
    dist = np.hypot(p[ok, 0] - q[ok, 0], p[ok, 1] - q[ok, 1])
    return float(np.median(dist)), float((dist < 2.0).mean()), int(ok.sum())


def nn_distances(pred: np.ndarray, actual: np.ndarray) -> Optional[np.ndarray]:
    """Distance from each finite predicted pixel to the nearest actual one.

    The primitive both :func:`cloud_distance` and :func:`verify_take` are built
    on, so the frame-level statistic the tests pin and the pooled statistic the
    CLI reports cannot drift apart. ``None`` when either cloud is too small to
    say anything.
    """
    pred = pred[np.isfinite(pred).all(axis=1)]
    if pred.shape[0] < 32 or actual.shape[0] < 32:
        return None
    xy = np.ascontiguousarray(actual[:, :2], dtype=np.float64)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _nn_brute(pred, xy)
    return cKDTree(xy).query(pred, k=1)[0]


def _nn_brute(pred: np.ndarray, actual: np.ndarray,
              chunk: int = 512) -> np.ndarray:
    """:func:`nn_distances` without scipy — every pair, in blocks.

    scipy is in this project's ``demo`` extra and not its core dependencies, and
    an acceptance test that cannot run because of a missing optional package is
    an acceptance test nobody runs. A kd-tree written here would be a second
    thing to get right; this is the definition of the answer instead, and
    ``test_verify_camera.py`` pins it against scipy's.

    Two clouds of ~5 000 points cost ~25 M distances a frame, which is about
    100 ms of numpy — a hundred times slower than the kd-tree and irrelevant at
    the handful of takes this test looks at.
    """
    out = np.empty(pred.shape[0], dtype=np.float64)
    for i in range(0, pred.shape[0], chunk):
        blk = pred[i:i + chunk]
        d2 = ((blk[:, None, 0] - actual[None, :, 0]) ** 2
              + (blk[:, None, 1] - actual[None, :, 1]) ** 2)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def cloud_distance(pred: np.ndarray, actual: np.ndarray) -> RotationResult:
    """One frame's worth of :func:`nn_distances`, as a :class:`RotationResult`."""
    dist = nn_distances(pred, actual)
    if dist is None:
        n = int(np.isfinite(pred).all(axis=1).sum())
        return RotationResult(-1, float("nan"), 0.0, 0.0, 0.0, n)
    return RotationResult(-1, float(np.median(dist)),
                          float((dist < 0.5).mean()), float((dist < 1.0).mean()),
                          float((dist < 2.0).mean()), int(dist.size))


def verify_take(take_dir: str, calib_json: str, dataset: str,
                frames: Sequence[int] = (0, 40, 80, 120),
                clips: int = 2) -> Dict[int, RotationResult]:
    """Each of the four quarter turns, scored over several frames of a take."""
    fov, focal, render = rectification_of(take_dir)
    centre = rect_centre(render)
    npzs = sorted(glob.glob(os.path.join(take_dir, "sparse_depth", "*.npz")))[:clips]
    if not npzs:
        raise SystemExit(f"[slambench] {take_dir} has no sparse_depth/*.npz")

    out: Dict[int, RotationResult] = {}
    for k in range(4):
        cam = C.load(calib_json, dataset=dataset, out_size=render, rotation=k)
        d_all, n_all, t_all = [], 0, []
        for npz in npzs:
            for fr in frames:
                rect = read_variant(npz, "rectified", fr)
                fish = read_variant(npz, "fisheye", fr)
                if rect.shape[0] < 32 or fish.shape[0] < 32:
                    continue
                pred = predicted_pixels(rect, focal, centre, cam)
                dist = nn_distances(pred, fish)
                if dist is None:
                    continue
                d_all.append(dist)
                n_all += int(dist.size)
                t_all.append(twin_residual(pred, rect[:, 2], fish))
        if not d_all:
            out[k] = RotationResult(k, float("nan"), 0.0, 0.0, 0.0, 0)
            continue
        dist = np.concatenate(d_all)
        tw = [t for t in t_all if np.isfinite(t[0])]
        out[k] = RotationResult(
            k, float(np.median(dist)), float((dist < 0.5).mean()),
            float((dist < 1.0).mean()), float((dist < 2.0).mean()), n_all,
            twin_px=float(np.mean([t[0] for t in tw])) if tw else float("nan"),
            twin_within_2=float(np.mean([t[1] for t in tw])) if tw else float("nan"),
            n_twin=int(sum(t[2] for t in tw)))
    return out


def decide(results: Dict[int, RotationResult]) -> Tuple[Optional[int], str]:
    """The winning rotation, or ``None`` with the reason it cannot be called.

    Three ways to fail, and each is reported rather than rounded past: nobody
    clears the bar, the winner does not clear it, or two rotations do — which
    means the statistic is not separating them on this take and the honest
    answer is that nothing was verified.
    """
    ranked = sorted(results.values(), key=lambda r: -r.within_1)
    best, runner = ranked[0], ranked[1]
    if best.within_1 < MIN_WITHIN_1PX:
        return None, (f"best rotation puts only {100 * best.within_1:.1f}% of "
                      f"points within 1 px (need {100 * MIN_WITHIN_1PX:.0f}%); "
                      f"the model or the resolution convention is wrong, not "
                      f"just the rotation")
    if runner.within_1 > 0 and best.within_1 < MIN_MARGIN * runner.within_1:
        return None, (f"rotations {best.rotation * 90} and {runner.rotation * 90} "
                      f"deg score {100 * best.within_1:.1f}% and "
                      f"{100 * runner.within_1:.1f}% — not separated, so nothing "
                      f"is verified")
    if best.median_px > C.ORIENTATION_TOL_PX:
        return None, (f"rotation {best.rotation * 90} deg wins but its median is "
                      f"{best.median_px:.2f} px, above the "
                      f"{C.ORIENTATION_TOL_PX} px floor")
    return best.rotation, "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--egosynth-root", required=True)
    p.add_argument("--calib-root", required=True)
    p.add_argument("--datasets", default=",".join(D.DATASETS))
    p.add_argument("--takes", type=int, default=2,
                   help="takes per dataset to check")
    p.add_argument("--clips", type=int, default=2, help="clips per take")
    a = p.parse_args()

    verdicts: Dict[str, object] = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        ds_dir = os.path.join(a.egosynth_root, ds)
        if not os.path.isdir(ds_dir):
            print(f"\n=== {ds}: not present under {a.egosynth_root}")
            continue
        names = sorted(n for n in os.listdir(ds_dir)
                       if os.path.isfile(os.path.join(ds_dir, n, "meta.json")))
        checked = 0
        for name in names:
            calib = C.calibration_path(a.calib_root, ds, name)
            if not os.path.isfile(calib):
                continue
            print(f"\n=== {ds} / {name}")
            res = verify_take(os.path.join(ds_dir, name), calib, ds,
                              clips=a.clips)
            for k in sorted(res):
                print("    " + str(res[k]))
            win, why = decide(res)
            verdicts.setdefault(ds, []).append(win)
            print(f"    -> {'rotation ' + str(win * 90) + ' deg' if win is not None else 'UNVERIFIED'}"
                  f"{'' if win is not None else ': ' + why}")
            checked += 1
            if checked >= a.takes:
                break
        if not checked:
            print(f"\n=== {ds}: no calibration under {a.calib_root} "
                  f"(fetch it -- ticket 012)")

    print("\n" + "=" * 72)
    for ds, wins in verdicts.items():
        agree = len(set(wins)) == 1 and wins[0] is not None
        per_take = ", ".join("UNVERIFIED" if w is None else f"{w * 90} deg"
                             for w in wins)
        print(f"  {ds:10s} {'rotation ' + str(wins[0] * 90) + ' deg' if agree else 'UNVERIFIED'}"
              f"   ({len(wins)} take(s): {per_take})")
        if agree and ds not in C.VERIFIED_ROTATION:
            print(f"{'':13s}not yet in camera.VERIFIED_ROTATION — add it by hand")
        elif not agree and ds in C.VERIFIED_ROTATION:
            print(f"{'':13s}** {ds} IS in camera.VERIFIED_ROTATION but did not "
                  f"pass here — remove it **")
    print("\n  Add a dataset to camera.VERIFIED_ROTATION BY HAND once every take")
    print("  checked agrees. A measurement that promotes itself is not a check.")


if __name__ == "__main__":
    main()
