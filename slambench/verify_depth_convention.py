# Copyright (c) 2026.
"""Is ego-synth's ``d`` planar z about the camera axis, or euclidean range?

Ticket 016. Every ``slambench`` number is scored against ``pts.d`` as planar z on
the data card's word — and the npz's own ``meta`` agrees, "metric camera-frame Z
(meters)". Neither is a measurement. If ``d`` is range and we score it as z,
every number carries a ``1/cos(theta)`` error: 1.00 on axis, 1.36 at 43 deg,
**1.74 at 55 deg**. It is radial, so the per-frame affine cannot absorb any of
it, and it lands hardest exactly where ``rect_derect`` is supposed to differ
from ``raw``.

Two things already in ``slambench`` look like checks of this and cannot be one
--------------------------------------------------------------------------------
Both are documented where they live; neither is used here.

1. **The rectified/fisheye depth agreement** (``baselines``). Euclidean range is
   a property of the *ray*, so it is exactly as invariant under a co-axial
   rectification as planar z is. It separates nothing.
2. **``verify_camera``'s twin residual.** ``predicted_pixels`` builds
   ``xyz = d * [(u-cx)/f, (v-cy)/f, 1]``; the range reading is ``d`` times that
   same vector *normalised*. They differ by the positive scalar ``|ray|``, and
   projection starts from ``x/z, y/z``. The predicted pixel is **bit-identical**
   under both hypotheses — that residual could go to zero and say nothing here.

What does decide it
-------------------
The source MPS semi-dense points, which are what ego-synth re-projected, in the
recording's world frame (``tools/fetch_egosynth_mps_points.py``), plus the
release's own ``camera_poses.json`` (``T_world_camera`` per frame, already
composed from ``closed_loop_trajectory`` and ``T_device_camera`` by the
producer). Bring a world point into the camera frame and *both* readings are
computable for it:

    p_cam = T_camera_world @ p_world
    z     = p_cam[2]
    range = |p_cam|          # = z / cos(theta)

**The matching must not presuppose either one.** A world point is matched to a
stored row by *ray direction alone* — where the point sits on the image sphere,
which is a function of ``u, v`` and the camera model and of no depth convention
whatsoever. Depth enters only after the correspondence is fixed. A match is kept
only when it is **unique** inside the angular tolerance, so a second point lying
along the same ray drops the row rather than resolving it by depth (which would
smuggle a hypothesis back in).

Then ``d`` is compared against ``z`` and against ``range`` for the same matched
points, **binned by theta** — the whole point, since on axis the two readings
agree and a pooled statistic would be dominated by the centre and read as
agreement either way.

A second, independent reading is reported beside it: reconstruct each stored row
into the world under each hypothesis and measure how far the reconstruction
lands from the actual point cloud. It shares no arithmetic with the regression
and answers the same question.

What "undecided" costs, so it does not get rounded to "z"
---------------------------------------------------------
The hypotheses are 1 % apart at 8 deg and 74 % apart at 55 deg. **If the matched
points do not reach past ~30 deg this has separated nothing**, and the honest
outcome is that the answer is still the data card's. :func:`verdict` refuses on
exactly that ground rather than reporting the smaller number as a winner.

Usage
-----
    python -m slambench.verify_depth_convention \\
        --egosynth-root ~/Desktop/ADT/ego-synth-5b-sample \\
        --points-root   ~/Desktop/ADT/ego-synth-5b-mps \\
        --calib-root    ~/Desktop/ADT/ego-synth-5b-calib \\
        --datasets aea,nymeria
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from slambench import camera as C

#: Theta bins (degrees). The last one is where the hypotheses are far apart; the
#: first is where they coincide and so is a built-in floor for both.
THETA_BINS: Tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 90.0)

#: Angular match tolerance, in pixels of the 896 frame. One pixel is the
#: tolerance ticket 016 specifies; the stored ``u, v`` are float16, whose
#: quantisation at 896 is already ~0.5 px.
MATCH_TOL_PX = 1.0

#: A bin needs at least this many uniquely matched points to be reported as
#: anything but empty.
MIN_BIN_POINTS = 200

#: The verdict needs this many points beyond :data:`SEPARATING_THETA` degrees,
#: where the two readings are far enough apart for the difference to survive
#: float16 depth.
SEPARATING_THETA = 30.0
MIN_SEPARATING_POINTS = 1000

#: Ratio by which the losing hypothesis' residual must exceed the winner's
#: before the result is called rather than left undecided.
DECISIVE_RATIO = 3.0


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def load_world_points(path: str) -> np.ndarray:
    """``(N, 3)`` world points from MPS's ``semidense_points.csv.gz``."""
    with gzip.open(path, "rt") as fh:
        header = fh.readline().strip().split(",")
        try:
            ix = [header.index(c) for c in ("px_world", "py_world", "pz_world")]
        except ValueError as e:                       # noqa: BLE001
            raise RuntimeError(f"{path}: no px_world/py_world/pz_world in "
                               f"{header}") from e
        rows = []
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) <= ix[2]:
                continue
            rows.append((p[ix[0]], p[ix[1]], p[ix[2]]))
    return np.asarray(rows, dtype=np.float64)


def load_poses(path: str) -> Dict[str, Dict[int, np.ndarray]]:
    """``{clip_name: {frame_index: T_world_camera (4x4)}}``.

    The release states this pose's own provenance in ``convention``; it is the
    producer's composition of the closed-loop trajectory with the RGB camera
    extrinsic, rolled to the upright image frame. Reading it is the reason this
    check does not have to fetch or interpolate a trajectory.
    """
    with open(path) as fh:
        doc = json.load(fh)
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for clip, entry in doc["clips"].items():
        frames = {}
        for fr in entry["frames"]:
            frames[int(fr["frame_index"])] = np.asarray(
                fr["T_world_camera"], dtype=np.float64).reshape(4, 4)
        out[str(clip)] = frames
    return out


def load_rows(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(frame_index, uvd)`` for the fisheye variant of one clip's npz.

    The fisheye set, not the rectified one: it is what ``slambench`` scores.
    """
    z = np.load(path)
    return (np.asarray(z["fisheye_frame"], dtype=np.int64),
            np.asarray(z["fisheye_uvd"], dtype=np.float64))


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #

def build_tree(cloud: np.ndarray):
    """A queryable index over ``cloud``, or the array itself without scipy.

    scipy when it is here, an exact blocked search when it is not — the same
    call ``verify_camera`` makes, and for the same reason: scipy is in this
    project's ``demo`` extra, and a check that cannot run without an optional
    package is a check nobody runs.

    The two build flags are pure speed and change no answer: they skip the
    median-split balancing, which on a 6.7 M-point recording is most of the
    build and buys a query this never makes enough of to notice.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return cloud
    return cKDTree(cloud, balanced_tree=False, compact_nodes=False)


def _nn(query: np.ndarray, tree, k: int = 1):
    """Nearest neighbour(s) for each row of ``query``, against :func:`build_tree`."""
    if isinstance(tree, np.ndarray):
        cloud = tree
        d = np.empty((query.shape[0], k))
        i = np.empty((query.shape[0], k), dtype=np.int64)
        for s in range(0, query.shape[0], 256):
            blk = query[s:s + 256]
            dd = np.linalg.norm(blk[:, None, :] - cloud[None, :, :], axis=-1)
            order = np.argsort(dd, axis=1)[:, :k]
            i[s:s + 256] = order
            d[s:s + 256] = np.take_along_axis(dd, order, axis=1)
        return (d[:, 0], i[:, 0]) if k == 1 else (d, i)
    # ``workers`` arrived in scipy 1.6. An acceptance test that cannot run on
    # the machine it is being read on is an acceptance test nobody runs, and
    # this file's own brute-force branch above exists for the same reason --
    # so an older scipy loses the threads, not the check.
    try:
        d, i = tree.query(query, k=k, workers=-1)
    except TypeError:
        d, i = tree.query(query, k=k)
    return (d, i)


def frame_pairs(cam: C.Fisheye624, uvd: np.ndarray, T_world_cam: np.ndarray,
                world: np.ndarray, tol_rad: float
                ) -> Optional[Dict[str, np.ndarray]]:
    """``d`` beside both readings, for one frame's uniquely matched rows.

    The correspondence is fixed on direction alone (see the module docstring),
    and a row whose second-nearest world direction also falls inside ``tol_rad``
    is dropped rather than resolved.
    """
    if uvd.shape[0] < 32:
        return None
    u, v, d = uvd[:, 0], uvd[:, 1], uvd[:, 2]
    keep = np.isfinite(u) & np.isfinite(v) & np.isfinite(d) & (d > 0)
    if keep.sum() < 32:
        return None
    u, v, d = u[keep], v[keep], d[keep]

    rays = cam.unproject(u, v)                       # (N, 3), unit
    theta = np.degrees(np.arccos(np.clip(rays[:, 2], -1.0, 1.0)))

    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    p_cam = (world - t) @ R                          # R^T (p_w - t), (M, 3)
    front = p_cam[:, 2] > 1e-6
    if front.sum() < 32:
        return None
    p_cam = p_cam[front]
    norm = np.linalg.norm(p_cam, axis=1)
    dirs = p_cam / norm[:, None]

    # Drop world points lying outside the angular support of this frame's rays.
    # Neutral by construction: a direction further from the axis than the widest
    # ray, by more than the tolerance, cannot be within the tolerance of any of
    # them, so it can neither win a match nor spoil a uniqueness test. Note this
    # is the ONLY filter applied — in particular nothing is dropped by distance,
    # because a far point along a stored point's ray is genuine ambiguity and
    # removing it would manufacture unique matches.
    cos_max = np.cos(np.radians(min(90.0, float(theta.max()) + 2.0)))
    near = dirs[:, 2] >= cos_max
    if near.sum() < 32:
        return None
    p_cam, norm, dirs = p_cam[near], norm[near], dirs[near]

    # Chord distance is monotone in angle, so an angular tolerance is a radius.
    dist, idx = _nn(rays, build_tree(dirs), k=2)
    chord = 2.0 * np.sin(tol_rad / 2.0)
    unique = (dist[:, 0] <= chord) & (dist[:, 1] > chord)
    if unique.sum() < 8:
        return None

    m = idx[unique, 0]
    return {
        "d": d[unique],
        "z": p_cam[m, 2],
        "range": norm[m],
        "theta": theta[unique],
        "n_rows": np.int64(u.size),
        "n_unique": np.int64(unique.sum()),
    }


def reconstruction_error(cam: C.Fisheye624, uvd: np.ndarray,
                         T_world_cam: np.ndarray, world_tree
                         ) -> Optional[Dict[str, np.ndarray]]:
    """How far each hypothesis' reconstructed point lands from the cloud.

    Independent of :func:`frame_pairs` — it never matches anything, so it cannot
    inherit a matching bias. Under the true reading the reconstruction *is* a
    subset of the cloud and the distance collapses to the float16 floor.
    """
    if uvd.shape[0] < 32:
        return None
    u, v, d = uvd[:, 0], uvd[:, 1], uvd[:, 2]
    keep = np.isfinite(u) & np.isfinite(v) & np.isfinite(d) & (d > 0)
    if keep.sum() < 32:
        return None
    u, v, d = u[keep], v[keep], d[keep]
    rays = cam.unproject(u, v)
    theta = np.degrees(np.arccos(np.clip(rays[:, 2], -1.0, 1.0)))

    # z-hypothesis: d is the third component, so scale the ray to reach it.
    p_z = rays * (d / np.clip(rays[:, 2], 1e-9, None))[:, None]
    # range-hypothesis: d is the length of the ray.
    p_r = rays * d[:, None]

    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    w_z = p_z @ R.T + t
    w_r = p_r @ R.T + t
    return {
        "z": _nn(w_z, world_tree)[0],
        "range": _nn(w_r, world_tree)[0],
        "theta": theta,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def bin_stats(theta: np.ndarray, resid: Dict[str, np.ndarray]
              ) -> List[dict]:
    """Median residual per hypothesis per theta bin."""
    out = []
    for lo, hi in zip(THETA_BINS[:-1], THETA_BINS[1:]):
        m = (theta >= lo) & (theta < hi)
        row = {"lo": lo, "hi": hi, "n": int(m.sum())}
        for name, r in resid.items():
            row[name] = float(np.median(np.abs(r[m]))) if m.sum() else float("nan")
        out.append(row)
    return out


def verdict(theta: np.ndarray, rel_z: np.ndarray, rel_r: np.ndarray) -> str:
    """``z``, ``range`` or ``undecided`` — and undecided is a real outcome.

    Refuses on coverage before it compares anything: the two hypotheses are 1 %
    apart at 8 deg, so a check whose points never leave the centre has not
    separated them however clean its medians look.
    """
    far = theta >= SEPARATING_THETA
    if far.sum() < MIN_SEPARATING_POINTS:
        return (f"undecided — only {int(far.sum())} matched points beyond "
                f"{SEPARATING_THETA:.0f} deg (need {MIN_SEPARATING_POINTS}); "
                f"the readings are too close together below that to separate")
    mz = float(np.median(np.abs(rel_z[far])))
    mr = float(np.median(np.abs(rel_r[far])))
    if mz <= mr / DECISIVE_RATIO:
        return f"z — planar camera-frame Z (|rel| {mz:.4f} vs {mr:.4f} beyond {SEPARATING_THETA:.0f} deg)"
    if mr <= mz / DECISIVE_RATIO:
        return f"range — euclidean distance (|rel| {mr:.4f} vs {mz:.4f} beyond {SEPARATING_THETA:.0f} deg)"
    return (f"undecided — neither reading wins by {DECISIVE_RATIO}x beyond "
            f"{SEPARATING_THETA:.0f} deg (z {mz:.4f}, range {mr:.4f})")


def verify_take(egosynth_root: str, points_root: str, calib_root: str,
                ds: str, take: str, n_clips: int, n_frames: int) -> Optional[dict]:
    take_dir = os.path.join(egosynth_root, ds, take)
    pts_path = os.path.join(points_root, ds, take, "semidense_points.csv.gz")
    if not os.path.isdir(take_dir):
        print(f"  {ds}/{take}: not in the release root — skipping")
        return None
    if not os.path.exists(pts_path):
        print(f"  {ds}/{take}: no semidense points — run "
              f"tools/fetch_egosynth_mps_points.py")
        return None

    cam = C.load(C.calibration_path(calib_root, ds, take), dataset=ds, take=take)
    world = load_world_points(pts_path)
    poses = load_poses(os.path.join(take_dir, "camera_poses.json"))
    tol_rad = MATCH_TOL_PX / cam.f
    # The world cloud never moves, so its index is built once for the take
    # rather than once per frame -- on Nymeria that is 6.7 M points a frame.
    world_tree = build_tree(world)

    clips = sorted(os.listdir(os.path.join(take_dir, "sparse_depth")))[:n_clips]
    D, Z, RG, TH = [], [], [], []
    rc_z, rc_r, rc_th = [], [], []
    n_rows = n_uniq = 0
    for clip_file in clips:
        clip = os.path.splitext(clip_file)[0]
        if clip not in poses:
            continue
        frame_ix, uvd = load_rows(os.path.join(take_dir, "sparse_depth", clip_file))
        picks = sorted(set(frame_ix.tolist()))[:n_frames]
        for fi in picks:
            T = poses[clip].get(int(fi))
            if T is None:
                continue
            rows = uvd[frame_ix == fi]
            got = frame_pairs(cam, rows, T, world, tol_rad)
            if got is not None:
                D.append(got["d"]); Z.append(got["z"])
                RG.append(got["range"]); TH.append(got["theta"])
                n_rows += int(got["n_rows"]); n_uniq += int(got["n_unique"])
            rc = reconstruction_error(cam, rows, T, world_tree)
            if rc is not None:
                rc_z.append(rc["z"]); rc_r.append(rc["range"]); rc_th.append(rc["theta"])

    if not D:
        print(f"  {ds}/{take}: nothing matched")
        return None
    d = np.concatenate(D); z = np.concatenate(Z)
    rg = np.concatenate(RG); th = np.concatenate(TH)
    rel_z = (d - z) / np.clip(z, 1e-6, None)
    rel_r = (d - rg) / np.clip(rg, 1e-6, None)

    print(f"\n  {ds}/{take}")
    print(f"    lens f={cam.f:.2f} c=({cam.cx:.1f},{cam.cy:.1f}) rot={cam.rotation} "
          f"| {world.shape[0]:,} world points | {len(clips)} clip(s)")
    print(f"    {n_rows:,} stored rows -> {n_uniq:,} uniquely matched "
          f"({100.0 * n_uniq / max(n_rows, 1):.1f} %) at {MATCH_TOL_PX:.0f} px")
    print(f"\n    d vs each reading, |relative| median, by incidence angle")
    print(f"      each hypothesis predicts the OTHER's residual, and they are")
    print(f"      different functions:  d=z => |d-range|/range = 1-cos(theta)")
    print(f"                            d=range => |d-z|/z     = sec(theta)-1")
    print(f"      {'theta':>12s} {'n':>9s} {'|d-z|/z':>10s} {'|d-r|/r':>10s}"
          f" {'pred 1-cos':>12s} {'pred sec-1':>12s}")
    for row in bin_stats(th, {"z": rel_z, "range": rel_r}):
        mid = math_mid(row["lo"], row["hi"], th)
        if mid == mid:
            cos = np.cos(np.radians(mid))
            pred_cos, pred_sec = 1.0 - cos, 1.0 / cos - 1.0
        else:
            pred_cos = pred_sec = float("nan")
        flag = "" if row["n"] >= MIN_BIN_POINTS else "  (thin)"
        print(f"      {row['lo']:5.0f}-{row['hi']:<6.0f} {row['n']:>9,} "
              f"{row['z']:>10.4f} {row['range']:>10.4f}"
              f" {pred_cos:>12.4f} {pred_sec:>12.4f}{flag}")

    if rc_z:
        rz = np.concatenate(rc_z); rr = np.concatenate(rc_r); rt = np.concatenate(rc_th)
        print(f"\n    independent check — distance from the reconstructed point "
              f"to the cloud (metres)")
        print(f"      {'theta':>12s} {'n':>9s} {'as z':>10s} {'as range':>12s}")
        for row in bin_stats(rt, {"z": rz, "range": rr}):
            print(f"      {row['lo']:5.0f}-{row['hi']:<6.0f} {row['n']:>9,} "
                  f"{row['z']:>10.4f} {row['range']:>12.4f}")

    v = verdict(th, rel_z, rel_r)
    print(f"\n    VERDICT: {v}")
    return {"dataset": ds, "take": take, "verdict": v,
            "n_matched": int(th.size), "n_rows": n_rows,
            "bins": bin_stats(th, {"z": rel_z, "range": rel_r})}


def math_mid(lo: float, hi: float, theta: np.ndarray) -> float:
    """Median theta actually present in a bin — the honest place to quote sec()."""
    m = (theta >= lo) & (theta < hi)
    return float(np.median(theta[m])) if m.any() else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--egosynth-root", required=True)
    p.add_argument("--points-root", required=True,
                   help="where tools/fetch_egosynth_mps_points.py wrote")
    p.add_argument("--calib-root", required=True)
    p.add_argument("--datasets", default="aea,nymeria")
    p.add_argument("--takes", default=None,
                   help="comma-separated take names; default is every take "
                        "present in both roots")
    p.add_argument("--clips", type=int, default=4, help="clips per take")
    p.add_argument("--frames", type=int, default=3, help="frames per clip")
    a = p.parse_args()

    ego = os.path.expanduser(a.egosynth_root)
    pts = os.path.expanduser(a.points_root)
    cal = os.path.expanduser(a.calib_root)
    print("=" * 78)
    print("  ego-synth depth convention — is `d` planar z, or euclidean range?")
    print("=" * 78)
    if not C.reference_available():
        print("  note: projectaria_tools absent — using the numpy fallback lens")

    out = []
    for ds in [s for s in a.datasets.split(",") if s]:
        takes = ([s for s in a.takes.split(",") if s] if a.takes
                 else sorted(os.listdir(os.path.join(pts, ds)))
                 if os.path.isdir(os.path.join(pts, ds)) else [])
        for take in takes:
            r = verify_take(ego, pts, cal, ds, take, a.clips, a.frames)
            if r:
                out.append(r)

    print("\n" + "=" * 78)
    for r in out:
        print(f"  {r['dataset']:9s} {r['take'][:44]:46s} {r['verdict']}")
    if not out:
        print("  nothing verified")
    print("=" * 78)


if __name__ == "__main__":
    main()
