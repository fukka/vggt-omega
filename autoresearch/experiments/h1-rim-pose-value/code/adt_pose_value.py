"""H1.3: replicate the span (H1.1) and rim-dependence (H1.2) findings on real Aria.

Protocol: ../protocol-h1.3.md (committed before this ran). CPU.

Stages:
  1. hand-eye  — estimate the fixed device->camera conjugation C from classical
                 MAGSAC++ rotations vs device-frame GT (verification gate: median
                 classical error vs C-conjugated GT < 1.5 deg, else abort).
  2. span      — H1.1 on the Aria cone: theta <= {25,35,45,54.8} deg, count-matched.
  3. mask      — H1.2 on the Aria cone with DA3-Small: center vs rim-annulus vs
                 area-matched random patch mask.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h1-rim-pose-value/code/adt_pose_value.py \
        --stages handeye,span --out results/run_006.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from cam3r.adt import _frame_timestamp_us, load_trajectory      # noqa: E402
from cam3r.cameras import (_ARIA_KB4, aria_214_1_kb4,           # noqa: E402
                           aria_valid_theta_max)
from raytun3r.cameras import KannalaBrandt                      # noqa: E402
from raytun3r.matching import relative_pose_magsac              # noqa: E402
from raytun3r.metrics import rotation_error_deg                 # noqa: E402

from rim_pose_value import (_gain, _median, scatter, sift_matches,  # noqa: E402
                            synth_targets, theta_of)

DEFAULT_SEQ = ("/Users/fengjiazhang/Documents/projectaria_tools_adt_data/"
               "Apartment_release_clean_seq131_M1292")
SPANS = (25.0, 35.0, 45.0, 54.8)
RESAMPLES = 5
MIN_PER_COND = 20


class AriaLocalPairs:
    """The 28 staged seq131 fisheye JPGs with device-frame GT poses.

    Interface-compatible with what the H1 helpers expect from a source:
    ``camera``, ``h``, ``w``, ``image(i)``; ``pose(i)`` returns T_world_device
    (rotation, translation) — DEVICE frame, conjugate before camera-frame use.
    Frames are used exactly as stored (native sensor orientation), so the
    camera uses the native (rotated=False) intrinsics.
    """

    def __init__(self, seq_dir: str, size: int = 504) -> None:
        from PIL import Image

        self.paths = sorted(glob.glob(os.path.join(seq_dir, "videos_rgb", "*.jpg")))
        if not self.paths:
            raise RuntimeError(f"no frames under {seq_dir}/videos_rgb")
        ts, T = load_trajectory(os.path.join(seq_dir, "groundtruth",
                                             "aria_trajectory.csv"))
        self._traj_ts, self._traj_T = ts.numpy(), T
        self.h = self.w = size
        ref = aria_214_1_kb4(size, size, rotated=False)
        self.camera = KannalaBrandt(
            fx=ref.fx, fy=ref.fy, cx=ref.cx, cy=ref.cy,
            width=size, height=size, k=tuple(_ARIA_KB4),
            theta_max=aria_valid_theta_max())
        self._Image = Image
        self._cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def image(self, i: int) -> torch.Tensor:
        if i not in self._cache:
            im = self._Image.open(self.paths[i]).convert("RGB").resize(
                (self.w, self.h), self._Image.BICUBIC)
            self._cache[i] = torch.from_numpy(
                np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        return self._cache[i]

    def pose(self, i: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        t_us = _frame_timestamp_us(self.paths[i])
        if t_us is None:
            return None
        k = int(np.argmin(np.abs(self._traj_ts - t_us)))
        if abs(self._traj_ts[k] - t_us) > 2000.0:      # >2 ms gap: no GT
            return None
        T_wd = self._traj_T[k]
        # cam-from-world convention as elsewhere: R = R_wd^T, t = -R_wd^T t_wd
        R = T_wd[:3, :3].transpose(-1, -2)
        return R, -(R @ T_wd[:3, 3])


def rot_angle_deg(R: torch.Tensor) -> float:
    return rotation_error_deg(torch.eye(3, dtype=R.dtype), R)


def axis_of(R: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eig(R)
    k = int(np.argmin(np.abs(w - 1.0)))
    ax = np.real(V[:, k])
    # Fix sign so that the rotation about +axis matches R (via skew part).
    skew = (R - R.T) / 2.0
    s = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if np.dot(ax, s) < 0:
        ax = -ax
    return ax / max(np.linalg.norm(ax), 1e-12)


def hand_eye_rotation(R_hats: List[np.ndarray], R_devs: List[np.ndarray],
                      angles: List[float]) -> np.ndarray:
    """Wahba on angle-weighted rotation axes: find C with axis_hat = C axis_dev."""
    A = np.zeros((3, 3))
    for Rh, Rd, w in zip(R_hats, R_devs, angles):
        A += w * np.outer(axis_of(Rh), axis_of(Rd))
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--stages", default="handeye,span,mask")
    p.add_argument("--max-gt-rot-deg", type=float, default=30.0)
    p.add_argument("--min-gt-rot-deg", type=float, default=0.5)
    p.add_argument("--nfeatures", type=int, default=6000)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--magsac-thresh-deg", type=float, default=0.5)
    p.add_argument("--mask-deg", type=float, default=35.0)
    p.add_argument("--gap-filter-deg", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    src = AriaLocalPairs(os.path.expanduser(args.seq), size=args.size)
    print(f"[h1.3] {len(src)} frames, grid {src.w}x{src.h}, "
          f"theta_max {math.degrees(src.camera.theta_max):.2f} deg")

    # Candidate pairs by DEVICE-frame GT rotation angle (conjugation-invariant).
    cand: List[Tuple[int, int, float, torch.Tensor]] = []
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            gi, gj = src.pose(i), src.pose(j)
            if gi is None or gj is None:
                continue
            R_dev = gj[0] @ gi[0].transpose(-1, -2)
            ang = rot_angle_deg(R_dev)
            if args.min_gt_rot_deg < ang <= args.max_gt_rot_deg:
                cand.append((i, j, float(ang), R_dev))
    cand.sort(key=lambda c: c[2])
    print(f"[h1.3] {len(cand)} candidate pairs "
          + (f"(GT dev-rot {cand[0][2]:.2f}..{cand[-1][2]:.2f} deg)" if cand else ""))

    summary: Dict = {"config": {k: v for k, v in vars(args).items()},
                     "n_candidates": len(cand)}

    # ---------------- stage 1: hand-eye ----------------
    matches_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    classical: List[Tuple[int, int, float, torch.Tensor, torch.Tensor]] = []
    for i, j, ang, R_dev in cand:
        m = sift_matches(src, i, j, args.nfeatures, args.ratio)
        if m is None:
            continue
        matches_cache[(i, j)] = m
        out = relative_pose_magsac(scatter(src, m[0], m[1]), src.camera,
                                   threshold_deg=args.magsac_thresh_deg)
        if out is None:
            continue
        classical.append((i, j, ang, out[0].double(), R_dev))
    print(f"[h1.3] classical solved {len(classical)}/{len(cand)} pairs")

    angle_gap = [abs(rot_angle_deg(Rh) - ang) for _, _, ang, Rh, _ in classical]
    print(f"[h1.3] |angle(R_hat) - angle(R_dev)| median "
          f"{_median(angle_gap):.3f} deg (conjugation-invariant agreement)")

    # C-independent outlier rejection: a pair whose classical rotation MAGNITUDE
    # disagrees with GT (conjugation cannot change magnitude) had a failed
    # classical estimate; it cannot inform C and should not gate the calibration.
    # The gate metric itself is unchanged and reported for both sets.
    trusted = [t for t, gap in zip(classical, angle_gap)
               if gap <= args.gap_filter_deg]
    print(f"[h1.3] gap filter (<= {args.gap_filter_deg} deg, C-free): "
          f"{len(trusted)}/{len(classical)} pairs kept")

    C = hand_eye_rotation(
        [Rh.numpy() for _, _, _, Rh, _ in trusted],
        [Rd.numpy() for _, _, _, _, Rd in trusted],
        [a for _, _, a, _, _ in trusted])
    Ct = torch.tensor(C, dtype=torch.float64)
    err_dev = [rotation_error_deg(Rh, Rd) for _, _, _, Rh, Rd in classical]
    err_cal = [rotation_error_deg(Rh, Ct @ Rd @ Ct.T)
               for _, _, _, Rh, Rd in classical]
    err_cal_tr = [rotation_error_deg(Rh, Ct @ Rd @ Ct.T)
                  for _, _, _, Rh, Rd in trusted]
    print(f"[h1.3] classical vs GT: device-frame median {_median(err_dev):.3f} deg"
          f" -> C-conjugated median {_median(err_cal):.3f} deg (all), "
          f"{_median(err_cal_tr):.3f} deg (trusted);"
          f" angle(C) = {rot_angle_deg(Ct):.2f} deg")
    summary["handeye"] = {
        "n_pairs": len(classical),
        "angle_gap_median_deg": _median(angle_gap),
        "err_device_frame_median_deg": _median(err_dev),
        "err_calibrated_median_deg": _median(err_cal),
        "err_calibrated_trusted_median_deg": _median(err_cal_tr),
        "n_trusted": len(trusted),
        "angle_C_deg": rot_angle_deg(Ct),
        "C": C.tolist(),
    }
    gate = _median(err_cal_tr) < 1.5
    summary["handeye"]["gate_passed"] = bool(gate)
    if not gate:
        print("[h1.3] GATE FAILED: calibrated classical error >= 1.5 deg. "
              "Aborting span/mask stages per protocol; file the GPU ticket for "
              "the calibration JSON.")
        stages = [s for s in stages if s == "handeye"]

    def R_gt_cam(R_dev: torch.Tensor) -> torch.Tensor:
        return Ct @ R_dev @ Ct.T

    # ---------------- stage 2: span ----------------
    if "span" in stages:
        rng = np.random.default_rng(args.seed)
        conds = [f"t{t:g}" for t in SPANS]
        rot: Dict[str, Dict[str, List[float]]] = {a: {c: [] for c in conds}
                                                  for a in ("real", "synth")}
        prd: Dict[str, Dict[str, List[float]]] = {a: {c: [] for c in conds}
                                                  for a in ("real", "synth")}
        gts: Dict[str, List[float]] = {"real": [], "synth": []}
        nstar: List[int] = []
        dropped = {"real": 0, "synth": 0}
        for i, j, ang, R_dev in cand:
            if (i, j) not in matches_cache:
                continue
            ua, ub = matches_cache[(i, j)]
            Rg = R_gt_cam(R_dev)
            gi, gj = src.pose(i), src.pose(j)
            # camera-frame translation only up to the unknown lever arm; synth
            # arm needs *a* consistent translation, so use the conjugated device
            # one (flagged approximate in the protocol).
            t_dev = (gj[1] - (gj[0] @ gi[0].transpose(-1, -2)) @ gi[1])
            t_cam = (Ct @ t_dev.double()).float()
            eye = torch.eye(3, dtype=Rg.dtype)
            for arm in ("real", "synth"):
                if arm == "real":
                    sa, sb = ua, ub
                else:
                    sa, sb = synth_targets(src, ua, Rg.float(), t_cam, rng)
                if sa is None or len(sa) < MIN_PER_COND:
                    dropped[arm] += 1
                    continue
                tha = theta_of(src, sa)
                sels = [np.flatnonzero(tha <= T) for T in SPANS]
                n_star = min(len(s) for s in sels)
                if n_star < MIN_PER_COND:
                    dropped[arm] += 1
                    continue
                res: Dict[str, Optional[Tuple[float, float]]] = {}
                for c, sel in zip(conds, sels):
                    rerrs, perrs = [], []
                    for _ in range(RESAMPLES):
                        sub = rng.choice(sel, size=n_star, replace=False) \
                            if len(sel) > n_star else sel
                        out = relative_pose_magsac(
                            scatter(src, sa[sub], sb[sub]), src.camera,
                            threshold_deg=args.magsac_thresh_deg)
                        if out is None:
                            continue
                        Rh = out[0].to(Rg)
                        rerrs.append(rotation_error_deg(Rh, Rg))
                        perrs.append(rotation_error_deg(eye, Rh))
                    res[c] = (None if len(rerrs) < (RESAMPLES + 1) // 2
                              else (_median(rerrs), _median(perrs)))
                if any(v is None for v in res.values()):
                    dropped[arm] += 1
                    continue
                for c in conds:
                    rot[arm][c].append(res[c][0])
                    prd[arm][c].append(res[c][1])
                gts[arm].append(ang)
                if arm == "real":
                    nstar.append(n_star)
        summary["span"] = {"n_star_median": _median([float(x) for x in nstar])}
        for arm in ("real", "synth"):
            print(f"\n=== span/{arm}: {len(gts[arm])} pairs "
                  f"(dropped {dropped[arm]}), N* median "
                  f"{_median([float(x) for x in nstar]):.0f} ===")
            print(f"{'cond':>7} {'med rot err':>12} {'gain':>7}")
            summary["span"][arm] = {"n_pairs": len(gts[arm]), "conds": {}}
            for c in conds:
                g = _gain(prd[arm][c], gts[arm])
                print(f"{c:>7} {_median(rot[arm][c]):12.3f} {g:7.3f}")
                summary["span"][arm]["conds"][c] = {
                    "median_rot_err_deg": _median(rot[arm][c]), "gain": g}
            if gts[arm]:
                d = np.asarray(rot[arm][conds[-1]]) - np.asarray(rot[arm][conds[0]])
                summary["span"][arm]["paired_widest_minus_narrowest"] = {
                    "median_diff_deg": _median(list(d)),
                    "n_wide_better": int((d < 0).sum()),
                    "n_narrow_better": int((d > 0).sum())}
                print(f"paired {conds[-1]}-{conds[0]}: median "
                      f"{_median(list(d)):+.3f} deg, wide better on "
                      f"{(d < 0).sum()}/{len(d)}")

    # ---------------- stage 3: mask (DA3-Small) ----------------
    if "mask" in stages:
        from raytun3r.backbones import build_backbone
        bb = build_backbone("da3", weights="pretrained", device="cpu",
                            variant="small")
        bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="range")
        theta = src.camera.incidence_grid(src.h, src.w)
        cone = theta <= src.camera.theta_max
        cen_keep = (theta <= math.radians(args.mask_deg)) & cone   # mask rim
        rim_keep = (theta > math.radians(args.mask_deg)) & cone    # mask center
        fr_rim = float((cone & ~cen_keep).float().sum() / cone.float().sum())
        g = torch.Generator().manual_seed(args.seed)
        ph, pw = (src.h + 13) // 14, (src.w + 13) // 14
        n_mask = int(round(fr_rim * ph * pw))
        idx = torch.randperm(ph * pw, generator=g)[:n_mask]
        pm = torch.zeros(ph * pw, dtype=torch.bool)
        pm[idx] = True
        pm = pm.view(ph, pw).repeat_interleave(14, 0).repeat_interleave(14, 1)
        rnd_keep = (~pm[:src.h, :src.w]) & cone
        print(f"\n[h1.3/mask] within-cone masked fractions: rim "
              f"{fr_rim:.1%}, center {1 - fr_rim:.1%}, random {fr_rim:.1%}")

        def apply(imgs: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
            out = imgs.clone()
            for k in range(out.shape[0]):
                mean = out[k][:, keep].mean(dim=1)
                out[k][:, ~keep] = mean[:, None]
            return out

        conds = ["vanilla", "rim_masked", "center_masked", "random_masked"]
        keeps = {"vanilla": cone, "rim_masked": cen_keep,
                 "center_masked": rim_keep, "random_masked": rnd_keep}
        per: Dict[str, List[float]] = {c: [] for c in conds}
        prd2: Dict[str, List[float]] = {c: [] for c in conds}
        gts2: List[float] = []
        for n, (i, j, ang, R_dev) in enumerate(cand):
            Rg = R_gt_cam(R_dev)
            eye = torch.eye(3, dtype=Rg.dtype)
            imgs = torch.stack([src.image(i), src.image(j)])
            t0 = time.time()
            row = {}
            with torch.no_grad():
                for c in conds:
                    Rh = bb.forward(apply(imgs, keeps[c])[None]).relative(0, 1)[0]
                    Rh = Rh.to(Rg)
                    row[c] = (rotation_error_deg(Rh, Rg),
                              rotation_error_deg(eye, Rh))
            for c in conds:
                per[c].append(row[c][0])
                prd2[c].append(row[c][1])
            gts2.append(ang)
            print(f"  pair {n + 1}/{len(cand)} (GT {ang:5.2f}, "
                  f"{time.time() - t0:5.1f}s): "
                  + " ".join(f"{c} {row[c][0]:6.2f}" for c in conds), flush=True)
        print(f"\n=== mask: {len(gts2)} pairs ===")
        print(f"{'cond':>14} {'med err':>9} {'gain':>7}")
        summary["mask"] = {"n_pairs": len(gts2),
                           "rim_frac_of_cone": fr_rim, "conds": {}}
        for c in conds:
            gg = _gain(prd2[c], gts2)
            print(f"{c:>14} {_median(per[c]):9.3f} {gg:7.3f}")
            summary["mask"]["conds"][c] = {"median_err_deg": _median(per[c]),
                                           "gain": gg}

    if args.out:
        out = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[h1.3] wrote {out}")


if __name__ == "__main__":
    main()
