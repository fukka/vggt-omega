"""Ticket 029 (H4.1): do hand/body pixels disrupt DA3-Small's pose and depth?

Two arms on each of the three skeleton-flagged ADT sequences (raw VRS, opened
via ``AriaDigitalTwinDataPathsProvider(skeleton_flag=True)`` as in ticket 028's
``hand_pixel_stats.py``):

Pose arm  -- for ~40 temporally-adjacent RGB frame pairs, spread evenly across
    the sequence, DA3-Small's relative-rotation error under three inputs:
    vanilla, GT-dynamic pixels mean-filled (from the skeleton segmentation),
    and an area-matched random mean-fill (same masked fraction per frame,
    seeded, so any gap between it and the GT mask isn't just "less pixels").
    GT relative pose comes from ``get_aria_3d_pose_by_timestamp_ns`` conjugated
    by ticket 027's audited ``T_device_camera`` (no hand-eye bootstrap here --
    that JSON is the point of not needing one).

Depth arm -- on the same anchor frames with dyn_frac > 2%, DA3-Small AbsRel on
    STATIC pixels only (scale_shift per frame, range domain -- the protocol of
    record in ``finetune/eval/metrics.align_depth``), vanilla vs GT-dynamic
    mean-filled input: does removing hands from the input change static depth
    around them?

Camera intrinsics/distortion follow the rest of this repo's Aria KB4 model
(``cam3r.cameras.aria_214_1_kb4`` + ``raytun3r.cameras.KannalaBrandt``, as in
``adt_pose_value.py`` and ``feature_head.py``); the calibration JSON supplies
only the device<->camera extrinsic rotation.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h4-dynamics/code/hands_pose_depth.py \
        --seq-dirs /path/to/Apartment_release_meal_skeleton_seq131_M1292 ... \
        --out-dir results/autoresearch-h4-pose
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from cam3r.cameras import _ARIA_KB4, aria_214_1_kb4, aria_valid_theta_max  # noqa: E402
from raytun3r.cameras import KannalaBrandt              # noqa: E402
from raytun3r.metrics import rotation_error_deg          # noqa: E402
from finetune.eval.metrics import align_depth            # noqa: E402
from rim_pose_value import _gain, _median                # noqa: E402

import projectaria_tools.projects.adt as adt             # noqa: E402
from projectaria_tools.core.stream_id import StreamId    # noqa: E402

RGB_STREAM_ID = StreamId("214-1")
DEPTH_SCALE_M = 0.001
DYN_FRAC_DEPTH_GATE = 0.02
SEED = 0


def quat_xyzw_to_matrix(q: List[float]) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class ADTSkeletonSource:
    """Skeleton-flagged ADT sequence: RGB, dynamic-pixel mask, depth, device
    pose, all keyed by the same RGB device-capture timestamp."""

    def __init__(self, seq_dir: str, size: int = 504) -> None:
        self.seq_dir = seq_dir
        self.name = os.path.basename(os.path.normpath(seq_dir))
        paths_provider = adt.AriaDigitalTwinDataPathsProvider(seq_dir)
        self.gt = adt.AriaDigitalTwinDataProvider(paths_provider.get_datapaths(True))
        self.size = size

        human_ids = set()
        for iid in self.gt.get_instance_ids():
            info = self.gt.get_instance_info_by_id(iid)
            if info.instance_type == adt.InstanceType.HUMAN:
                human_ids.add(int(iid))
        self.human_id_arr = np.array(sorted(human_ids), dtype=np.uint64)

        calib = self.gt.get_aria_camera_calibration(RGB_STREAM_ID)
        self.native_w, self.native_h = (int(x) for x in calib.get_image_size())
        self.camera = KannalaBrandt(
            *self._scaled_intrinsics(size), width=size, height=size,
            k=tuple(_ARIA_KB4), theta_max=aria_valid_theta_max())
        self.ts = list(self.gt.get_aria_device_capture_timestamps_ns(RGB_STREAM_ID))

    def _scaled_intrinsics(self, size: int) -> Tuple[float, float, float, float]:
        ref = aria_214_1_kb4(size, size, rotated=False)
        return ref.fx, ref.fy, ref.cx, ref.cy

    def _resize(self, arr: np.ndarray, nearest: bool) -> np.ndarray:
        t = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t[None, None].float()
        else:
            t = t.permute(2, 0, 1)[None].float()
        mode = "nearest" if nearest else "bicubic"
        out = torch.nn.functional.interpolate(
            t, size=(self.size, self.size), mode=mode,
            align_corners=None if nearest else False)
        return out[0, 0].numpy() if arr.ndim == 2 else out[0].permute(1, 2, 0).numpy()

    def frame(self, t: int) -> Optional[Dict]:
        img = self.gt.get_aria_image_by_timestamp_ns(t, RGB_STREAM_ID)
        seg = self.gt.get_segmentation_image_by_timestamp_ns(t, RGB_STREAM_ID)
        depth = self.gt.get_depth_image_by_timestamp_ns(t, RGB_STREAM_ID)
        pose3d = self.gt.get_aria_3d_pose_by_timestamp_ns(t)
        if not (img.is_valid() and seg.is_valid() and depth.is_valid()
                and pose3d.is_valid()):
            return None
        rgb = self._resize(img.data().to_numpy_array().astype(np.float32) / 255.0,
                           nearest=False)
        seg_img = seg.data().to_numpy_array()
        dyn = np.isin(seg_img, self.human_id_arr) if self.human_id_arr.size else \
            np.zeros_like(seg_img, dtype=bool)
        dyn = self._resize(dyn.astype(np.float32), nearest=True) > 0.5
        depth_m = depth.data().to_numpy_array().astype(np.float32) * DEPTH_SCALE_M
        depth_m = self._resize(depth_m, nearest=True)
        Twd = pose3d.data().transform_scene_device.to_matrix()  # world<-device
        R_dw = Twd[:3, :3].T
        t_dw = -R_dw @ Twd[:3, 3]
        return {"rgb": torch.from_numpy(rgb).permute(2, 0, 1).float(),
                "dyn": dyn, "depth_m": depth_m, "R_dw": R_dw, "t_dw": t_dw}


def mean_fill(rgb: torch.Tensor, mask_out: np.ndarray) -> torch.Tensor:
    """Replace pixels where mask_out is True with the mean color of the rest."""
    keep = ~torch.from_numpy(mask_out)
    out = rgb.clone()
    if keep.sum() == 0 or (~keep).sum() == 0:
        return out
    mean = out[:, keep].mean(dim=1)
    out[:, ~keep] = mean[:, None]
    return out


def random_matched_mask(area_mask: np.ndarray, cone: np.ndarray, seed: int) -> np.ndarray:
    """A random mask of the same pixel count as area_mask, drawn from cone."""
    rng = np.random.default_rng(seed)
    n = int(area_mask.sum())
    idx = np.flatnonzero(cone)
    if n == 0 or idx.size == 0:
        return np.zeros_like(area_mask)
    sel = rng.choice(idx, size=min(n, idx.size), replace=False)
    out = np.zeros(area_mask.size, dtype=bool)
    out[sel] = True
    return out.reshape(area_mask.shape)


def pick_anchors(ts: List[int], n: int) -> List[int]:
    if len(ts) <= n:
        return list(range(len(ts)))
    idx = np.linspace(0, len(ts) - 1, n)
    return sorted(set(int(round(i)) for i in idx))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dirs", nargs="+", required=True)
    ap.add_argument("--calib", default=str(Path(__file__).resolve().parents[4]
                                           / "cam3r" / "data"
                                           / "adt_camera_rgb_calibration.json"))
    ap.add_argument("--n-anchors", type=int, default=41)
    ap.add_argument("--size", type=int, default=504)
    ap.add_argument("--depth-max-m", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)

    calib = json.load(open(args.calib))
    R_dc = quat_xyzw_to_matrix(calib["T_device_camera"]["quaternion_xyzw"])
    Ct = torch.tensor(R_dc.T, dtype=torch.float64)  # camera<-device

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu", variant="small")

    os.makedirs(args.out_dir, exist_ok=True)
    for seq_dir in args.seq_dirs:
        src = ADTSkeletonSource(seq_dir, size=args.size)
        bb.install(None, src.camera, (src.camera.height, src.camera.width),
                  patch_undistort=False, border_token=False, dpt_grid=False,
                  depth_convention="range")
        cone = src.camera.incidence_grid(src.camera.height,
                                         src.camera.width).numpy() <= src.camera.theta_max
        anchors = pick_anchors(src.ts, args.n_anchors)
        print(f"[{src.name}] {len(src.ts)} frames, {len(anchors)} anchors")

        frames: Dict[int, Dict] = {}
        for a in anchors:
            fr = src.frame(src.ts[a])
            if fr is not None:
                frames[a] = fr

        # ---- pose arm: consecutive pairs among the anchor set ----
        conds = ["vanilla", "gt_dyn_masked", "random_matched"]
        rot: Dict[str, List[float]] = {c: [] for c in conds}
        prd: Dict[str, List[float]] = {c: [] for c in conds}
        gts: List[float] = []
        dyn_frac_pair: List[float] = []
        pair_ids = [(anchors[k], anchors[k + 1]) for k in range(len(anchors) - 1)
                   if anchors[k] in frames and anchors[k + 1] in frames]
        for n, (i, j) in enumerate(pair_ids):
            fi, fj = frames[i], frames[j]
            R_dev = torch.from_numpy(fj["R_dw"] @ fi["R_dw"].T).double()
            Rg = (Ct @ R_dev @ Ct.T).float()
            eye = torch.eye(3, dtype=Rg.dtype)
            dyn_i, dyn_j = fi["dyn"] & cone, fj["dyn"] & cone
            df = 0.5 * (dyn_i.sum() + dyn_j.sum()) / max(cone.sum(), 1)
            dyn_frac_pair.append(float(df))
            rnd_i = random_matched_mask(dyn_i, cone, args.seed * 100000 + n * 2)
            rnd_j = random_matched_mask(dyn_j, cone, args.seed * 100000 + n * 2 + 1)
            imgs_by_cond = {
                "vanilla": torch.stack([fi["rgb"], fj["rgb"]]),
                "gt_dyn_masked": torch.stack([mean_fill(fi["rgb"], dyn_i),
                                              mean_fill(fj["rgb"], dyn_j)]),
                "random_matched": torch.stack([mean_fill(fi["rgb"], rnd_i),
                                               mean_fill(fj["rgb"], rnd_j)]),
            }
            t0 = time.time()
            with torch.no_grad():
                for c in conds:
                    Rh = bb.forward(imgs_by_cond[c][None]).relative(0, 1)[0].to(Rg)
                    rot[c].append(rotation_error_deg(Rh, Rg))
                    prd[c].append(rotation_error_deg(eye, Rh))
            gts.append(rotation_error_deg(eye, Rg))
            print(f"  pair {n + 1}/{len(pair_ids)} dyn_frac~{df:.4f} "
                  f"({time.time() - t0:4.1f}s): "
                  + " ".join(f"{c} {rot[c][-1]:6.2f}" for c in conds), flush=True)

        pose_summary = {"n_pairs": len(pair_ids), "conds": {}}
        print(f"\n=== {src.name} pose arm: {len(pair_ids)} pairs ===")
        for c in conds:
            g = _gain(prd[c], gts)
            print(f"{c:>16} median_err {_median(rot[c]):7.3f}  gain {g:6.3f}")
            pose_summary["conds"][c] = {"median_rot_err_deg": _median(rot[c]),
                                        "gain": g}
        if pair_ids:
            d_rand = np.asarray(rot["random_matched"]) - np.asarray(rot["gt_dyn_masked"])
            hi = [v for v, df in zip(d_rand, dyn_frac_pair) if df > DYN_FRAC_DEPTH_GATE]
            pose_summary["random_minus_gtmask_all"] = {
                "median_deg": _median(list(d_rand)),
                "n_gtmask_better": int((d_rand > 0).sum()), "n_pairs": len(d_rand)}
            pose_summary["random_minus_gtmask_high_dynfrac"] = {
                "median_deg": _median(hi) if hi else None,
                "n_gtmask_better": int(sum(1 for v in hi if v > 0)), "n_pairs": len(hi)}
            print(f"random - gt_dyn_masked, all pairs: median "
                  f"{_median(list(d_rand)):+.3f} deg, gt-mask better on "
                  f"{(d_rand > 0).sum()}/{len(d_rand)}")
            print(f"random - gt_dyn_masked, dyn_frac>{DYN_FRAC_DEPTH_GATE}: "
                  f"median {(_median(hi) if hi else float('nan')):+.3f} deg "
                  f"({len(hi)} pairs)")

        # ---- depth arm: single frames with dyn_frac > 2% ----
        depth_before, depth_after = [], []
        n_depth_frames = 0
        for a in anchors:
            fr = frames.get(a)
            if fr is None:
                continue
            dyn = fr["dyn"] & cone
            df = float(dyn.sum()) / max(cone.sum(), 1)
            if df <= DYN_FRAC_DEPTH_GATE:
                continue
            n_depth_frames += 1
            static = cone & ~dyn & (fr["depth_m"] > 0)
            gz = fr["depth_m"]
            with torch.no_grad():
                pr_v = bb.forward(fr["rgb"][None, None])
                pr_v.require_convention("range")
                d_v = pr_v.depth[0].numpy()
                masked_rgb = mean_fill(fr["rgb"], dyn)
                pr_m = bb.forward(masked_rgb[None, None])
                pr_m.require_convention("range")
                d_m = pr_m.depth[0].numpy()
            cos_t = torch.cos(src.camera.incidence_grid(
                src.camera.height, src.camera.width)).numpy()
            gr = gz / np.clip(cos_t, 1e-6, None)
            valid = static & (gr <= args.depth_max_m) & (d_v > 1e-6) & (d_m > 1e-6)
            if valid.sum() < 30:
                n_depth_frames -= 1
                continue
            av = align_depth(d_v, gr, valid, mode="scale_shift")
            am = align_depth(d_m, gr, valid, mode="scale_shift")
            depth_before.append(float(np.median(np.abs(av - gr)[valid] / gr[valid])))
            depth_after.append(float(np.median(np.abs(am - gr)[valid] / gr[valid])))

        depth_summary = {
            "n_frames_dyn_frac_gt_2pct": n_depth_frames,
            "static_absrel_vanilla_median": _median(depth_before),
            "static_absrel_dyn_masked_median": _median(depth_after),
        }
        print(f"\n=== {src.name} depth arm: {n_depth_frames} frames, dyn_frac>"
              f"{DYN_FRAC_DEPTH_GATE} ===")
        print(f"static AbsRel vanilla {_median(depth_before):.4f} -> "
              f"dyn-masked {_median(depth_after):.4f}")

        out_path = os.path.join(args.out_dir, f"{src.name}.json")
        with open(out_path, "w") as f:
            json.dump({"seq": src.name, "pose": pose_summary,
                      "depth": depth_summary, "config": vars(args)}, f, indent=2)
        print(f"[{src.name}] wrote {out_path}")


if __name__ == "__main__":
    main()
