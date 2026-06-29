# Copyright (c) 2026.
"""Verify the fisheye→ERP projection is geometrically correct (esp. for ADT/Aria).

Two checks:

1. ``--synthetic`` (default, no data needed): paint a lat/lon **graticule** onto a
   synthetic Aria fisheye using an *independent* forward projection, warp it with
   ``fisheye_to_erp_fwd``, and verify it comes back as a **straight rectangular
   grid**. This is the gold-standard ERP test: parallels (constant latitude) must
   become straight horizontal lines and meridians (constant longitude) straight
   vertical lines — the one invariant a correct equirectangular projection
   guarantees. Prints a quantitative PASS/FAIL (perpendicular deviation in px).

2. ``--adt-root <path>`` (run on the box where ADT lives): pull one real ADT frame
   through the same loader ``run_baselines`` uses, warp it, and save the input +
   ERP with the analytic graticule overlaid, so the unwrap can be eyeballed on real
   data. ``--frame-image <png>`` does the same for any saved fisheye image.

Why a graticule and not "are the room's lines straight?": in ERP, an arbitrary
straight world line (e.g. a table edge or a soccer goal bar) is **supposed to
curve** — only parallels, meridians and rays through the optical axis stay
straight. So a curving goal bar is correct ERP, not a projection error; the
graticule is what must be straight.

Examples
--------
    python -m finetune.eval.baselines.verify_erp_projection                # synthetic proof
    python -m finetune.eval.baselines.verify_erp_projection \
        --adt-root /group-volume/Fengjia/data/projectaria_tools_adt_data_clean --frame 0
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np

from .aria_fisheye import AriaFisheye, aria_intrinsics, kb4_max_incidence
from .erp_utils import crop_size, fisheye_to_erp_fwd


# --------------------------------------------------------------------------- #
# forward model: (lat, lon) -> fisheye pixel, independent of cam_to_erp_patch_fast
# (DAC convention, patch centred on the optical axis: theta=phi=0).
# --------------------------------------------------------------------------- #
def latlon_to_fisheye(cam: AriaFisheye, lat: np.ndarray, lon: np.ndarray):
    k1, k2, k3, k4 = cam.k
    cos_c = np.cos(lat) * np.cos(lon)
    x_num = np.cos(lat) * np.sin(lon)
    y_num = np.sin(lat)
    th = np.arccos(np.clip(cos_c, -1.0, 1.0))
    thd = th * (1 + th**2 * (k1 + th**2 * (k2 + th**2 * (k3 + th**2 * k4))))
    r = np.hypot(x_num, y_num) + 1e-12
    u = cam.fx * (thd * x_num / r) + cam.cx
    v = cam.fy * (thd * y_num / r) + cam.cy
    return u, v, th


def _draw(img, u, v, th, th_max, color, rad=1):
    import cv2

    ok = th <= th_max * 0.99
    for uu, vv in zip(u[ok], v[ok]):
        if 0 <= uu < img.shape[1] and 0 <= vv < img.shape[0]:
            cv2.circle(img, (int(uu), int(vv)), rad, color, -1)


def synthetic_graticule_test(cam: AriaFisheye, cano: int, fwd_sz: Tuple[int, int],
                             crop_wfov: float, out_dir: str) -> bool:
    """Paint an in-cone lat/lon graticule, warp it, and check it is a straight grid."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    H, W = cam.H, cam.W
    th_max = kb4_max_incidence(cam.k)
    deg = np.deg2rad

    # full graticule (visual)
    img = np.zeros((H, W, 3), np.uint8)
    for lat in range(-55, 56, 11):
        lon = np.linspace(deg(-88), deg(88), 4000)
        _draw(img, *latlon_to_fisheye(cam, deg(lat) * np.ones_like(lon), lon), th_max, (0, 255, 255))
    for lon in range(-77, 78, 11):
        lat = np.linspace(deg(-60), deg(60), 4000)
        _draw(img, *latlon_to_fisheye(cam, lat, deg(lon) * np.ones_like(lat)), th_max, (0, 255, 255))
    cv2.imwrite(os.path.join(out_dir, "graticule_fisheye.png"), img)
    erp = fisheye_to_erp_fwd(img.astype(np.float32) / 255., np.zeros((H, W), np.float32),
                             np.zeros((H, W), np.float32), cam.opencv_fisheye_params(),
                             cano, fwd_sz, crop_wfov)["image_u8"]
    cv2.imwrite(os.path.join(out_dir, "graticule_erp.png"), cv2.cvtColor(erp, cv2.COLOR_RGB2BGR))

    # quantitative: equator (lat=0) must be one horizontal row; meridian (lon=0) one vertical col
    def warp_line(lat, lon, ch):
        im = np.zeros((H, W, 3), np.uint8)
        col = [0, 0, 0]; col[ch] = 255
        _draw(im, *latlon_to_fisheye(cam, lat, lon), th_max, tuple(col))
        w = fisheye_to_erp_fwd(im.astype(np.float32) / 255., np.zeros((H, W), np.float32),
                               np.zeros((H, W), np.float32), cam.opencv_fisheye_params(),
                               cano, fwd_sz, crop_wfov)["image_u8"]
        rows, cols = np.where(w[:, :, ch] > 80)
        return rows, cols

    s = np.linspace(deg(-60), deg(60), 6000)
    er, ec = warp_line(np.zeros_like(s), s, 1)          # equator
    mr, mc = warp_line(s, np.zeros_like(s), 2)          # meridian
    eq_dev = float(er.std()) if er.size else float("nan")   # perpendicular spread (rows)
    mer_dev = float(mc.std()) if mc.size else float("nan")  # perpendicular spread (cols)
    fh, fw = fwd_sz
    ok = eq_dev < 0.02 * fh and mer_dev < 0.02 * fw
    print(f"[verify] turnover (lens FOV half-angle) = {np.rad2deg(th_max):.1f} deg")
    print(f"[verify] equator  (lat=0): row={er.mean():.1f}±{eq_dev:.2f}px, cols {ec.min()}–{ec.max()}  "
          f"(expect a HORIZONTAL line at row {fh//2})")
    print(f"[verify] meridian (lon=0): col={mc.mean():.1f}±{mer_dev:.2f}px, rows {mr.min()}–{mr.max()}  "
          f"(expect a VERTICAL line at col {fw//2})")
    print(f"[verify] straightness: equator ⟂dev={eq_dev:.2f}px, meridian ⟂dev={mer_dev:.2f}px "
          f"-> {'PASS' if ok else 'FAIL'} (graticule is a straight rectangular grid)")
    print(f"[verify] saved {out_dir}/graticule_{{fisheye,erp}}.png")
    return ok


def graticule_overlay(cam: AriaFisheye, cano: int, fwd_sz: Tuple[int, int],
                      crop_wfov: float) -> np.ndarray:
    """RGB overlay of where the analytic lat/lon grid lands in the ERP output (straight)."""
    fh, fw = fwd_sz
    crop_h, crop_w = crop_size(cano, fwd_sz, crop_wfov)
    hF = crop_h / cano * np.pi
    wF = crop_w / (cano * 2) * 2 * np.pi
    ov = np.zeros((fh, fw, 3), np.uint8)
    # lat in [-hF/2,hF/2] -> rows; lon in [-wF/2,wF/2] -> cols (linear)
    for lat in np.deg2rad(range(-55, 56, 11)):
        row = int((lat + hF / 2) / hF * fh)
        if 0 <= row < fh:
            ov[row, :, 1] = 255
    for lon in np.deg2rad(range(-77, 78, 11)):
        col = int((lon + wF / 2) / wF * fw)
        if 0 <= col < fw:
            ov[:, col, 1] = 255
    return ov


def warp_real_frame(rgb_u8: np.ndarray, cam: AriaFisheye, cano: int, fwd_sz: Tuple[int, int],
                    crop_wfov: float, out_dir: str, tag: str) -> None:
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    H, W = rgb_u8.shape[:2]
    z = np.zeros((H, W), np.float32)
    erp = fisheye_to_erp_fwd(rgb_u8.astype(np.float32) / 255., z, z,
                             cam.opencv_fisheye_params(), cano, fwd_sz, crop_wfov)["image_u8"]
    ov = graticule_overlay(cam, cano, fwd_sz, crop_wfov)
    blend = np.where(ov.sum(2, keepdims=True) > 0, (0.5 * erp + 0.5 * ov).astype(np.uint8), erp)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_input.png"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f"{tag}_erp.png"), cv2.cvtColor(erp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f"{tag}_erp_graticule.png"), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print(f"[verify] real frame warped -> {out_dir}/{tag}_{{input,erp,erp_graticule}}.png  "
          f"(grid lines are straight = ERP axes correct; scene content curves = normal ERP)")


def _load_adt_frame(adt_root: str, rgb_subdir: str, frame: int, res: int):
    from ..adt_depth import ADTWindowDataset
    from ..run_eval import _find_seq_dirs

    seq_dirs = _find_seq_dirs(adt_root, rgb_subdir, "depth_npy")
    if not seq_dirs:
        raise SystemExit(f"[verify] no ADT seqs under {adt_root!r}")
    ds = ADTWindowDataset(seq_dirs, seq_len=1, window_stride=1, image_resolution=res,
                          depth_scale=0.001, depth_max_m=10.0, max_frames=frame + 1,
                          rgb_subdir=rgb_subdir, depth_subdir="depth_npy", rectify=False)
    item = ds[min(frame, len(ds) - 1)]
    img = item["images"][0]
    return (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify fisheye->ERP projection correctness")
    p.add_argument("--res", type=int, default=512, help="frame size (ADT uses 512)")
    p.add_argument("--rotated", action="store_true", default=True,
                   help="Aria 270deg-CCW principal-point correction (ADT default)")
    p.add_argument("--no-rotated", dest="rotated", action="store_false")
    p.add_argument("--cano", type=int, default=1400)
    p.add_argument("--fwd-h", type=int, default=500)
    p.add_argument("--fwd-w", type=int, default=750)
    p.add_argument("--crop-wfov", type=float, default=180.0)
    p.add_argument("--out", default="eval_out/verify_erp")
    p.add_argument("--adt-root", default=None, help="run the real-frame check on ADT (GPU box)")
    p.add_argument("--adt-rgb-subdir", default="videos_synthetic")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--frame-image", default=None, help="warp an arbitrary saved fisheye PNG instead")
    a = p.parse_args()

    cam = aria_intrinsics(a.res, a.res, rotated=a.rotated)
    fwd = (a.fwd_h, a.fwd_w)
    print(f"[verify] camera = Aria-214-1 KB4 @ {a.res} (rotated={a.rotated}); "
          f"ERP cano={a.cano} fwd={fwd} crop_wFoV={a.crop_wfov}")
    ok = synthetic_graticule_test(cam, a.cano, fwd, a.crop_wfov, a.out)

    if a.frame_image:
        import cv2
        rgb = cv2.cvtColor(cv2.imread(a.frame_image), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (a.res, a.res))
        warp_real_frame(rgb, cam, a.cano, fwd, a.crop_wfov, a.out, "image")
    if a.adt_root:
        rgb = _load_adt_frame(a.adt_root, a.adt_rgb_subdir, a.frame, a.res)
        warp_real_frame(rgb, cam, a.cano, fwd, a.crop_wfov, a.out, f"adt_f{a.frame}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
