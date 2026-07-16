# Copyright (c) 2026.
"""Correctness checks for the fisheye <-> perspective geometry.

Every coordinate convention bug in a warp pipeline (u/v swap, y flip, wrong
rotation sign, forward/inverse mismatch) is invisible in code review and fatal
at run time.  This script proves the geometry right *before* any GPU run, with
three synthetic tests that need no data and one visual test on a real ADT
frame.  numpy/cv2/PIL only — runs on a laptop without torch.

Tests
-----
A. **Ray round-trip** (numeric): for a grid of pixels in every view, compute
   the view ray, project it into the fisheye with the KB4 *forward* model
   (the path used by ``fisheye_to_persp``), then unproject that fisheye pixel
   back to a ray with the KB4 *inverse* (the path used by fusion's ray LUT).
   The two rays must agree to < 0.05 deg — this certifies that the render
   direction and the fusion direction are exact inverses.

B. **Synthetic sphere pattern** (numeric + visual): a procedural pattern
   defined on ray directions (incidence rings + azimuth wedges + 3 colored
   landmark dots at known azimuths) is rendered analytically to a fisheye
   image; each perspective view is then produced two ways — (1) by
   ``fisheye_to_persp`` on that fisheye image (code under test) and (2) by
   evaluating the pattern directly on the view rays (ground truth).  Any
   flip/rotation/scale error shows up as a gross mismatch; correct code
   differs only by interpolation at pattern edges.

C. **Fusion round-trip** (numeric): a smooth synthetic "range field" f(ray)
   is evaluated on every view's rays and fused back to the fisheye grid with
   uniform weights through ``fuse_views_to_fisheye`` (code under test).  The
   result must match f evaluated on the fisheye rays to <1% — this exercises
   the *entire* fusion path (rotation, gnomonic projection, remap, weighting)
   end to end.  Also asserts the 1+8 view layout covers the imaged cone with
   no holes (coverage >= 1 everywhere inside theta_max - 1 deg).

D. **Real ADT frame** (visual; needs --adt-root): view montage, view-footprint
   overlay on the fisheye frame (+ theta_max circle), coverage heatmap, and an
   RGB re-fusion — the 9 views fused back to the fisheye grid should
   reproduce the original photo (PSNR reported; interpolation-only losses).

Usage
-----
    python VGGT-360-fisheye/checks/check_fisheye2persp.py \
        [--adt-root /path/to/adt] [--rgb-subdir videos_rgb] \
        [--frame 6] [--out outputs/fisheye_checks]

Exit code 0 iff all numeric tests pass.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from utils.fisheye_cam import (aria_intrinsics, fisheye_ray_lut,
                               kb4_forward_theta, kb4_unproject_theta)
from utils.fisheye_fusion import fuse_views_to_fisheye
from utils.fisheye_views import (base_views, fisheye_to_persp, view_rotation)

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def view_ray_grid(fov_deg: float, H: int, W: int, R: np.ndarray) -> np.ndarray:
    """Unit rays of a view's pixel grid, rotated into the camera frame.

    Must replicate ``fisheye_to_persp`` exactly (tangent grid, y down,
    v_cam = v_view @ R^T) — it is the reference for tests A/B/C.
    """
    t = math.tan(math.radians(fov_deg) / 2.0)
    xs = np.linspace(-t, t, W)
    ys = np.linspace(-t, t, H)
    xv, yv = np.meshgrid(xs, ys)
    vec = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec @ R.T


def kb4_project(rays: np.ndarray, cam) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """KB4 forward projection of unit rays -> (u, v, theta)."""
    z = np.clip(rays[..., 2], -1.0, 1.0)
    theta = np.arccos(z)
    theta_d = kb4_forward_theta(theta, cam.k)
    rxy = np.sqrt(rays[..., 0] ** 2 + rays[..., 1] ** 2)
    inv = np.where(rxy > 1e-12, 1.0 / rxy, 0.0)
    u = cam.cx + cam.fx * theta_d * rays[..., 0] * inv
    v = cam.cy + cam.fy * theta_d * rays[..., 1] * inv
    return u, v, theta


def kb4_unproject(u: np.ndarray, v: np.ndarray, cam) -> np.ndarray:
    """KB4 inverse projection of pixel coords -> unit rays (fusion-LUT math)."""
    x = (u - cam.cx) / cam.fx
    y = (v - cam.cy) / cam.fy
    theta_d = np.sqrt(x * x + y * y)
    theta = kb4_unproject_theta(theta_d, cam.k)
    sin_t = np.sin(theta)
    inv = np.where(theta_d > 1e-9, 1.0 / theta_d, 0.0)
    rays = np.stack([sin_t * x * inv, sin_t * y * inv, np.cos(theta)], axis=-1)
    rays[theta_d <= 1e-9] = (0.0, 0.0, 1.0)
    return rays


def sphere_pattern(rays: np.ndarray) -> np.ndarray:
    """Procedural RGB pattern on ray directions (uint8).

    * brightness rings every 10 deg of incidence (checks radial scale),
    * hue wedges every 30 deg of azimuth (checks rotation direction),
    * landmark dots: RED toward +x (image right, azimuth 0), GREEN toward +y
      (image bottom, azimuth 90), WHITE on the optical axis — any y-flip or
      u/v swap moves them to obviously wrong places.
    """
    theta = np.degrees(np.arccos(np.clip(rays[..., 2], -1.0, 1.0)))
    phi = np.degrees(np.arctan2(rays[..., 1], rays[..., 0])) % 360.0

    ring = ((theta // 10).astype(int) % 2).astype(np.float32)
    wedge = ((phi // 30).astype(int) % 3).astype(np.float32) / 2.0
    img = np.stack([0.25 + 0.5 * wedge,
                    0.25 + 0.5 * ring,
                    0.75 - 0.5 * wedge], axis=-1)

    def dot(center_dir, color, radius_deg=4.0):
        ang = np.degrees(np.arccos(np.clip(rays @ np.asarray(center_dir), -1, 1)))
        m = ang < radius_deg
        img[m] = color

    s45 = math.sin(math.radians(45.0))
    dot([s45, 0.0, math.cos(math.radians(45.0))], (1.0, 0.0, 0.0))   # right
    dot([0.0, s45, math.cos(math.radians(45.0))], (0.0, 1.0, 0.0))   # bottom
    dot([0.0, 0.0, 1.0], (1.0, 1.0, 1.0))                            # center
    return (img * 255).astype(np.uint8)


def montage(images, cols=3, pad=4, labels=None) -> np.ndarray:
    """Tile equally-sized HxWx3 uint8 images into a grid."""
    H, W = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), 30, np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        img = img.copy()
        if labels:
            cv2.putText(img, labels[i], (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 60), 2, cv2.LINE_AA)
        out[r * (H + pad): r * (H + pad) + H,
            c * (W + pad): c * (W + pad) + W] = img
    return out


def colorize(x: np.ndarray, vmax: float) -> np.ndarray:
    x8 = np.clip(x / max(vmax, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(x8, cv2.COLORMAP_VIRIDIS),
                        cv2.COLOR_BGR2RGB)


def save(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(img).save(path)
    print(f"    saved {path}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_a_ray_roundtrip(cam, views) -> bool:
    """Forward (render) and inverse (fusion) projections must be inverses."""
    print("\n[A] ray round-trip: view ray -> KB4 forward -> KB4 inverse")
    theta_max = cam.theta_max()
    worst = 0.0
    for (psi, tilt, fov) in views:
        R = view_rotation(psi, tilt)
        rays = view_ray_grid(fov, 64, 64, R)
        u, v, theta = kb4_project(rays, cam)
        ok = theta <= theta_max - 1e-3
        back = kb4_unproject(u[ok], v[ok], cam)
        dot = np.clip(np.sum(back * rays[ok], axis=-1), -1.0, 1.0)
        err = np.degrees(np.arccos(dot))
        worst = max(worst, float(err.max()))
    good = worst < 0.05
    print(f"    max angular error over all views: {worst:.5f} deg  "
          f"{PASS if good else FAIL} (< 0.05)")
    return good


def test_b_synthetic_views(cam, views, out_dir) -> bool:
    """fisheye_to_persp(analytic fisheye) must equal the direct render."""
    print("\n[B] synthetic sphere pattern: warp vs direct render")
    rays_fe, cone = fisheye_ray_lut(cam)
    fe_img = sphere_pattern(rays_fe.astype(np.float64))
    fe_img[~cone] = 0
    save(os.path.join(out_dir, "B_synthetic_fisheye.png"), fe_img)

    warped_all, direct_all, diff_all = [], [], []
    worst_frac = 0.0
    for (psi, tilt, fov) in views:
        warped, valid = fisheye_to_persp(fe_img, cam, psi, tilt, fov, 256, 256)
        R = view_rotation(psi, tilt)
        direct = sphere_pattern(view_ray_grid(fov, 256, 256, R))
        direct = direct * valid[..., None].astype(np.uint8)
        # interior only: pattern edges alias under bilinear resampling
        interior = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0.5
        d = np.abs(warped.astype(np.int16) - direct.astype(np.int16)).max(axis=-1)
        frac_bad = float((d[interior] > 40).mean()) if interior.any() else 1.0
        worst_frac = max(worst_frac, frac_bad)
        warped_all.append(warped.astype(np.uint8))
        direct_all.append(direct.astype(np.uint8))
        diff_all.append(colorize(d.astype(np.float32), 60.0))

    labels = [f"az{int(p)} tilt{int(t)}" for (p, t, _) in views]
    save(os.path.join(out_dir, "B_views_warped.png"),
         montage(warped_all, labels=labels))
    save(os.path.join(out_dir, "B_views_direct.png"),
         montage(direct_all, labels=labels))
    save(os.path.join(out_dir, "B_views_absdiff.png"), montage(diff_all))
    good = worst_frac < 0.03
    print(f"    worst view: {worst_frac * 100:.2f}% interior pixels differ "
          f"by > 40/255  {PASS if good else FAIL} (< 3%)")
    return good


def test_c_fusion_roundtrip(cam, views, out_dir) -> bool:
    """fuse(f on view rays) must reproduce f on the fisheye rays; no holes.

    Two coverage assertions with different guarantees:
      * pure-frustum coverage (geometry only) must have NO holes inside the
        cone — this certifies the center+ring layout itself;
      * valid-mask coverage (the real fusion path, with per-view valid masks
        eroded by ``erode_valid_px``) must have no holes away from the rim:
        the erosion legitimately retires a thin band at the cone edge /
        frame border, so those pixels are excluded (they are also excluded
        from evaluation by the ``coverage > 0`` mask in main_adt.py).
    """
    print("\n[C] fusion round-trip + cone coverage")

    def f(rays):  # smooth synthetic range field, ~[0.9, 2.1] m
        return (1.5 + 0.3 * np.sin(3.0 * rays[..., 0])
                + 0.2 * np.cos(2.0 * rays[..., 1])
                + 0.1 * np.sin(4.0 * rays[..., 2] + 1.0)).astype(np.float32)

    values, valids = [], []
    for (psi, tilt, fov) in views:
        R = view_rotation(psi, tilt)
        rays_v = view_ray_grid(fov, 518, 518, R)
        values.append(f(rays_v))
        # analytic per-view validity, exactly as the pipeline provides it
        _, valid = fisheye_to_persp(np.zeros((cam.H, cam.W), np.float32),
                                    cam, psi, tilt, fov, 518, 518)
        valids.append(valid)

    erode_px = 3
    fused, coverage = fuse_views_to_fisheye(values, views, cam,
                                            view_valids=valids,
                                            erode_valid_px=erode_px)
    rays_fe, cone = fisheye_ray_lut(cam)
    target = f(rays_fe.astype(np.float64))

    m = (coverage > 0) & cone
    rel = np.abs(fused[m] - target[m]) / target[m]
    rel_err = float(rel.mean())

    # 1) layout correctness: pure-frustum coverage, no masks/erosion involved
    theta = np.degrees(np.arccos(np.clip(rays_fe[..., 2], -1, 1)))
    inner = theta <= math.degrees(cam.theta_max()) - 1.0
    frustum_cov = np.zeros((cam.H, cam.W), np.int32)
    for (psi, tilt, fov) in views:
        d_v = rays_fe.astype(np.float64) @ view_rotation(psi, tilt)
        t = math.tan(math.radians(fov) / 2.0)
        with np.errstate(all="ignore"):
            xc, yc = d_v[..., 0] / d_v[..., 2], d_v[..., 1] / d_v[..., 2]
        frustum_cov += ((d_v[..., 2] > 1e-6) & (np.abs(xc) <= t)
                        & (np.abs(yc) <= t) & cone).astype(np.int32)
    layout_holes = int(np.count_nonzero(inner & (frustum_cov == 0)))

    # 2) fusion-path coverage: with the rim-rescue tier (un-eroded fallback,
    #    see fuse_views_to_fisheye) the WHOLE cone must be covered, up to one
    #    view pixel (~fov/518 deg) of nearest-neighbour quantisation at the
    #    very rim and 1 px at the frame border.
    rim_deg = 2.0 * views[0][2] / 518.0
    core = theta <= math.degrees(cam.theta_max()) - rim_deg
    core[:1, :] = core[-1:, :] = False
    core[:, :1] = core[:, -1:] = False
    holes = int(np.count_nonzero(core & (coverage == 0)))

    save(os.path.join(out_dir, "C_fused_field.png"), colorize(fused * cone, 2.2))
    save(os.path.join(out_dir, "C_target_field.png"), colorize(target * cone, 2.2))
    save(os.path.join(out_dir, "C_coverage.png"),
         colorize(coverage.astype(np.float32), float(coverage.max())))

    good = rel_err < 0.01 and layout_holes == 0 and holes == 0
    print(f"    mean |rel err| = {rel_err * 100:.3f}%  (< 1%)   "
          f"layout holes = {layout_holes}  fusion holes (core) = {holes}  "
          f"(both == 0)   {PASS if good else FAIL}")
    print(f"    coverage: min in core = {int(coverage[core].min())}, "
          f"overlap (>=2 views) = {float((coverage[core] >= 2).mean()) * 100:.1f}%")
    return good


def test_d_real_frame(cam_native, views, adt_root, rgb_subdir, frame_idx,
                      out_dir) -> None:
    """Visual sanity on a real ADT frame (no pass/fail — for human eyes)."""
    print("\n[D] real ADT frame visuals")
    from datasets.adt import ADTFisheyeFrames, find_adt_sequences
    seqs = find_adt_sequences(adt_root, rgb_subdir=rgb_subdir)
    if not seqs:  # fall back: sequences with RGB but maybe no depth pairing
        raise SystemExit(f"no sequences with {rgb_subdir}/ + depth_npy/ under {adt_root}")
    ds = ADTFisheyeFrames(seqs[:1], rgb_subdir=rgb_subdir,
                          max_frames=frame_idx + 1)
    item = ds[min(frame_idx, len(ds) - 1)]
    rgb = item["rgb"]
    H, W = rgb.shape[:2]
    from utils.fisheye_cam import aria_intrinsics as _ai
    cam = _ai(H, W, rotated=True)

    # 1) view montage
    persps, valids, labels = [], [], []
    for (psi, tilt, fov) in views:
        p, v = fisheye_to_persp(rgb, cam, psi, tilt, fov, 512, 512)
        persps.append(p.astype(np.uint8))
        valids.append(v)
        labels.append(f"az{int(psi)} tilt{int(tilt)}")
    save(os.path.join(out_dir, "D_view_montage.png"),
         montage(persps, labels=labels))

    # 2) footprint overlay: project each view's border into the fisheye frame
    overlay = rgb.copy()
    palette = [(255, 60, 60), (255, 160, 40), (250, 250, 60), (60, 255, 60),
               (60, 250, 250), (80, 120, 255), (200, 80, 255), (255, 120, 200),
               (255, 255, 255)]
    for i, (psi, tilt, fov) in enumerate(views):
        R = view_rotation(psi, tilt)
        t = math.tan(math.radians(fov) / 2.0)
        n = 160
        xs = np.linspace(-t, t, n)
        border = np.concatenate([
            np.stack([xs, np.full(n, -t)], axis=1),
            np.stack([np.full(n, t), xs], axis=1),
            np.stack([xs[::-1], np.full(n, t)], axis=1),
            np.stack([np.full(n, -t), xs[::-1]], axis=1)])
        rays = np.concatenate([border, np.ones((border.shape[0], 1))], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        rays = rays @ R.T
        u, v, theta = kb4_project(rays, cam)
        ok = theta <= cam.theta_max() - 1e-3
        pts = np.stack([u[ok], v[ok]], axis=1).astype(np.int32)
        for j in range(len(pts) - 1):
            if np.all(np.abs(pts[j + 1] - pts[j]) < 60):  # skip cone gaps
                cv2.line(overlay, tuple(pts[j]), tuple(pts[j + 1]),
                         palette[i % len(palette)], 3)
    # theta_max circle
    phis = np.linspace(0, 2 * np.pi, 720)
    ring = np.stack([np.sin(cam.theta_max() - 1e-3) * np.cos(phis),
                     np.sin(cam.theta_max() - 1e-3) * np.sin(phis),
                     np.full_like(phis, np.cos(cam.theta_max() - 1e-3))], axis=1)
    u, v, _ = kb4_project(ring, cam)
    for j in range(len(u) - 1):
        cv2.line(overlay, (int(u[j]), int(v[j])), (int(u[j + 1]), int(v[j + 1])),
                 (0, 0, 0), 2)
    save(os.path.join(out_dir, "D_footprints.png"), overlay)

    # 3) RGB re-fusion: views -> fisheye should reproduce the photo
    fused, coverage = fuse_views_to_fisheye(
        [p.astype(np.float32) for p in persps], views, cam, view_valids=valids)
    fused_u8 = np.clip(fused, 0, 255).astype(np.uint8)
    m = coverage > 0
    mse = float(np.mean((fused[m].astype(np.float64)
                         - rgb[m].astype(np.float64)) ** 2))
    psnr = 10.0 * math.log10(255.0 ** 2 / max(mse, 1e-9))
    diff = np.abs(fused.astype(np.int16) - rgb.astype(np.int16)).max(axis=-1)
    diff[~m] = 0
    side = np.concatenate([rgb, fused_u8,
                           colorize(diff.astype(np.float32), 60.0)], axis=1)
    save(os.path.join(out_dir, "D_refusion_rgb_fused_diff.png"), side)
    save(os.path.join(out_dir, "D_coverage.png"),
         colorize(coverage.astype(np.float32), float(coverage.max())))
    print(f"    RGB re-fusion PSNR (inside coverage): {psnr:.2f} dB "
          f"(double-bilinear resampling; >~24 dB expected on real photos)")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="fisheye geometry checks")
    ap.add_argument("--adt-root", default=None,
                    help="optional: run visual test D on a real ADT frame")
    ap.add_argument("--rgb-subdir", default="videos_rgb",
                    help="RGB stream for test D (videos_rgb needs no GT alignment)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_PKG, "outputs", "fisheye_checks"))
    ap.add_argument("--size", type=int, default=704,
                    help="synthetic fisheye frame size (native geometry, scaled)")
    args = ap.parse_args()

    cam = aria_intrinsics(args.size, args.size, rotated=True)
    views = base_views()
    print(f"camera: {args.size}x{args.size}, theta_max = "
          f"{math.degrees(cam.theta_max()):.2f} deg;  {len(views)} base views")

    ok_a = test_a_ray_roundtrip(cam, views)
    ok_b = test_b_synthetic_views(cam, views, args.out)
    ok_c = test_c_fusion_roundtrip(cam, views, args.out)
    if args.adt_root:
        test_d_real_frame(cam, views, args.adt_root, args.rgb_subdir,
                          args.frame, args.out)

    all_ok = ok_a and ok_b and ok_c
    print(f"\n== geometry checks: {PASS if all_ok else FAIL} ==")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
