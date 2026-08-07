"""Verify the ground-truth depth path end to end, on a real scene's camera.

Tab. 3 (``AbsRel``/``delta_1.25``) is the one reproduction target the frame-span
degeneracy cannot reach -- it is per-pixel and single-frame, so no pair sampling
enters it. That makes the ground-truth depth path load-bearing, and it was wrong:
``ScanNetPPFisheye.depth`` returned ScanNet++'s **z-buffer** ``render_depth``
unconverted, while predictions are euclidean range (decision 4). The two differ
by a per-pixel ``1/cos(theta)`` -- 10.9x at 3f15a9266d's 84.8 deg rim -- and
``align_scale`` fits one global scalar, which cannot absorb it.

Measured on 3f15a9266d before the fix, a *perfect* range predictor scored
``AbsRel`` 0.426 / ``delta_1.25`` 0.412 -- worse than the paper's worst reported
method (vanilla, 0.282 / 0.601). Tab. 3 was unmeasurable before any backbone ran.

This synthesises a ``render_depth`` for a scene at known constant euclidean
range, writes it the way the toolkit does (uint16 mm z-buffer, native
resolution), reads it back through the real loader and the real camera, and
scores it with ``depth_metrics`` -- the function ``eval.py`` calls. A perfect
predictor must come back at ``AbsRel`` ~ 0.

Unit coverage lives in ``tests/test_raytun3r.py``
(``test_scannetpp_render_depth_is_planar_z_and_gets_converted``); this exists
because that test builds a synthetic camera, and the residual floor is a property
of the *real* lens and its rendered resolution.

    python -m raytun3r.experiments.depth_gt_verify --path <scene>

When ticket #11 delivers real ``render_depth``, run this first: it fails loudly
if the delivered maps are in a convention the loader does not expect.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from ..data import ScanNetPPFisheye
from ..metrics import depth_metrics

#: Tab. 3 left, ScanNet++, DA3-Small -- the scale any artifact must be read against.
PAPER = {"vanilla": (0.282, 0.601), "center_ph": (0.066, 0.961),
         "raytun3r": (0.108, 0.886)}


def _stage(real: str, tmp: str) -> str:
    """A scene sharing the real transforms/images, with a writable depth dir."""
    scene = os.path.join(tmp, os.path.basename(real.rstrip("/")))
    os.makedirs(os.path.join(scene, "dslr"))
    for sub in ("nerfstudio", "resized_images", "undistorted_images"):
        s = os.path.join(real, "dslr", sub)
        if os.path.isdir(s):
            os.symlink(s, os.path.join(scene, "dslr", sub))
    os.makedirs(os.path.join(scene, "dslr", "render_depth"))
    return scene


def main(argv=None) -> int:
    p = argparse.ArgumentParser("raytun3r.experiments.depth_gt_verify",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="a ScanNet++ scene directory")
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--range-m", type=float, default=3.0,
                   help="the constant euclidean range to synthesise")
    p.add_argument("--tol", type=float, default=0.01,
                   help="AbsRel a perfect predictor may still score")
    args = p.parse_args(argv)

    tmp = tempfile.mkdtemp(prefix="depth_gt_verify_")
    try:
        from PIL import Image

        scene = _stage(args.path, tmp)
        src = ScanNetPPFisheye(scene, max_size=args.max_size)
        omega = src.camera.valid_mask(src.h, src.w)
        cos_w = src.camera.ray_grid(src.h, src.w)[..., 2]

        # The renderer works at native resolution, so build z there.
        full = src.camera.resized(src.orig_w, src.orig_h)
        cos_o = full.ray_grid(src.orig_h, src.orig_w)[..., 2]
        z_mm = np.round((args.range_m * cos_o).numpy() * 1000.0).astype(np.uint16)

        stem = os.path.splitext(os.path.basename(src.frames[0]["file_path"]))[0]
        Image.fromarray(z_mm, mode="I;16").save(
            os.path.join(scene, "dslr", "render_depth", stem + ".png"))

        got = src.depth(0)
        if got is None:
            print("[FAIL] loader found no render_depth it had just been given")
            return 1
        d, valid = got
        rng = torch.full((src.h, src.w), args.range_m)
        m = depth_metrics(rng, d, valid=valid & omega, max_depth=None)

        print(f"scene        {src.name}   {src.w}x{src.h}   "
              f"native {src.orig_w}x{src.orig_h}")
        print(f"theta_max    {math.degrees(src.camera.theta_max):.1f} deg   "
              f"1/cos at the rim of Omega {1.0 / float(cos_w[omega].min()):.2f}x")
        print(f"convention   {src.depth_convention}   scored {m['n']} px")
        print()
        print("a perfect euclidean-range predictor, through the real loader:")
        print(f"    AbsRel {m['AbsRel']:.4f}    delta_1.25 {m['delta_1.25']:.4f}"
              f"    scale {m['scale']:.4f}")
        print()
        print("    for scale -- paper Tab. 3 left, ScanNet++, DA3-Small:")
        for k, (a, dl) in PAPER.items():
            print(f"      {k:10s} AbsRel {a:.3f}   delta_1.25 {dl:.3f}")

        ok = m["AbsRel"] < args.tol and m["delta_1.25"] > 0.99
        print()
        if ok:
            print(f"VERDICT: PASS -- the residual {m['AbsRel']:.4f} is the "
                  f"resample-plus-quantisation floor, "
                  f"{m['AbsRel'] / PAPER['center_ph'][0] * 100:.1f}% of the "
                  f"tightest target (Center-PH 0.066).")
        else:
            print(f"VERDICT: FAIL -- the loader injects {m['AbsRel']:.4f} AbsRel "
                  f"on a perfect predictor. Tab. 3 is not measurable; check the "
                  f"planar-z -> range conversion in ScanNetPPFisheye.depth.")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
