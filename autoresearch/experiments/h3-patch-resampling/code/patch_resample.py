"""H3: zero-parameter local patch undistortion on DA3-Small, seq131.

Protocol: ../protocol.md (committed before this ran). CPU.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h3-patch-resampling/code/patch_resample.py \
        --out results/run_014.json
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
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ    # noqa: E402
from finetune.eval.metrics import align_depth             # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
PATCH = 14


def resample_grid(camera, h: int, w: int) -> torch.Tensor:
    """(H, W, 2) sampling coordinates implementing per-patch local gnomonic
    resampling: identity at each patch center, locally pinhole-like content."""
    gh, gw = h // PATCH, w // PATCH
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                            torch.arange(w, dtype=torch.float64), indexing="ij")
    out_u = xs.clone()
    out_v = ys.clone()
    for pi in range(gh):
        for pj in range(gw):
            v0 = pi * PATCH + (PATCH - 1) / 2.0
            u0 = pj * PATCH + (PATCH - 1) / 2.0
            d0 = camera.unproject(torch.tensor([[u0, v0]], dtype=torch.float64))[0]
            theta0 = math.acos(float(d0[2].clamp(-1, 1)))
            if theta0 > float(camera.theta_max):
                continue                       # outside the cone: leave as-is
            # Local linearization of the projection: find tangent directions
            # that the projection's differential maps to the PIXEL axes, so the
            # resampling is identity to first order at the patch center — same
            # orientation, correct anisotropic scale (the first implementation
            # used an arbitrary tangent basis and rotated every patch by its
            # azimuth; caught by the protocol's visual sanity check, run_014).
            up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
            a1 = torch.linalg.cross(up, d0)
            if a1.norm() < 1e-8:
                a1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            a1 = a1 / a1.norm()
            a2 = torch.linalg.cross(d0, a1)
            eps = 1e-5
            base = camera.project(d0[None])[0]
            J = torch.empty(2, 2, dtype=torch.float64)
            for k, ak in enumerate((a1, a2)):
                dk = d0 + eps * ak
                dk = dk / dk.norm()
                J[:, k] = (camera.project(dk[None])[0] - base) / eps
            Jinv = torch.linalg.inv(J)          # pixel offset -> tangent coords
            ys_l = (torch.arange(PATCH, dtype=torch.float64) - (PATCH - 1) / 2.0)
            gy, gx = torch.meshgrid(ys_l, ys_l.clone(), indexing="ij")
            pix = torch.stack([gx, gy], dim=-1)          # (P, P, 2) pixel offsets
            tc = pix @ Jinv.T                            # tangent coords (rad)
            dirs = (d0[None, None, :]
                    + tc[..., 0:1] * a1[None, None, :]
                    + tc[..., 1:2] * a2[None, None, :])
            dirs = dirs / dirs.norm(dim=-1, keepdim=True)
            uv = camera.project(dirs.reshape(-1, 3)).reshape(PATCH, PATCH, 2)
            sl_v = slice(pi * PATCH, (pi + 1) * PATCH)
            sl_u = slice(pj * PATCH, (pj + 1) * PATCH)
            out_u[sl_v, sl_u] = uv[..., 0]
            out_v[sl_v, sl_u] = uv[..., 1]
    gx = 2.0 * out_u / (w - 1) - 1.0
    gy = 2.0 * out_v / (h - 1) - 1.0
    return torch.stack([gx, gy], dim=-1).float()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--save-example", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = AriaLocalPairs(os.path.expanduser(args.seq), size=args.size)
    theta = src.camera.incidence_grid(src.h, src.w)
    cone = (theta <= src.camera.theta_max).numpy()
    cos_t = torch.cos(theta).numpy()
    t_edges = np.linspace(0.0, float(src.camera.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)

    print("[h3] building resample grid (exact KB4, per patch)...")
    grid = resample_grid(src.camera, src.h, src.w)

    # sanity: center patch identity
    ci, cj = src.h // 2, src.w // 2
    ident = torch.stack(torch.meshgrid(
        torch.arange(src.h, dtype=torch.float32),
        torch.arange(src.w, dtype=torch.float32), indexing="ij")[::-1], -1)
    ident = torch.stack([2 * ident[..., 0] / (src.w - 1) - 1,
                         2 * ident[..., 1] / (src.h - 1) - 1], -1)
    center_dev = (grid[ci - 3:ci + 3, cj - 3:cj + 3]
                  - ident[ci - 3:ci + 3, cj - 3:cj + 3]).abs().max()
    print(f"[h3] center-patch deviation from identity: "
          f"{float(center_dev) * src.w / 2:.3f} px (should be ~0)")

    def resample(img: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            img[None], grid[None], mode="bilinear", padding_mode="border",
            align_corners=True)[0]

    if args.save_example:
        from PIL import Image
        im = src.image(0)
        both = torch.cat([im, resample(im)], dim=2)
        Image.fromarray((both.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        ).save(args.save_example)
        print(f"[h3] wrote example {args.save_example}")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu",
                        variant="small")
    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")

    depth_paths = {os.path.basename(q).replace(".npy", ""): q for q in
                   glob.glob(os.path.join(os.path.expanduser(args.seq),
                                          "depth_npy", "*.npy"))}
    nb_d = len(GT_DEPTH_EDGES) - 1
    acc = {arm: [np.zeros((THETA_BINS, nb_d)),
                 np.zeros((THETA_BINS, nb_d), dtype=np.int64)]
           for arm in ("vanilla", "resampled")}
    for n, path in enumerate(src.paths):
        stem = os.path.basename(path).replace(".jpg", "")
        if stem not in depth_paths:
            continue
        gz = np.load(depth_paths[stem]).astype(np.float32)
        gz = torch.nn.functional.interpolate(
            torch.from_numpy(gz)[None, None], size=(src.h, src.w),
            mode="nearest")[0, 0].numpy() / 1000.0
        gr = gz / np.clip(cos_t, 1e-6, None)
        t0 = time.time()
        for arm in ("vanilla", "resampled"):
            img = src.image(n)
            if arm == "resampled":
                img = resample(img)
            with torch.no_grad():
                pred = bb.forward(img[None, None])
            pred.require_convention("range")
            d = pred.depth[0].numpy()
            valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
            flat = ti * nb_d + di
            acc[arm][0] += np.bincount(flat, weights=absrel,
                                       minlength=THETA_BINS * nb_d
                                       ).reshape(THETA_BINS, nb_d)
            acc[arm][1] += np.bincount(flat, minlength=THETA_BINS * nb_d
                                       ).reshape(THETA_BINS, nb_d)
        print(f"  frame {n + 1}/{len(src.paths)} ({time.time() - t0:4.1f}s)",
              flush=True)

    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    tables = {}
    for arm in ("vanilla", "resampled"):
        tables[arm] = (acc[arm][0] / np.maximum(acc[arm][1], 1))
    print("\njoint AbsRel vanilla > resampled (rows=GT depth, cols=theta):")
    for j in range(nb_d):
        row = " ".join(f"{tables['vanilla'][i, j]:5.3f}>"
                       f"{tables['resampled'][i, j]:5.3f}"
                       for i in range(THETA_BINS))
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")
    marg_v = acc["vanilla"][0].sum(1) / np.maximum(acc["vanilla"][1].sum(1), 1)
    marg_r = acc["resampled"][0].sum(1) / np.maximum(acc["resampled"][1].sum(1), 1)
    print("\nper-theta marginal: " + " ".join(
        f"{t:.0f}deg {v:.3f}>{r:.3f}" for t, v, r in zip(t_mid, marg_v, marg_r)))

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"theta_bin_mid_deg": t_mid,
                       "vanilla": tables["vanilla"].tolist(),
                       "resampled": tables["resampled"].tolist(),
                       "counts": acc["vanilla"][1].tolist(),
                       "marginal_vanilla": marg_v.tolist(),
                       "marginal_resampled": marg_r.tolist(),
                       "center_identity_px": float(center_dev) * src.w / 2,
                       "config": vars(args)}, f, indent=2)
        print(f"\n[h3] wrote {dst}")


if __name__ == "__main__":
    main()
