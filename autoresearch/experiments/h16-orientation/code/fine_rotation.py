"""How much does being off-upright by theta degrees actually cost?

The quarter-turn sweep says k=3 is upright and k=0 costs -64%. It does not say
what happens in between, and in between is where a head-mounted camera lives:
the wearer's head rolls continuously, so a FIXED correction is exact only when
the head is level. Whether a per-frame gravity alignment is worth building
depends entirely on the shape of this curve.

Rotation is about the PRINCIPAL POINT, not the frame centre -- Aria's is 4.5 px
off centre at 504, and rotating about the wrong point translates the imaged disc
instead of rolling it. The prediction is rotated back through the same
transform, so every angle is scored on the same pixels.

`--resample-control` runs 0 degrees through the same interpolator, which
separates "the model dislikes being rolled" from "the model dislikes resampled
pixels". Without it a smooth fall-off could be either.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "common"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402


def roll_grid(size: int, cx: float, cy: float, deg: float) -> torch.Tensor:
    """Sampling grid for a rotation by ``deg`` about ``(cx, cy)``."""
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    x, y = xs - cx, ys - cy
    sx, sy = ca * x + sa * y + cx, -sa * x + ca * y + cy
    return torch.stack((2 * (sx + 0.5) / size - 1, 2 * (sy + 0.5) / size - 1), dim=-1)


def warp(t: torch.Tensor, g: torch.Tensor, mode="bilinear") -> torch.Tensor:
    x = t[None] if t.dim() == 3 else t[None, None]
    out = F.grid_sample(x, g[None].to(x.device, x.dtype), mode=mode,
                        padding_mode="zeros", align_corners=False)
    return out[0] if t.dim() == 3 else out[0, 0]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--angles", default="-40,-30,-20,-10,0,10,20,30,40")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--variant", default="small")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    theta = cam.incidence_grid(a.size, a.size)
    cos_t = torch.cos(theta)
    cone = (theta <= cam.theta_max).numpy()
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
    NB = len(EDG) - 1

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)
    bb.install(None, cam, (a.size, a.size), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="z")

    angles = [float(x) for x in a.angles.split(",")]
    out = {}
    print(f"{'roll off upright':>18s}{'all':>10s}{'near_rim':>11s}{'center':>10s}{'rim/ctr':>9s}")
    for deg in angles:
        g_in = roll_grid(a.size, float(cam.cx), float(cam.cy), deg).to(a.device)
        g_out = roll_grid(a.size, float(cam.cx), float(cam.cy), -deg).to(a.device)
        s_ = np.zeros((THETA_BINS, NB)); n_ = np.zeros((THETA_BINS, NB))
        for f in s.frames:
            img = U.to_model(s.src.image(f).to(a.device))       # the fixed quarter turn
            if deg != 0.0:
                img = warp(img, g_in)
            with torch.no_grad():
                z = bb.forward(img[None, None]).depth[0]
            if deg != 0.0:
                z = warp(z, g_out)
            z = U.from_model(z).cpu()
            d = (z / cos_t.clamp_min(1e-6)).numpy()
            gt = s.gt_range(f, cos_t).numpy()
            v = cone & (gt > 0) & (gt <= a.depth_max_m) & (d > 1e-6)
            if v.sum() < 1000:
                continue
            al = align_depth(d, gt, v, mode="scale_shift")
            ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[v]
            di = np.clip(np.digitize(gt[v], EDG) - 1, 0, NB - 1)
            flat = t_idx[v] * NB + di
            s_ += np.bincount(flat, weights=ar, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
            n_ += np.bincount(flat, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
        tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
        z_ = {}
        for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                         ("center", lambda i, j: t_mid[i] <= 11),
                         ("all", lambda i, j: True)):
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
            w = np.array([n_[i, j] for i, j in cells], float)
            z_[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / w.sum())
        z_["rim_over_center"] = z_["near_rim"] / z_["center"]
        out[str(deg)] = z_
        print(f"{deg:>17.0f}°{z_['all']:>10.4f}{z_['near_rim']:>11.4f}"
              f"{z_['center']:>10.4f}{z_['rim_over_center']:>9.2f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"seq": s.name, "frames": len(s.frames),
                                           "angles": out, "config": vars(a)}, indent=1))
        print(f"[h16] wrote {a.out}")


if __name__ == "__main__":
    main()
