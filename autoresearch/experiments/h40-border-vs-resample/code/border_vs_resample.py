"""H40 - border x resampling, 2x2, at a freely chosen angle with no border.

The 89 degree square view used elsewhere in this line has a corner ray of 62.9
deg and already runs off Aria's 54.83 deg cone, which is why rotating it creates
black corners. A 60 degree view has a corner ray of 42.4 deg and never leaves
the cone, so it can be rotated by any angle with NO border at all. Its inscribed
disc is at 30 deg, so scoring is capped there and is invariant under rotation.

Four arms, one angle:

    direct      no rotation, no mask     baseline
    rt          rotate -d then +d        resampling alone
    masked      inscribed disc mask      border alone
    rt_masked   both                     both

`rt` shows the backbone the same content as `direct` up to two resamplings: no
roll is removed or added, and nothing leaves the cone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "common"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h39-gravity-render" / "code"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402
from gravity_render import rot  # noqa: E402  (same grid-rotation helper)


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402

ARMS = ("direct", "rt", "masked", "rt_masked")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--delta-deg", type=float, default=30.0)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--models", default="da3:small,vggt_omega")
    p.add_argument("--omega-ckpt", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--view-fov", type=float, default=60.0,
                   help="60 deg keeps the corner ray at 42.4 deg, inside Aria's "
                        "54.83 deg cone, so rotation never creates a border")
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=29.0,
                   help="inside the 60 deg view's inscribed disc (30 deg)")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    theta = cam.incidence_grid(a.size, a.size)
    cos_t = torch.cos(theta)
    theta_np = theta.numpy()
    cone = (theta <= cam.theta_max).numpy()
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
    NB = len(EDG) - 1
    common = cone & (np.rad2deg(theta_np) <= a.common_theta_deg)
    gts = {f: s.gt_range(f, cos_t).numpy() for f in s.frames}

    from raytun3r.backbones import build_backbone

    def all_absrel(preds, mask):
        s_ = np.zeros((THETA_BINS, NB)); n_ = np.zeros((THETA_BINS, NB))
        for f, d in preds.items():
            gt = gts[f]
            v = mask & (gt > 0) & (gt <= a.depth_max_m) & (d > 1e-6)
            if v.sum() < 1000:
                continue
            al = align_depth(d, gt, v, mode="scale_shift")
            ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[v]
            di = np.clip(np.digitize(gt[v], EDG) - 1, 0, NB - 1)
            flat = t_idx[v] * NB + di
            s_ += np.bincount(flat, weights=ar, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
            n_ += np.bincount(flat, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
        return float(s_.sum() / max(n_.sum(), 1.0))

    out = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w8, device=a.device, **kw)
        except Exception as exc:
            print(f"[h40] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
            continue
        ps = bb.patch_size
        vs = int(round(a.view_size / ps)) * ps
        c = (vs - 1) / 2.0
        yy, xx = torch.meshgrid(torch.arange(vs, dtype=torch.float32),
                                torch.arange(vs, dtype=torch.float32), indexing="ij")
        disc = ((((xx - c) ** 2 + (yy - c) ** 2) <= (vs / 2.0 - 1.0) ** 2)
                .to(a.device))
        rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                         height=vs, roll_deg=0.0)], patch=ps)
        bb.install(None, rig.views[0].pin, (vs, vs), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")
        cov = rig.covered.numpy()
        mask = cone & cov & common

        # --- construction check: the rotated view must contain no black -----
        probe = {}
        with torch.no_grad():
            def grab(w, _v):
                probe["img"] = w.detach().clone()
                return torch.ones(w.shape[-2:], device=w.device)
            rig.teach(grab, s.src.image(s.frames[0]).to(a.device), align=False)
        img = probe["img"]
        blk0 = float((img.abs().sum(0) < 1e-6).float().mean())
        blkr = float((rot(img, -a.delta_deg).abs().sum(0) < 1e-6).float().mean())
        print(f"[h40] {spec}: view {vs}x{vs} @ {a.view_fov} deg  "
              f"black fraction: direct {blk0:.4f}, rotated {blkr:.4f}"
              f"{'   <-- NOT BORDER-FREE' if blkr > 0.002 else ''}", flush=True)

        R = {}
        for arm in ARMS:
            preds = {}
            for f in s.frames:
                def fz(w, _v, _a=arm):
                    if _a == "direct":
                        return U.forward_z(bb, w)
                    if _a == "masked":
                        return U.forward_z(bb, w * disc)
                    x = rot(rot(w, -a.delta_deg), a.delta_deg)
                    return U.forward_z(bb, x * disc if _a == "rt_masked" else x)
                with torch.no_grad():
                    d, _ = rig.teach(fz, s.src.image(f).to(a.device), align=False)
                preds[f] = np.where(cov, d.float().cpu().numpy(), 0.0)
            R[arm] = all_absrel(preds, mask)
        base = R["direct"]
        cost = {k: 100 * (R[k] / base - 1) for k in ARMS}
        add = 100 * ((1 + cost["rt"] / 100) * (1 + cost["masked"] / 100) - 1)
        out[spec] = {"absrel": R, "cost_pct": cost, "multiplicative_pct": add,
                     "black_direct": blk0, "black_rotated": blkr}
        print(f"  direct {base:.4f} | resampling {cost['rt']:+.2f}% | "
              f"border {cost['masked']:+.2f}% | both {cost['rt_masked']:+.2f}% "
              f"(multiplicative would be {add:+.2f}%)", flush=True)
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": len(s.frames), "models": out,
             "config": vars(a)}, indent=1))
        print(f"[h40] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
