"""Is the fisheye rim penalty partly an artefact of input ORIENTATION?

Aria stores its RGB frames a quarter turn off upright, and this repo fed them to
the backbone that way for six hypotheses (see
`autoresearch/experiments/common/upright.py`). Correcting it on DA3-Small cost
-64% whole-image AbsRel and -71% near-rim, and -- the part that matters -- the
near_rim/centre ratio fell from 4.30x to 2.00x.

That raises a question bigger than one bug: **how much of the radial error field
that the wide-FOV depth literature reports is the same artefact?** Every
egocentric Aria dataset ships frames in sensor orientation. If the effect
generalises across backbones, then "the model is worse at the rim" is partly
"the model was shown a sideways picture", and any radially-honest benchmark has
to control orientation the way it controls the depth-vs-eccentricity confound.

The measurement is the same four-way turn for every backbone, with the
prediction rotated BACK so every cell is scored on identical pixels, GT and
masks, and with the backbone installed in its native `z` so no cos(theta) is
applied to a rotated prediction.

Reported per backbone: whole-image AbsRel, near_rim, centre, and the
**near_rim / centre ratio**, which is the quantity this project's whole
diagnosis line is built on.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--models", default="da3:small,da3:large,vggt,vggt_omega")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--omega-ckpt",
                   default="/user/f.zhang2/projects/vggt-omega-organized/checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    theta = cam.incidence_grid(a.size, a.size)
    cos_t = torch.cos(theta)
    cone = (theta <= cam.theta_max).numpy()
    th_np = theta.numpy()
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(th_np, t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
    NB = len(EDG) - 1

    def zones(tab, cnt):
        out = {}
        for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                         ("center", lambda i, j: t_mid[i] <= 11),
                         ("all", lambda i, j: True)):
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
            w = np.array([cnt[i, j] for i, j in cells], float)
            if w.sum() == 0:
                continue
            out[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / w.sum())
        return out

    from raytun3r.backbones import build_backbone
    results = {}
    print(f"{'backbone':16s}{'rot':>5s}{'all':>9s}{'near_rim':>10s}{'center':>9s}{'rim/ctr':>9s}")
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        # vggt_omega's weights are a gated checkpoint, not a hub name: passing
        # "pretrained" makes it look for a file called `pretrained`.
        w = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w, device=a.device, **kw)
        except Exception as exc:
            print(f"{spec:16s}  unavailable: {exc.__class__.__name__}: {exc}")
            continue
        bb.install(None, cam, (a.size, a.size), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")
        results[spec] = {}
        for k in (0, 1, 2, 3):
            s_ = np.zeros((THETA_BINS, NB)); n_ = np.zeros((THETA_BINS, NB))
            for f in s.frames:
                with torch.no_grad():
                    z = U.forward_z(bb, s.src.image(f).to(a.device), k)
                d = (z.cpu() / cos_t.clamp_min(1e-6)).numpy()
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
            z_ = zones(tab, n_)
            z_["rim_over_center"] = (z_["near_rim"] / z_["center"]
                                     if z_.get("center") else float("nan"))
            results[spec][k] = z_
            print(f"{spec:16s}{k * 90:>5d}{z_['all']:>9.4f}{z_['near_rim']:>10.4f}"
                  f"{z_['center']:>9.4f}{z_['rim_over_center']:>9.2f}")
        bb.remove()
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": len(s.frames), "results": results,
             "config": vars(a)}, indent=1))
        print(f"[h16] wrote {a.out}")


if __name__ == "__main__":
    main()
