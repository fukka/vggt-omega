"""H17.2 — is the roll prior a property of the pretraining data, or of depth?

arXiv 2608.00678 finds the horizontal prior on four single-image depth models
(Marigold, GenPercept, DAv2, DistillAD) and attributes it to the orientation
long tail in photo collections. If that attribution is right, backbones
pretrained on MULTI-VIEW geometry -- VGGT and VGGT-Omega, which see cameras at
many orientations by construction -- should be measurably less roll-sensitive
than the single-image DA3 variants, even where their absolute accuracy is
better.

Only the clean arm from `h16-orientation/code/roll_controls.py` is run here: a
co-axial pinhole view whose VIRTUAL CAMERA is rolled and re-imaged, so the model
never sees a black border at any angle and the comparison is not contaminated by
the boundary sensitivity measured in h16 (a hard mask over 1.7% of the cone
costs as much as a 20 deg roll).

Two per-backbone details that are derived, not guessed:

* the view is sized to a multiple of the backbone's own patch size (VGGT-Omega
  is patch 16, the rest are patch 14), keeping the field of view fixed rather
  than the pixel count;
* every backbone is scored on the SAME theta <= 44 deg disc, so the pixel set
  does not move between models or between angles.

Reported per backbone: AbsRel at each angle, and the rise relative to that
backbone's own 0 deg -- absolute values are not comparable across backbones
(different capacity), the rise is.
"""
from __future__ import annotations

import argparse
import json
import math
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

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--models", default="da3:small,da3:large,vggt,vggt_omega")
    p.add_argument("--omega-ckpt", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--angles", default="-40,-30,-20,-10,0,10,20,30,40")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--view-fov", type=float, default=89.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=44.0)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    H = W = a.size
    theta = cam.incidence_grid(H, W)
    cos_t = torch.cos(theta)
    theta_np = theta.numpy()
    cone = (theta <= cam.theta_max).numpy()
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
    NB = len(EDG) - 1
    common = cone & (np.rad2deg(theta_np) <= a.common_theta_deg)
    gts = {f: s.gt_range(f, cos_t).numpy() for f in s.frames}
    angles = [float(x) for x in a.angles.split(",")]

    def zones(preds, mask):
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
        tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
        out = {}
        for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                         ("center", lambda i, j: t_mid[i] <= 11),
                         ("all", lambda i, j: True)):
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
            w = np.array([n_[i, j] for i, j in cells], float)
            out[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / max(w.sum(), 1.0))
        out["rim_over_center"] = out["near_rim"] / max(out["center"], 1e-9)
        return out

    from raytun3r.backbones import build_backbone
    results = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w, device=a.device, **kw)
        except Exception as exc:
            print(f"[roll] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
            continue
        # Same FOV for every backbone; the pixel count moves to stay patch-aligned.
        ps = bb.patch_size
        vs = int(round(a.view_size / ps)) * ps
        print(f"[roll] {spec}: patch {ps}, view {vs}x{vs} @ {a.view_fov} deg")
        results[spec] = {"view_size": vs, "patch": ps, "angles": {}}
        for deg in angles:
            rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs, height=vs,
                                             roll_deg=deg)], patch=ps)
            v = rig.views[0]
            bb.install(None, v.pin, (vs, vs), patch_undistort=False,
                       border_token=False, dpt_grid=False, depth_convention="z")
            cov = rig.covered.numpy()
            preds = {}
            for f in s.frames:
                with torch.no_grad():
                    d, _ = rig.teach(lambda warped, _view: U.forward_z(bb, warped),
                                     s.src.image(f).to(a.device), align=False)
                preds[f] = np.where(cov, d.float().cpu().numpy(), 0.0)
            r = zones(preds, cone & cov & common)
            r["fill"] = rig.fill_fraction
            results[spec]["angles"][str(deg)] = r
            print(f"       {deg:>5.0f}deg  all {r['all']:.4f}  rim {r['near_rim']:.4f}  "
                  f"ctr {r['center']:.4f}  rim/ctr {r['rim_over_center']:.2f}  fill {r['fill']:.3f}")
        del bb
        torch.cuda.empty_cache()

    print(f"\n{'backbone':<14s}{'0deg all':>10s}" + "".join(f"{'+/-'+str(int(d)):>10s}"
          for d in (10, 20, 30, 40)) + "   (rise vs its own 0 deg, all-image AbsRel)")
    for spec, r in results.items():
        A = r["angles"]
        if "0.0" not in A:
            continue
        base = A["0.0"]["all"]
        row = []
        for d in (10, 20, 30, 40):
            vals = [A[k]["all"] for k in (f"{d}.0", f"-{d}.0") if k in A]
            row.append(f"{100 * (np.mean(vals) / base - 1):>9.0f}%" if vals else "        -")
        print(f"{spec:<14s}{base:>10.4f}" + "".join(row))
    print("\n[roll] locked prediction: both VGGT variants rise LESS than DA3-Small "
          "at 30 deg (DA3-Small is +48%); prediction is under +30%.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": len(s.frames), "models": results,
             "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
